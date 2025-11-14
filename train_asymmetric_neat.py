"""
Pipeline de Treinamento Assimétrico com NEAT (NeuroEvolution of Augmenting Topologies)

Estratégia:
- MacroNet NEAT: Rede que evolui topologia (long-term, 41h)
- MicroNet NEAT: Rede que evolui topologia (short-term, 5h)
- Evolução assimétrica: Macro evolui 1x a cada 10 episódios, Micro evolui 1x por episódio
- Ratio: 1:10 (extremamente ágil, seguindo padrão do RL)
- Vantagens: topologias adaptadas ao problema, sem precisar definir arquitetura manualmente

Fluxo (seguindo padrão RL):
1. SEMPRE avalia macro (fornece contexto estratégico)
2. SEMPRE avalia micro usando melhor macro (recebe contexto)
3. Evolui apenas as redes indicadas pelo ratio (macro 1x : micro 10x)

Componentes NEAT:
1. Genomas: representam topologia (nós, conexões, pesos)
2. População: múltiplos indivíduos evoluindo em paralelo
3. Fitness: calculado sobre episódios de trading
4. Especiação: agrupa genomas similares para preservar inovações
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Callable
import matplotlib.pyplot as plt
import pickle
import os
import tempfile

# NEAT imports
import neat
from neat.parallel import ParallelEvaluator
from multiprocessing import Pool, cpu_count
from functools import partial

from src.pipeline import TradingPipeline
from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# FUNÇÕES TOP-LEVEL PARA MULTIPROCESSING (devem estar fora de classes)
# ============================================================================

def evaluate_macro_genome_worker(genome_data, config_macro, envs_data, max_steps=200):
    """
    Função worker para avaliar um genoma macro em paralelo.
    Deve ser top-level para ser picklável.
    """
    genome_id, genome = genome_data
    
    # Recriar ambientes a partir dos dados
    envs = []
    for env_data in envs_data:
        env = TradingEnvironmentRL(
            prices=env_data['prices'],
            macro_features=env_data['macro_features'],
            micro_features=env_data['micro_features'],
            initial_capital=env_data['initial_capital'],
            commission=env_data['commission']
        )
        envs.append(env)
    
    # Criar rede NEAT uma vez (otimização!)
    net = neat.nn.FeedForwardNetwork.create(genome, config_macro)
    
    total_reward = 0.0
    num_envs = 0
    
    for env in envs:
        state = env.reset()
        steps = 0
        episode_reward = 0.0
        
        while steps < max_steps:
            # Forward pass
            macro_output = net.activate(state['macro_features'])
            macro_output = np.asarray(macro_output, dtype=np.float32)
            
            # Usar saída bruta da rede como previsão (-1 a +1)
            prediction_value = float(macro_output[0])
            
            # Escolher ação (mantido para compatibilidade, mas não usado no reward)
            action_logits = macro_output[:3]
            if action_logits.shape[0] < 3:
                action_logits = np.pad(action_logits, (0, 3 - action_logits.shape[0]), constant_values=0.0)
            
            if np.allclose(action_logits, action_logits[0], atol=1e-6):
                action = np.random.randint(0, 3)
            else:
                action = int(np.argmax(action_logits))
            
            next_state, reward, done = env.step(action, prediction_value)
            episode_reward += reward
            steps += 1
            
            if done or next_state is None:
                break
            
            state = next_state
        
        total_reward += episode_reward
        num_envs += 1
    
    fitness = total_reward / max(1, num_envs)
    genome.fitness = fitness
    return genome_id, genome, fitness


def evaluate_micro_genome_worker(genome_data, config_micro, best_macro_genome, config_macro, envs_data, max_steps=200):
    """
    Função worker para avaliar um genoma micro em paralelo.
    """
    genome_id, genome = genome_data
    
    # Recriar ambientes
    envs = []
    for env_data in envs_data:
        env = TradingEnvironmentRL(
            prices=env_data['prices'],
            macro_features=env_data['macro_features'],
            micro_features=env_data['micro_features'],
            initial_capital=env_data['initial_capital'],
            commission=env_data['commission']
        )
        envs.append(env)
    
    # Criar redes NEAT uma vez
    macro_net = neat.nn.FeedForwardNetwork.create(best_macro_genome, config_macro)
    micro_net = neat.nn.FeedForwardNetwork.create(genome, config_micro)
    
    total_reward = 0.0
    num_envs = 0
    
    for env in envs:
        state = env.reset()
        steps = 0
        episode_reward = 0.0
        
        while steps < max_steps:
            # Macro context
            macro_output = np.asarray(macro_net.activate(state['macro_features']), dtype=np.float32)
            
            # Micro input
            micro_input = np.concatenate([
                state['micro_features'],
                macro_output,
                [state['position'], state['cash'] / 10000.0]
            ])
            
            micro_output = np.asarray(micro_net.activate(micro_input), dtype=np.float32)
            
            # Usar saída bruta da rede como previsão (-1 a +1)
            prediction_value = float(micro_output[0])
            
            action = np.argmax(micro_output) % 3
            
            next_state, reward, done = env.step(action, prediction_value)
            episode_reward += reward
            steps += 1
            
            if done or next_state is None:
                break
            
            state = next_state
        
        total_reward += episode_reward
        num_envs += 1
    
    fitness = total_reward / max(1, num_envs)
    genome.fitness = fitness
    return genome_id, genome, fitness


class NEATNetworkAdapter:
    """
    Adapter para usar redes NEAT (que são grafos) como módulos PyTorch.
    Converte genoma NEAT em representação PyTorch para integração com otimizador.
    """
    
    def __init__(self, config: neat.Config):
        self.config = config
    
    def genome_to_tensor_network(self, genome: neat.DefaultGenome) -> Dict:
        """
        Converte genoma NEAT para representação matricial PyTorch.
        
        Retorna dicionário com:
        - weights: lista de pesos das conexões
        - connections: lista de (input_node, output_node)
        - node_ids: IDs dos nós ocultos
        """
        # Extrair conexões ativas
        connections = []
        weights = []
        
        for cg in genome.connections.values():
            if cg.enabled:
                connections.append((cg.key[0], cg.key[1], float(cg.weight)))
                weights.append(float(cg.weight))
        
        # Extrair nós e seus funções de ativação
        node_ids = list(genome.nodes.keys())
        
        return {
            'connections': connections,
            'weights': np.array(weights, dtype=np.float32),
            'node_ids': node_ids,
            'bias': {nid: genome.nodes[nid].bias for nid in node_ids if nid in genome.nodes}
        }
    
    def forward_neat(self, genome: neat.DefaultGenome, inputs: np.ndarray) -> np.ndarray:
        """
        Executa forward pass usando genoma NEAT (interpretação pura sem PyTorch).
        Mais lento mas preserva fidelidade da topologia NEAT.
        """
        feed_forward = getattr(self.config.genome_config, "feed_forward", True)
        net_cls = neat.nn.FeedForwardNetwork if feed_forward else neat.nn.RecurrentNetwork
        net = net_cls.create(genome, self.config)
        output = net.activate(inputs)
        if not feed_forward:
            # Garantir que estado interno não vaze entre chamadas
            net.reset()
        return np.asarray(output, dtype=np.float32)
    
    def forward_neat_batch(self, genome: neat.DefaultGenome, batch_inputs: np.ndarray) -> np.ndarray:
        """
        Forward batch: aplica rede NEAT para múltiplas amostras.
        
        Args:
            genome: genoma NEAT
            batch_inputs: (batch_size, input_dim)
        
        Returns:
            (batch_size, output_dim)
        """
        feed_forward = getattr(self.config.genome_config, "feed_forward", True)
        net_cls = neat.nn.FeedForwardNetwork if feed_forward else neat.nn.RecurrentNetwork
        outputs = []
        # Criar nova rede por batch para evitar reuso de estado interno antigo
        for inputs in batch_inputs:
            net = net_cls.create(genome, self.config)
            output = net.activate(inputs)
            if not feed_forward:
                net.reset()
            outputs.append(output)
        return np.array(outputs, dtype=np.float32)


class TradingEnvironmentRL:
    """Ambiente de trading simplificado para RL com NEAT"""
    
    def __init__(
        self,
        prices: np.ndarray,
        macro_features: np.ndarray,
        micro_features: np.ndarray,
        initial_capital: float = 10000.0,
        commission: float = 0.001
    ):
        self.prices = prices
        self.macro_features = macro_features
        self.micro_features = micro_features
        self.initial_capital = initial_capital
        self.commission = commission
        
        self.reset()
    
    def reset(self):
        """Reset environment"""
        self.step_idx = 0
        self.cash = self.initial_capital
        self.position = 0.0
        self.portfolio_value = self.initial_capital
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state"""
        if self.step_idx >= len(self.prices):
            return None
        
        return {
            'macro_features': self.macro_features[self.step_idx],
            'micro_features': self.micro_features[self.step_idx],
            'price': self.prices[self.step_idx],
            'position': self.position,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value
        }
    
    def step(self, action: int, prediction_value: float = 0.0) -> Tuple[dict, float, bool]:
        """
        Execute action usando valor bruto da rede como previsão.
        
        Args:
            action: ação escolhida (não usado mais)
            prediction_value: valor bruto da saída da rede (pode ser positivo ou negativo)
                             - Negativo = prevê queda
                             - Positivo = prevê alta
                             - Magnitude = confiança
        
        Fitness = (prediction_value * 100) * (price_change_pct * 100)
        
        Returns:
            next_state, reward, done
        """
        if self.step_idx >= len(self.prices) - 1:
            return None, 0.0, True
        
        current_price = self.prices[self.step_idx]
        next_price = self.prices[self.step_idx + 1]
        
        # Calculate actual price change percentage
        price_change_pct = ((next_price - current_price) / current_price) * 100
        
        # Reward = (previsão normalizada) * (mudança real normalizada)
        # Exemplo: previsão -0.5, real -0.3% → (100*-0.5)*(100*-0.3) = (-50)*(-30) = +1500
        # Exemplo: previsão -0.5, real +0.3% → (100*-0.5)*(100*+0.3) = (-50)*(30) = -1500
        reward = (prediction_value * 100) * (price_change_pct * 100)
        
        # Advance
        self.step_idx += 1
        
        # Update portfolio
        self.portfolio_value = self.cash + (self.position * next_price)
        
        done = self.step_idx >= len(self.prices) - 1
        next_state = self._get_state()
        
        return next_state, reward, done


class AsymmetricNEATTrainer:
    """
    Treinador NEAT para arquitetura assimétrica.
    
    Gerencia duas populações NEAT:
    - macro_population: Evolui mais lentamente (1 gen a cada 2 gerações da micro)
    - micro_population: Evolui mais rápido (1 gen a cada geração)
    """
    
    def __init__(
        self,
        config_macro: neat.Config,
        config_micro: neat.Config,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.config_macro = config_macro
        self.config_micro = config_micro
        
        # Populações NEAT
        self.macro_population = neat.Population(config_macro)
        self.micro_population = neat.Population(config_micro)
        
        # Adaptadores para forward pass
        self.macro_adapter = NEATNetworkAdapter(config_macro)
        self.micro_adapter = NEATNetworkAdapter(config_micro)
        
        # Histórico
        self.generation_macro = 0
        self.generation_micro = 0
        self.best_macro_fitness = -np.inf
        self.best_micro_fitness = -np.inf
        
        logger.info("asymmetric_neat_trainer_initialized")
    
    def eval_macro_genome(self, genome: neat.DefaultGenome, envs: List[TradingEnvironmentRL], max_steps: int = 200) -> float:
        """
        Avaliar fitness de um genoma MacroNet sobre múltiplos ambientes.
        
        Args:
            genome: genoma NEAT para MacroNet
            envs: lista de ambientes para teste
        
        Returns:
            fitness (média de retorno % dos ambientes)
        """
        total_return = 0.0
        num_envs = 0
        
        # Criar rede NEAT uma vez para reutilização (OTIMIZAÇÃO!)
        net = neat.nn.FeedForwardNetwork.create(genome, self.config_macro)
        
        # 🔍 Debug: coletar estatísticas de saída da rede
        all_predictions = []
        
        for env_idx, env in enumerate(envs):
            state = env.reset()
            total_reward = 0.0
            steps = 0
            
            while steps < max_steps:
                # Forward pass: usar rede já criada (mais rápido!)
                macro_output = np.asarray(net.activate(state['macro_features']), dtype=np.float32)
                
                # Usar primeiro valor da saída como previsão bruta
                # Valor negativo = prevê queda, positivo = prevê alta
                prediction_value = float(macro_output[0]) if len(macro_output) > 0 else 0.0
                all_predictions.append(prediction_value)
                
                # Action não é mais usado, mas mantemos por compatibilidade
                action = 0
                
                next_state, reward, done = env.step(action, prediction_value)
                total_reward += reward
                steps += 1
                
                if done or next_state is None:
                    break
                
                state = next_state
            
            # Fitness = média dos rewards acumulados
            total_return += total_reward
            num_envs += 1
        
        fitness = total_return / max(1, num_envs)
        avg_portfolio = 10000  # Não usado mais
        avg_step_reward = total_return / max(1, num_envs)
        return fitness, avg_portfolio, avg_step_reward

    def eval_micro_genome(
        self,
        macro_genome: neat.DefaultGenome,
        micro_genome: neat.DefaultGenome,
        envs: List[TradingEnvironmentRL],
        max_steps: int = 200
    ) -> float:
        """
        Avaliar fitness de um genoma MicroNet.
        Usa melhor MacroNet como contexto.

        Args:
            macro_genome: melhor genoma MacroNet (fixa)
            micro_genome: genoma MicroNet a avaliar
            envs: lista de ambientes

        Returns:
            fitness (média de retorno %)
        """
        total_return = 0.0
        num_envs = 0

        # Criar redes NEAT uma vez (OTIMIZAÇÃO!)
        macro_net = neat.nn.FeedForwardNetwork.create(macro_genome, self.config_macro)
        micro_net = neat.nn.FeedForwardNetwork.create(micro_genome, self.config_micro)

        # 🔍 Debug: coletar estatísticas de saída da rede
        all_predictions = []

        for env in envs:
            state = env.reset()
            total_reward = 0.0
            steps = 0

            while steps < max_steps:
                # Macro: usar rede já criada
                macro_output = np.asarray(macro_net.activate(state['macro_features']), dtype=np.float32)

                # Micro: recebe micro_features + macro_output concatenados
                micro_input = np.concatenate([
                    state['micro_features'],
                    macro_output,
                    [state['position'], state['cash'] / 10000.0]
                ])

                micro_output = np.asarray(micro_net.activate(micro_input), dtype=np.float32)

                # Usar saída bruta da rede como previsão (-1 a +1)
                prediction_value = float(micro_output[0])
                all_predictions.append(prediction_value)

                # Ação: argmax de micro output (mantido para compatibilidade, mas não usado no reward)
                action = np.argmax(micro_output) % 3

                next_state, reward, done = env.step(action, prediction_value)
                total_reward += reward
                steps += 1

                if done or next_state is None:
                    break

                state = next_state

            # Fitness = reward acumulado
            total_return += total_reward
            num_envs += 1

        fitness = total_return / max(1, num_envs)
        avg_portfolio = 10000  # Não usado mais
        avg_step_reward = total_return / max(1, num_envs)
        return fitness, avg_portfolio, avg_step_reward
    
    def evolve_generation(
        self,
        macro_genomes: Dict,
        micro_genomes: Dict,
        envs: List[TradingEnvironmentRL],
        update_macro: bool = False,
        update_micro: bool = True,
        use_multiprocessing: bool = True,
        max_steps: int = 200  # Reduzido de 500 para 200 para speedup
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Executar uma geração de evolução com MULTIPROCESSING REAL.
        
        Args:
            macro_genomes: dict {genome_id: genome} de genomas macro
            micro_genomes: dict {genome_id: genome} de genomas micro
            envs: ambientes para avaliação
            update_macro: se deve evoluir população macro
            update_micro: se deve evoluir população micro
            use_multiprocessing: usar Pool para paralelização
            max_steps: máximo de steps por episódio (reduzir para speedup)
        
        Returns:
            (best_macro_fitness, best_micro_fitness, avg_macro_portfolio, avg_micro_portfolio, avg_macro_reward, avg_micro_reward, eval_time)
        """
        
        avg_macro_portfolio = 0.0
        avg_micro_portfolio = 0.0
        avg_macro_reward = 0.0
        avg_micro_reward = 0.0
        eval_start_time = time.time()
        
        # Preparar dados dos ambientes para serialização
        envs_data = []
        for env in envs:
            envs_data.append({
                'prices': env.prices,
                'macro_features': env.macro_features,
                'micro_features': env.micro_features,
                'initial_capital': env.initial_capital,
                'commission': env.commission
            })

        # ═════════════════════════════════════════════════════════════
        # NEAT ASYMMETRIC PATTERN (CORRETO):
        # 1. SEMPRE avaliar primeiro (garante fitness para todos)
        # 2. DEPOIS reproduzir (usando fitness da avaliação)
        # Isso funciona porque após reproduce, na PRÓXIMA chamada
        # já avaliamos a nova população antes de reproduzir novamente.
        # ═══════════════════════════════════════════════════════════
        
        # Pegar população atual
        macro_genomes = self.macro_population.population
        micro_genomes = self.micro_population.population
        
        # ─────────────────────────────────────────────────────────────
        # PASSO 2: AVALIAR MACRO (sempre, nova geração precisa de fitness)
        # ─────────────────────────────────────────────────────────────
        macro_eval_start = time.time()
        total_genomes_macro = len(macro_genomes)
        
        if use_multiprocessing and total_genomes_macro > 1:
            with Pool(processes=cpu_count()) as pool:
                eval_func = partial(
                    evaluate_macro_genome_worker,
                    config_macro=self.config_macro,
                    envs_data=envs_data,
                    max_steps=max_steps
                )
                results = pool.map(eval_func, list(macro_genomes.items()))

            # Depuração: snapshot antes
            macro_ids_before = set(self.macro_population.population.keys())
            # Atualizar fitness nos genomas originais da população
            for genome_id, genome_result, fitness in results:
                try:
                    self.macro_population.population[genome_id].fitness = fitness
                except Exception:
                    pass

            # Depuração: estatísticas de fitness após avaliação
            macro_fitnesses = [g.fitness for g in self.macro_population.population.values() if g.fitness is not None]
        else:
            # Sequencial
            for idx, (gid, genome) in enumerate(macro_genomes.items(), 1):
                fitness, _, _ = self.eval_macro_genome(genome, envs, max_steps=max_steps)
                genome.fitness = fitness

        # Calcular métricas médias macro
        macro_portfolios = []
        for genome in list(macro_genomes.values())[:5]:
            _, portfolio, _ = self.eval_macro_genome(genome, envs[:2], max_steps=100)
            macro_portfolios.append(portfolio)

        if macro_genomes:
            self.best_macro_fitness = max(g.fitness for g in macro_genomes.values() if g.fitness is not None)
            avg_macro_portfolio = np.mean(macro_portfolios) if macro_portfolios else 0
        
        macro_eval_time = time.time() - macro_eval_start
        
        # ─────────────────────────────────────────────────────────────
        # PASSO 2: AVALIAR MICRO (sempre, usando melhor macro)
        # ─────────────────────────────────────────────────────────────
        micro_eval_start = time.time()
        best_macro_genome_id = max(self.macro_population.population, key=lambda g: self.macro_population.population[g].fitness or -np.inf)
        best_macro = self.macro_population.population[best_macro_genome_id]
        total_genomes_micro = len(micro_genomes)
        # Depuração: snapshot IDs micro antes da avaliação
        micro_ids_before = set(self.micro_population.population.keys())

        if use_multiprocessing and total_genomes_micro > 1:
            with Pool(processes=cpu_count()) as pool:
                eval_func = partial(
                    evaluate_micro_genome_worker,
                    config_micro=self.config_micro,
                    best_macro_genome=best_macro,
                    config_macro=self.config_macro,
                    envs_data=envs_data,
                    max_steps=max_steps
                )
                results = pool.map(eval_func, list(micro_genomes.items()))
            
            # Atualizar fitness nos genomas originais da população
            for genome_id, genome_result, fitness in results:
                try:
                    self.micro_population.population[genome_id].fitness = fitness
                except Exception:
                    pass

        else:
            # Sequencial
            for idx, (gid, genome) in enumerate(micro_genomes.items(), 1):
                fitness, _, _ = self.eval_micro_genome(best_macro, genome, envs, max_steps=max_steps)
                genome.fitness = fitness

        # Calcular métricas médias micro
        micro_portfolios = []
        for genome in list(micro_genomes.values())[:5]:
            _, portfolio, _ = self.eval_micro_genome(best_macro, genome, envs[:2], max_steps=100)
            micro_portfolios.append(portfolio)

        if micro_genomes:
            self.best_micro_fitness = max(g.fitness for g in micro_genomes.values() if g.fitness is not None)
            avg_micro_portfolio = np.mean(micro_portfolios) if micro_portfolios else 0
        
        micro_eval_time = time.time() - micro_eval_start
        
        # ─────────────────────────────────────────────────────────────
        # PASSO 3: REPRODUZIR (apenas se update_X=True)
        # Agora todos os genomas têm fitness fresh da avaliação acima
        # ─────────────────────────────────────────────────────────────
        if update_macro:
            # Limpar espécies vazias
            self.macro_population.species.species = {
                sid: s for sid, s in self.macro_population.species.species.items()
                if len(s.members) > 0
            }

            # Corrigido: Capturar a nova população retornada pela reprodução
            new_pop = self.macro_population.reproduction.reproduce(
                self.config_macro,
                self.macro_population.species,
                self.config_macro.pop_size,
                self.generation_macro
            )
            self.macro_population.population = new_pop

            # Corrigido: Re-especiar a nova população para a próxima geração
            self.macro_population.species.speciate(
                self.config_macro,
                self.macro_population.population,
                self.generation_macro
            )

            self.generation_macro += 1

        if update_micro:
            # Limpar espécies vazias
            self.micro_population.species.species = {
                sid: s for sid, s in self.micro_population.species.species.items()
                if len(s.members) > 0
            }

            # Corrigido: Capturar a nova população retornada pela reprodução
            new_pop = self.micro_population.reproduction.reproduce(
                self.config_micro,
                self.micro_population.species,
                self.config_micro.pop_size,
                self.generation_micro
            )
            self.micro_population.population = new_pop

            # Corrigido: Re-especiar a nova população para a próxima geração
            self.micro_population.species.speciate(
                self.config_micro,
                self.micro_population.population,
                self.generation_micro
            )

            self.generation_micro += 1
        
        eval_total_time = time.time() - eval_start_time
        return self.best_macro_fitness, self.best_micro_fitness, avg_macro_portfolio, avg_micro_portfolio, avg_macro_reward, avg_micro_reward, eval_total_time


class ParallelEvaluationHelper:
    """Helper class para encapsular contexto de avaliação paralela."""
    
    def __init__(self, trainer, envs, eval_type='macro', best_macro_genome=None):
        """
        Args:
            trainer: instância do AsymmetricNEATTrainer
            envs: lista de ambientes
            eval_type: 'macro' ou 'micro'
            best_macro_genome: genoma macro (usado apenas para eval_type='micro')
        """
        self.trainer = trainer
        self.envs = envs
        self.eval_type = eval_type
        self.best_macro_genome = best_macro_genome
    
    def evaluate_genome(self, genome_data, config):
        """
        Função de avaliação compatível com ParallelEvaluator.
        
        Args:
            genome_data: tupla (genome_id, genome)
            config: configuração NEAT
        
        Returns:
            fitness do genoma
        """
        genome_id, genome = genome_data
        
        if self.eval_type == 'macro':
            fitness, _, _ = self.trainer.eval_macro_genome(genome, self.envs)
        else:  # micro
            fitness, _, _ = self.trainer.eval_micro_genome(self.best_macro_genome, genome, self.envs)
        
        genome.fitness = fitness
        return fitness


def prepare_asymmetric_data(
    df: pd.DataFrame,
    macro_window: int = 492,
    micro_window: int = 60
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepara dados com janelas assimétricas para NEAT.
    
    Returns:
        prices, macro_features, micro_features
    """
    print(f"\n📊 Preparando dados assimétricos para NEAT...")
    print(f"   Macro window: {macro_window} candles ({macro_window*5/60:.1f}h)")
    print(f"   Micro window: {micro_window} candles ({micro_window*5/60:.1f}h)")
    
    numeric_cols = []
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            numeric_cols.append(col)
    
    features = df[numeric_cols].fillna(0).values.astype(np.float32)
    prices = df['close'].values.astype(np.float32)
    
    # Normalizar features
    feature_mean = features.mean(axis=0, keepdims=True)
    feature_std = features.std(axis=0, keepdims=True) + 1e-8
    features_norm = (features - feature_mean) / feature_std
    
    macro_features_list = []
    micro_features_list = []
    prices_list = []
    
    for i in range(macro_window, len(df)):
        # Macro: agregação de longo prazo
        macro_window_data = features_norm[i-macro_window:i]
        macro_agg = np.concatenate([
            macro_window_data.mean(axis=0),
            macro_window_data.std(axis=0),
            macro_window_data[-1]
        ])
        
        # Micro: agregação de curto prazo
        micro_start = max(0, i - micro_window)
        micro_window_data = features_norm[micro_start:i]
        if len(micro_window_data) < micro_window:
            pad_size = micro_window - len(micro_window_data)
            micro_window_data = np.vstack([
                np.zeros((pad_size, features_norm.shape[1])),
                micro_window_data
            ])
        
        micro_agg = np.concatenate([
            micro_window_data.mean(axis=0),
            micro_window_data.std(axis=0),
            micro_window_data[-1]
        ])
        
        macro_features_list.append(macro_agg)
        micro_features_list.append(micro_agg)
        prices_list.append(prices[i])
    
    macro_features = np.array(macro_features_list, dtype=np.float32)
    micro_features = np.array(micro_features_list, dtype=np.float32)
    prices_array = np.array(prices_list, dtype=np.float32)
    
    print(f"✅ Dados preparados:")
    print(f"   Samples: {len(prices_array)}")
    print(f"   Macro features dim: {macro_features.shape[1]}")
    print(f"   Micro features dim: {micro_features.shape[1]}")
    
    return prices_array, macro_features, micro_features


def create_vectorized_environments(
    prices: np.ndarray,
    macro_features: np.ndarray,
    micro_features: np.ndarray,
    num_envs: int,
    initial_capital: float = 10000.0,
    commission: float = 0.001
) -> List[TradingEnvironmentRL]:
    """Divide dataset em fatias para múltiplos ambientes."""
    total_samples = len(prices)
    if total_samples < 2:
        return []

    max_envs = max(1, min(num_envs, total_samples // 600))
    if max_envs < num_envs:
        print(
            f"⚠️  Ajustando num_envs de {num_envs} para {max_envs} (dataset pequeno)"
        )

    chunk_size = total_samples // max_envs
    environments: List[TradingEnvironmentRL] = []

    for env_idx in range(max_envs):
        start = env_idx * chunk_size
        end = total_samples if env_idx == max_envs - 1 else (env_idx + 1) * chunk_size
        if end - start < 2:
            continue

        env = TradingEnvironmentRL(
            prices=prices[start:end],
            macro_features=macro_features[start:end],
            micro_features=micro_features[start:end],
            initial_capital=initial_capital,
            commission=commission
        )
        environments.append(env)

    return environments


def create_neat_config(
    input_nodes: int,
    output_nodes: int,
    config_name: str = "default"
) -> neat.Config:
    """
    Cria configuração NEAT customizada.
    
    Args:
        input_nodes: número de inputs
        output_nodes: número de outputs
        config_name: nome para distinção
    
    Returns:
        neat.Config com parâmetros otimizados para trading
    """
    
    # Usar template base e modificar inputs/outputs
    template_path = Path("neat_config_template.txt")
    
    if not template_path.exists():
        raise FileNotFoundError(f"Arquivo de template NEAT não encontrado em {template_path.resolve()}")
    
    # Ler template
    with open(template_path, 'r') as f:
        config_text = f.read()
    
    # Modificar inputs/outputs
    config_text = config_text.replace("num_inputs              = 100", f"num_inputs              = {input_nodes}")
    config_text = config_text.replace("num_outputs             = 3", f"num_outputs             = {output_nodes}")
    
    # Escrever em arquivo temporário
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(config_text)
        config_path = f.name
    
    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
    finally:
        # Limpar arquivo temporário
        try:
            os.unlink(config_path)
        except:
            pass
    
    return config


def train_asymmetric_neat(
    duration_minutes: int = 10,
    log_interval_seconds: int = 30,
    portfolio_target: float = 12000.0,
    num_envs: int = 8,
    population_size: int = 50
):
    """
    Treinar redes com NEAT assimétrico.
    
    - MacroNet: evolui 1x a cada 10 episódios (estratégia)
    - MicroNet: evolui 1x por episódio (tática)
    - Ratio: 1:10 (extremamente ágil)
    """
    print("\n" + "="*70)
    print("  🧬 TREINAMENTO ASSIMÉTRICO COM NEAT")
    print("  MacroNet: Evolução 1x a cada 10 episódios (longo prazo)")
    print("  MicroNet: Evolução 1x por episódio (curto prazo, MUITO ágil)")
    print("  Ratio: 1:10 🚀")
    print("="*70 + "\n")
    print("  MicroNet: Evolução 1x por geração (curto prazo)")
    print("="*70 + "\n")
    
    # 1. Carregar dados
    print("📅 Carregando dados de 2024 a partir do parquet local...")
    from src.features.builder import FeatureBuilder
    from datetime import datetime

    data_path = Path("data/timeframe=5m/symbol=BTCUSDT/candles.parquet")
    if not data_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado em {data_path.resolve()}")

    df = pd.read_parquet(data_path, engine="pyarrow")
    if 'timestamp' not in df.columns:
        raise KeyError("Coluna 'timestamp' ausente")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    numeric_cols = [
        'open', 'high', 'low', 'close', 'volume', 'quote_volume',
        'trades_count', 'taker_buy_volume'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_2024 = df[
        (df['timestamp'] >= datetime(2024, 1, 1)) &
        (df['timestamp'] < datetime(2025, 1, 1))
    ].copy()

    if df_2024.empty:
        raise ValueError("Dataset de 2024 está vazio")

    builder = FeatureBuilder()
    df_2024 = builder.add_features(df_2024)
    
    print(f"✅ {len(df_2024)} candles de 2024")
    
    # 2. Preparar dados
    prices, macro_features, micro_features = prepare_asymmetric_data(
        df_2024,
        macro_window=492,
        micro_window=60
    )
    
    envs = create_vectorized_environments(
        prices=prices,
        macro_features=macro_features,
        micro_features=micro_features,
        num_envs=num_envs,
        initial_capital=10000.0,
        commission=0.001
    )

    if not envs:
        raise ValueError("Não foi possível criar ambientes")

    print(f"🧪 Ambientes ativos: {len(envs)}")
    
    # 3. Criar configurações NEAT
    print("\n⚙️  Criando configurações NEAT...")
    
    # Macro: input = macro_features.shape[1], output = dimensão de embedding (ex: 32)
    config_macro = create_neat_config(
        input_nodes=macro_features.shape[1],
        output_nodes=32,
        config_name="macro"
    )
    
    # Micro: input = micro_features.shape[1] + 32 (macro embedding) + 2 (pos, cash)
    config_micro = create_neat_config(
        input_nodes=micro_features.shape[1] + 32 + 2,
        output_nodes=3,  # HOLD, BUY, SELL
        config_name="micro"
    )
    
    # 4. Criar trainer NEAT
    trainer = AsymmetricNEATTrainer(
        config_macro=config_macro,
        config_micro=config_micro,
        device=config.device
    )
    
    print(f"\n🚀 Iniciando evolução assimétrica por {duration_minutes} minutos...")
    print(f"📊 Dataset: {len(prices)} candles")
    print(f"💰 Capital inicial: $10,000")
    print(f"🧬 População inicial: {population_size} indivíduos (macro + micro)")
    print(f"⚙️  Estratégia: 1 macro update : 10 micro updates (ALTA AGILIDADE)")
    print(f"🧪 Ambientes paralelos: {len(envs)}\n")
    
    # Multiprocessing ativado!
    print(f"🚀 Usando MULTIPROCESSING com {cpu_count()} workers (paralelização real!)")
    print(f"⚡ Steps reduzidos para 200 para maior velocidade")

    # 5. Evoluir
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    last_log_time = start_time
    
    episode = 0
    recent_portfolios = []
    
    history = {
        'time_min': [],
        'episode': [],
        'macro_updates': [],
        'micro_updates': [],
        'best_macro_fitness': [],
        'best_micro_fitness': [],
        'avg_reward': []
    }
    
    table_header_printed = False

    while time.time() < end_time:
        elapsed = time.time() - start_time
        
        # Padrão 1:10 - Macro evolui a cada 10 episódios, Micro evolui sempre
        macro_update = (episode % 10 == 0)  # Macro: episódios 0, 10, 20, 30...
        micro_update = True  # Micro: SEMPRE
        
        # Avaliar e evoluir (COM MULTIPROCESSING!)
        result = trainer.evolve_generation(
            macro_genomes=trainer.macro_population.population,
            micro_genomes=trainer.micro_population.population,
            envs=envs,
            update_macro=macro_update,
            update_micro=micro_update,
            use_multiprocessing=True,  # ATIVADO!
            max_steps=200  # Reduzido para speedup
        )
        best_macro_fitness, best_micro_fitness, avg_macro_portfolio, avg_micro_portfolio, avg_macro_reward, avg_micro_reward, eval_time = result
        
        # Usar portfolio micro como primário (sempre atualizado)
        current_portfolio = avg_micro_portfolio if avg_micro_portfolio > 0 else avg_macro_portfolio
        recent_portfolios.append(current_portfolio)
        if len(recent_portfolios) > 10:
            recent_portfolios.pop(0)
        
        episode += 1
        
        # Log periódico
        current_time = time.time()
        if current_time - last_log_time >= log_interval_seconds or episode % 5 == 0:
            if not table_header_printed:
                print("\nTempo(min) | Episódio | MacroUpd | MicroUpd | Ratio | Fitness Macro | Fitness Micro | Reward Médio")
                print("-" * 120)
                table_header_printed = True
            
            ratio = trainer.generation_micro / max(1, trainer.generation_macro)

            print(
                f"{elapsed/60:>9.1f} | {episode:>8} | {trainer.generation_macro:>8} | "
                f"{trainer.generation_micro:>8} | {ratio:>5.1f} | {best_macro_fitness:>13.6f} | "
                f"{best_micro_fitness:>13.6f} | {best_micro_fitness:>12.6f}"
            )
            
            history['time_min'].append(elapsed / 60)
            history['episode'].append(episode)
            history['macro_updates'].append(trainer.generation_macro)
            history['micro_updates'].append(trainer.generation_micro)
            history['best_macro_fitness'].append(best_macro_fitness)
            history['best_micro_fitness'].append(best_micro_fitness)
            history['avg_reward'].append(best_micro_fitness)
            
            last_log_time = current_time

        # Salvar modelo e tabela periodicamente
        if episode % 100 == 0 and episode > 0:
            print(f"\n💾 Salvamento periódico no episódio {episode}...")
            output_dir = Path("./training_results_neat")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Salvar melhores genomas
            best_macro_genome_id = max(trainer.macro_population.population, key=lambda g: trainer.macro_population.population[g].fitness or -np.inf)
            best_macro = trainer.macro_population.population[best_macro_genome_id]
            
            best_micro_genome_id = max(trainer.micro_population.population, key=lambda g: trainer.micro_population.population[g].fitness or -np.inf)
            best_micro = trainer.micro_population.population[best_micro_genome_id]

            with open(output_dir / f"best_macro_genome_ep{episode}.pkl", "wb") as f:
                pickle.dump(best_macro, f)
            
            with open(output_dir / f"best_micro_genome_ep{episode}.pkl", "wb") as f:
                pickle.dump(best_micro, f)

            # Salvar tabela de evolução
            history_df = pd.DataFrame(history)
            history_df.to_csv(output_dir / f"evolution_table_ep{episode}.csv", index=False)
            print(f"✅ Modelos e tabela de evolução salvos em {output_dir}/")

    
    # Salvar melhor genoma no final
    total_time = time.time() - start_time
    output_dir = Path("./training_results_neat")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_macro_genome_id = max(
        trainer.macro_population.population,
        key=lambda g: trainer.macro_population.population[g].fitness or -np.inf
    )
    best_macro = trainer.macro_population.population[best_macro_genome_id]

    best_micro_genome_id = max(
        trainer.micro_population.population,
        key=lambda g: trainer.micro_population.population[g].fitness or -np.inf
    )
    best_micro = trainer.micro_population.population[best_micro_genome_id]
    
    with open(output_dir / "best_macro_genome_final.pkl", "wb") as f:
        pickle.dump(best_macro, f)
    
    with open(output_dir / "best_micro_genome_final.pkl", "wb") as f:
        pickle.dump(best_micro, f)

    # Salvar tabela de evolução final
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "evolution_table_final.csv", index=False)
    
    print(f"\n✅ Treinamento NEAT assimétrico completo: {episode} episódios em {total_time/60:.1f}min")
    print(f"🔄 Total updates - Macro: {trainer.generation_macro} | Micro: {trainer.generation_micro}")
    print(f"📊 Ratio final: 1:{trainer.generation_micro/max(1, trainer.generation_macro):.2f}")
    print(f"🧬 Melhor MacroNet fitness: {trainer.best_macro_fitness:.6f}")
    print(f"🧬 Melhor MicroNet fitness: {trainer.best_micro_fitness:.6f}")
    print(f"💾 Genomas salvos em: {output_dir}/")
    
    # Plot
    if history['episode']:
        plot_neat_evolution(history, output_dir, total_time / 60, episode, trainer)


def plot_neat_evolution(history: dict, output_dir: Path, total_time: float, episodes: int, trainer):
    """Plotar evolução NEAT"""
    
    print(f"\n📊 Gerando gráfico de evolução...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Treinamento NEAT Assimétrico (1:10) - {total_time:.1f} min, {episodes} episódios',
        fontsize=14,
        fontweight='bold'
    )
    
    time_axis = history['time_min']
    
    # 1. Fitness Macro
    ax1 = axes[0, 0]
    ax1.plot(time_axis, history['best_macro_fitness'], 'r-', linewidth=2)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero')
    ax1.set_xlabel('Tempo (minutos)')
    ax1.set_ylabel('Fitness MacroNet')
    ax1.set_title('Evolução Fitness MacroNet')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Fitness Micro
    ax2 = axes[0, 1]
    ax2.plot(time_axis, history['best_micro_fitness'], 'b-', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero')
    ax2.set_xlabel('Tempo (minutos)')
    ax2.set_ylabel('Fitness MicroNet')
    ax2.set_title('Evolução Fitness MicroNet')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Updates Count
    ax3 = axes[1, 0]
    ax3.plot(time_axis, history['macro_updates'], 'r-', linewidth=2, label='Macro Updates')
    ax3.plot(time_axis, history['micro_updates'], 'b-', linewidth=2, label='Micro Updates')
    ax3.set_xlabel('Tempo (minutos)')
    ax3.set_ylabel('Número de Updates')
    ax3.set_title('Updates por Componente')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Update Ratio
    ax4 = axes[1, 1]
    ratios = [m / max(1, macro) for m, macro in zip(history['micro_updates'], history['macro_updates'])]
    ax4.plot(time_axis, ratios, 'purple', linewidth=2)
    ax4.axhline(y=10.0, color='orange', linestyle='--', alpha=0.5, label='Target Ratio (10:1)')
    ax4.set_xlabel('Tempo (minutos)')
    ax4.set_ylabel('Ratio (Micro:Macro)')
    ax4.set_title('Ratio de Updates')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    plot_path = output_dir / 'neat_asymmetric_evolution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico salvo: {plot_path}")
    
    plt.close()


if __name__ == "__main__":
    import os
    import subprocess
    
    # Prevenir que o Mac entre em sleep (caffeinate)
    print("\n" + "="*70)
    print("  🔋 PREVENINDO SLEEP DO MAC (caffeinate)")
    print("="*70)
    caffeinate_proc = None
    try:
        try:
            caffeinate_proc = subprocess.Popen([
                'caffeinate', '-dimsu'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ Caffeinate ativado (PID: {caffeinate_proc.pid})")
            print("⚡ Mac NÃO entrará em sleep durante o treinamento. Feche a tampa se desejar (conecte o carregador).")
        except Exception as e:
            caffeinate_proc = None
            print(f"⚠️  Não foi possível ativar caffeinate automaticamente: {e}")
            print("   Execute manualmente: caffeinate -dimsu &")

        # Setup MPS
        print("\n" + "="*70)
        print("  ⚙️  CONFIGURAÇÃO DE DISPOSITIVO")
        print("="*70)
        
        mps_available = torch.backends.mps.is_available()
        print(f"🔍 MPS disponível: {mps_available}")
        
        if mps_available:
            mps_built = torch.backends.mps.is_built()
            print(f"🔧 PyTorch compilado com MPS: {mps_built}")
            if mps_built:
                print(f"✅ Usando MPS (Metal Performance Shaders)")
                config.device = "mps"
            else:
                print(f"⚠️  PyTorch sem suporte MPS, usando CPU")
                config.device = "cpu"
        else:
            print(f"ℹ️  MPS não disponível, usando CPU")
            config.device = "cpu"
        
        torch.set_num_threads(os.cpu_count() or 4)
        print(f"✅ Threads CPU: {os.cpu_count() or 4}")
        print("="*70 + "\n")
        
        # Rodar treinamento NEAT assimétrico
        train_asymmetric_neat(
            duration_minutes=337.9,
            log_interval_seconds=30,
            portfolio_target=12000.0,
            num_envs=8,
            population_size=50
        )
    finally:
        # Finalizar caffeinate se foi iniciado
        if caffeinate_proc is not None:
            try:
                print(f"\n🛑 Finalizando caffeinate (PID: {caffeinate_proc.pid})")
                caffeinate_proc.terminate()
                caffeinate_proc.wait(timeout=5)
                print("✅ Caffeinate finalizado")
            except Exception:
                pass
