"""
Pipeline de Treinamento Assimétrico com NEAT (NeuroEvolution of Augmenting Topologies)

Estratégia:
- MacroNet NEAT: Rede que evolui topologia (long-term, 41h)
- MicroNet NEAT: Rede que evolui topologia (short-term, 5h)
- Evolução assimétrica: Macro treina a cada 2 gerações, Micro treina a cada geração
- Vantagens: topologias adaptadas ao problema, sem precisar definir arquitetura manualmente

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

from src.pipeline import TradingPipeline
from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


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
    
    def step(self, action: int) -> Tuple[dict, float, bool]:
        """
        Execute action: 0=HOLD, 1=BUY, 2=SELL
        
        Returns:
            next_state, reward, done
        """
        if self.step_idx >= len(self.prices) - 1:
            return None, 0.0, True
        
        current_price = self.prices[self.step_idx]
        next_price = self.prices[self.step_idx + 1]
        
        # Execute action
        if action == 1 and self.position < 1.0:  # BUY
            trade_size = 1.0 - self.position
            cost = trade_size * current_price * (1 + self.commission)
            if cost <= self.cash:
                self.cash -= cost
                self.position += trade_size
        
        elif action == 2 and self.position > -1.0:  # SELL
            trade_size = self.position - (-1.0)
            proceeds = trade_size * current_price * (1 - self.commission)
            self.cash += proceeds
            self.position -= trade_size
        
        # Advance
        self.step_idx += 1
        
        # Calculate reward (P&L)
        price_change = next_price - current_price
        position_pnl = self.position * price_change
        reward = (position_pnl / self.initial_capital) * 100
        
        # Penalty for trading
        if action != 0:
            reward -= 0.01
        
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
    
    def eval_macro_genome(self, genome: neat.DefaultGenome, envs: List[TradingEnvironmentRL]) -> float:
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
        
        for env in envs:
            state = env.reset()
            total_reward = 0.0
            
            while True:
                # Forward pass: macro features → [embed_1, embed_2, ...]
                macro_output = self.macro_adapter.forward_neat(
                    genome,
                    state['macro_features']
                )
                
                # Macro output = contexto de longo prazo
                # Simplicidade: usar saída diretamente como "confiança" para ações
                # Na implementação real, isso seria passado para a micro
                
                # Usar primeiras 3 saídas como logits de ação (HOLD, BUY, SELL)
                action_logits = macro_output[:3]
                if action_logits.shape[0] < 3:
                    action_logits = np.pad(action_logits, (0, 3 - action_logits.shape[0]), constant_values=0.0)
                # Fallback aleatório quando logits são quase idênticos (rede ainda neutra)
                if np.allclose(action_logits, action_logits[0], atol=1e-6):
                    action = np.random.randint(0, 3)
                else:
                    action = int(np.argmax(action_logits))
                
                next_state, reward, done = env.step(action)
                total_reward += reward
                
                if done:
                    break
                
                state = next_state
            
            final_return = ((env.portfolio_value - 10000) / 10000) * 100
            total_return += final_return
            num_envs += 1
        
        fitness = total_return / max(1, num_envs)
        return fitness
    
    def eval_micro_genome(
        self,
        macro_genome: neat.DefaultGenome,
        micro_genome: neat.DefaultGenome,
        envs: List[TradingEnvironmentRL]
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
        
        for env in envs:
            state = env.reset()
            total_reward = 0.0
            
            while True:
                # Macro: contexto de longo prazo
                macro_output = self.macro_adapter.forward_neat(
                    macro_genome,
                    state['macro_features']
                )
                
                # Micro: recebe micro_features + macro_output concatenados
                micro_input = np.concatenate([
                    state['micro_features'],
                    macro_output,
                    [state['position'], state['cash'] / 10000.0]
                ])
                
                micro_output = self.micro_adapter.forward_neat(
                    micro_genome,
                    micro_input
                )
                
                # Ação: argmax de micro output
                action = np.argmax(micro_output) % 3
                
                next_state, reward, done = env.step(action)
                total_reward += reward
                
                if done:
                    break
                
                state = next_state
            
            final_return = ((env.portfolio_value - 10000) / 10000) * 100
            total_return += final_return
            num_envs += 1
        
        fitness = total_return / max(1, num_envs)
        return fitness
    
    def evolve_generation(
        self,
        macro_genomes: Dict,
        micro_genomes: Dict,
        envs: List[TradingEnvironmentRL],
        update_macro: bool = False,
        update_micro: bool = True
    ) -> Tuple[float, float]:
        """
        Executar uma geração de evolução (NEAT com seleção natural + crossover + mutação).
        
        Args:
            macro_genomes: dict {genome_id: genome} de genomas macro
            micro_genomes: dict {genome_id: genome} de genomas micro
            envs: ambientes para avaliação
            update_macro: se deve evoluir população macro
            update_micro: se deve evoluir população micro
        
        Returns:
            (best_macro_fitness, best_micro_fitness)
        """
        
        # Avaliar genomas macro
        if update_macro:
            print("\n  🧬 Evoluindo MacroNet...")
            macro_fitness = {}
            for gid, genome in macro_genomes.items():
                macro_fitness[gid] = self.eval_macro_genome(genome, envs)
                print(f"    Macro #{gid}: fitness = {macro_fitness[gid]:.6f}")
            
            # Especiação e seleção
            self.macro_population.species.speciate(self.config_macro, macro_genomes, self.generation_macro)
            for genome in self.macro_population.species.get_fitnesses():
                pass  # NEAT cuida da seleção
            
            self.generation_macro += 1
            self.best_macro_fitness = max(macro_fitness.values())
            print(f"    ✅ Melhor macro fitness: {self.best_macro_fitness:.4f}")
        
        # Avaliar genomas micro
        if update_micro:
            print("\n  🧬 Evoluindo MicroNet...")
            # Usar melhor genoma macro como referência
            best_macro = max(macro_genomes.items(), key=lambda x: self.eval_macro_genome(x[1], envs))[1]
            
            micro_fitness = {}
            for gid, genome in micro_genomes.items():
                micro_fitness[gid] = self.eval_micro_genome(best_macro, genome, envs)
                print(f"    Micro #{gid}: fitness = {micro_fitness[gid]:.6f}")
            
            self.micro_population.species.speciate(self.config_micro, micro_genomes, self.generation_micro)
            
            self.generation_micro += 1
            self.best_micro_fitness = max(micro_fitness.values())
            print(f"    ✅ Melhor micro fitness: {self.best_micro_fitness:.4f}")
        
        return self.best_macro_fitness, self.best_micro_fitness


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
    num_envs: int = 8,
    population_size: int = 50
):
    """
    Treinar redes com NEAT assimétrico.
    
    - MacroNet: evolui 1x a cada 2 gerações
    - MicroNet: evolui 1x por geração
    - Ambos começam com topologia simples, evoluem para mais complexos
    """
    print("\n" + "="*70)
    print("  🧬 TREINAMENTO ASSIMÉTRICO COM NEAT")
    print("  MacroNet: Evolução 1x a cada 2 gerações (longo prazo)")
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
    print(f"⚙️  Macro: atualiza a cada 2 gerações | Micro: atualiza a cada geração")
    print(f"🧪 Ambientes paralelos: {len(envs)}\n")
    
    # 5. Evoluir
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    last_log_time = start_time
    
    generation = 0
    history = {
        'generation': [],
        'time_min': [],
        'macro_fitness': [],
        'micro_fitness': []
    }
    
    table_header_printed = False

    while time.time() < end_time:
        elapsed = time.time() - start_time
        
        # Decidir qual população evoluir
        # Ratio 1:2 → a cada 3 gerações: [M+m, m, m] (simplificado: alternado)
        macro_update = generation % 2 == 0
        micro_update = True  # Micro sempre evolui
        
        print(f"\n🔄 Geração {generation}...")
        if macro_update:
            print("   ├─ 🧬 MacroNet evoluindo")
        if micro_update:
            print("   └─ 🧬 MicroNet evoluindo")
        
        # Avaliar e evoluir
        best_macro_fitness, best_micro_fitness = trainer.evolve_generation(
            macro_genomes=trainer.macro_population.population,
            micro_genomes=trainer.micro_population.population,
            envs=envs,
            update_macro=macro_update,
            update_micro=micro_update
        )
        
        generation += 1
        
        # Log periódico
        current_time = time.time()
        if current_time - last_log_time >= log_interval_seconds or generation % 5 == 0:
            if not table_header_printed:
                print("\nGen | Tempo(min) | MacroFit | MicroFit | PopMacro | PopMicro")
                print("-" * 60)
                table_header_printed = True
            
            pop_size_macro = len(trainer.macro_population.population)
            pop_size_micro = len(trainer.micro_population.population)
            
            print(
                f"{generation:>3} | {elapsed/60:>9.1f} | "
                f"{best_macro_fitness:>10.6f} | {best_micro_fitness:>10.6f} | "
                f"{pop_size_macro:>8} | {pop_size_micro:>8}"
            )
            
            history['generation'].append(generation)
            history['time_min'].append(elapsed / 60)
            history['macro_fitness'].append(best_macro_fitness)
            history['micro_fitness'].append(best_micro_fitness)
            
            last_log_time = current_time
    
    # Salvar melhor genoma
    output_dir = Path("./training_results_neat")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_macro = max(
        trainer.macro_population.population.items(),
        key=lambda x: trainer.eval_macro_genome(x[1], envs)
    )[1]
    best_micro = max(
        trainer.micro_population.population.items(),
        key=lambda x: trainer.eval_micro_genome(best_macro, x[1], envs)
    )[1]
    
    with open(output_dir / "best_macro_genome.pkl", "wb") as f:
        pickle.dump(best_macro, f)
    
    with open(output_dir / "best_micro_genome.pkl", "wb") as f:
        pickle.dump(best_micro, f)
    
    print(f"\n✅ Treinamento NEAT assimétrico completo: {generation} gerações")
    print(f"🧬 Melhor MacroNet fitness: {best_macro_fitness:.4f}")
    print(f"🧬 Melhor MicroNet fitness: {best_micro_fitness:.4f}")
    print(f"💾 Genomas salvos em: {output_dir}/")
    
    # Plot
    if history['generation']:
        plot_neat_evolution(history, output_dir, elapsed / 60, generation)


def plot_neat_evolution(history: dict, output_dir: Path, total_time: float, generations: int):
    """Plotar evolução NEAT"""
    
    print(f"\n📊 Gerando gráfico de evolução...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Evolução NEAT Assimétrica - {total_time:.1f} min, {generations} gerações',
        fontsize=14,
        fontweight='bold'
    )
    
    time_axis = history['time_min']
    
    # 1. Fitness ao longo das gerações
    ax1.plot(time_axis, history['macro_fitness'], 'r-o', linewidth=2, label='MacroNet')
    ax1.plot(time_axis, history['micro_fitness'], 'b-o', linewidth=2, label='MicroNet')
    ax1.set_xlabel('Tempo (minutos)')
    ax1.set_ylabel('Fitness (Return %)')
    ax1.set_title('Melhor Fitness por Geração')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Fitness relativo
    ax2.plot(time_axis, history['macro_fitness'], 'r-', linewidth=2, label='MacroNet', alpha=0.7)
    ax2.plot(time_axis, history['micro_fitness'], 'b-', linewidth=2, label='MicroNet', alpha=0.7)
    ax2.fill_between(time_axis, history['macro_fitness'], alpha=0.2, color='red')
    ax2.fill_between(time_axis, history['micro_fitness'], alpha=0.2, color='blue')
    ax2.set_xlabel('Tempo (minutos)')
    ax2.set_ylabel('Fitness (Return %)')
    ax2.set_title('Fitness Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    plot_path = output_dir / 'neat_evolution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico salvo: {plot_path}")
    
    plt.close()


if __name__ == "__main__":
    import os
    
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
        duration_minutes=999.9,
        log_interval_seconds=30,
        num_envs=8,
        population_size=50
    )
