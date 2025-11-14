"""
Treinamento de MicroNet Recorrente com NEAT (1.5x maior)

Focado APENAS na rede micro com:
- Tamanho 1.5x maior (população 225 vs 150)
- Conexões recorrentes habilitadas
- Sem arquitetura assimétrica (apenas micro)
- Avaliação balanceada em 3 símbolos (BTC/ETH/BNB)
"""

import numpy as np
import pandas as pd
import torch
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
import tempfile
import os
import sys

# ============================================================================
# CONFIGURAÇÃO HEADLESS (TAMPA FECHADA)
# ============================================================================
# Força modo headless para evitar problemas com matplotlib em notebooks/terminal
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo (salva apenas arquivos)
import matplotlib.pyplot as plt

# NEAT imports
import neat
from multiprocessing import Pool, cpu_count
from functools import partial

from src.pipeline import TradingPipeline
from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)

# Suprimir warnings desnecessários
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================================
# FUNÇÕES AUXILIARES PARA ANÁLISE DA TOPOLOGIA NEAT
# ============================================================================

def calculate_network_depth(genome: neat.DefaultGenome, config: neat.Config) -> int:
    """Calcula a profundidade máxima da rede NEAT."""
    input_nodes = set(range(-config.genome_config.num_inputs, 0))
    output_nodes = set(range(config.genome_config.num_outputs))
    
    connections = {}
    for conn_key, conn in genome.connections.items():
        if conn.enabled:
            input_node, output_node = conn_key
            if output_node not in connections:
                connections[output_node] = []
            connections[output_node].append(input_node)
    
    max_depth = 0
    for output_node in output_nodes:
        depth = _calculate_node_depth(output_node, connections, input_nodes, set())
        max_depth = max(max_depth, depth)
    
    return max_depth


def _calculate_node_depth(node, connections, input_nodes, visited):
    """Helper recursivo para calcular profundidade de um nó."""
    if node in visited:
        return 0
    if node in input_nodes:
        return 1
    
    visited.add(node)
    
    if node not in connections:
        return 1
    
    max_predecessor_depth = 0
    for predecessor in connections[node]:
        depth = _calculate_node_depth(predecessor, connections, input_nodes, visited)
        max_predecessor_depth = max(max_predecessor_depth, depth)
    
    return 1 + max_predecessor_depth


def calculate_network_width(genome: neat.DefaultGenome, config: neat.Config) -> int:
    """Calcula a largura máxima da rede (maior número de neurônios em uma camada)."""
    input_nodes = set(range(-config.genome_config.num_inputs, 0))
    output_nodes = set(range(config.genome_config.num_outputs))
    hidden_nodes = set(genome.nodes.keys()) - output_nodes
    
    connections = {}
    for conn_key, conn in genome.connections.items():
        if conn.enabled:
            input_node, output_node = conn_key
            if output_node not in connections:
                connections[output_node] = []
            connections[output_node].append(input_node)
    
    layers = {}
    for output_node in output_nodes:
        _assign_layers(output_node, connections, input_nodes, layers, set())
    
    if not layers:
        return len(hidden_nodes) if hidden_nodes else 1
    
    layer_counts = {}
    for node, layer in layers.items():
        if node not in input_nodes and node not in output_nodes:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    return max(layer_counts.values()) if layer_counts else 1


def _assign_layers(node, connections, input_nodes, layers, visited):
    """Helper para atribuir camadas aos nós."""
    if node in visited:
        return layers.get(node, 0)
    if node in input_nodes:
        layers[node] = 0
        return 0
    
    visited.add(node)
    
    if node not in connections:
        layers[node] = 1
        return 1
    
    max_predecessor_layer = 0
    for predecessor in connections[node]:
        layer = _assign_layers(predecessor, connections, input_nodes, layers, visited)
        max_predecessor_layer = max(max_predecessor_layer, layer)
    
    layers[node] = max_predecessor_layer + 1
    return max_predecessor_layer + 1


# ============================================================================
# FUNÇÃO WORKER PARA AVALIAÇÃO PARALELA
# ============================================================================

def evaluate_genome_worker(genome_data, config_micro, envs_data, max_steps=200):
    """
    Worker para avaliar um genoma micro em paralelo.
    Usa RecurrentNetwork para suportar conexões recorrentes.
    Com sistema de peso progressivo: primeiros steps têm menos relevância.
    """
    genome_id, genome = genome_data
    
    # Recriar ambientes
    envs = []
    for env_data in envs_data:
        env = TradingEnvironmentRL(
            prices=env_data['prices'],
            micro_features=env_data['micro_features'],
            initial_capital=env_data['initial_capital'],
            commission=env_data['commission']
        )
        env.step_idx = env_data.get('step_idx', 0)
        env.symbol = env_data.get('symbol', 'UNKNOWN')
        envs.append(env)
    
    # Criar rede NEAT recorrente
    feed_forward = getattr(config_micro.genome_config, "feed_forward", True)
    net_cls = neat.nn.FeedForwardNetwork if feed_forward else neat.nn.RecurrentNetwork
    net = net_cls.create(genome, config_micro)
    
    total_reward = 0.0
    num_envs = 0
    
    for env in envs:
        state = env._get_state()
        steps = 0
        episode_reward = 0.0
        
        # Reset do estado recorrente no início de cada episódio
        if not feed_forward:
            net.reset()
        
        # Sistema de peso progressivo multiplicativo
        cumulative_weight = 0.1  # Peso inicial baixo (10%)
        
        while steps < max_steps:
            # Forward pass
            micro_output = np.asarray(net.activate(state['micro_features']), dtype=np.float32)
            
            # Usar saída bruta como previsão
            prediction_value = float(micro_output[0])
            
            # Escolher ação
            action = np.argmax(micro_output) % 3
            
            next_state, reward, done = env.step(action, prediction_value)
            
            # Aplicar peso progressivo ao reward
            weighted_reward = reward * cumulative_weight
            episode_reward += weighted_reward
            
            # Multiplicar peso apenas se reward > 0, senão zera
            if reward > 0:
                cumulative_weight = min(cumulative_weight * 1.02, 1.0)  # +2% por step, max 100%
            elif reward < 0:
                cumulative_weight = 0.0  # Reset para 10% em caso de erro
                break  # Termina episódio antecipadamente
            # Se reward == 0, mantém o peso atual
            
            steps += 1
            
            if done or next_state is None:
                break
            
            state = next_state
        
        total_reward += episode_reward
        num_envs += 1
    
    fitness = total_reward / max(1, num_envs)
    genome.fitness = fitness
    return genome_id, genome, fitness


# ============================================================================
# AMBIENTE DE TRADING
# ============================================================================

class TradingEnvironmentRL:
    """Ambiente de trading simplificado para RL com NEAT"""
    
    def __init__(
        self,
        prices: np.ndarray,
        micro_features: np.ndarray,
        initial_capital: float = 10000.0,
        commission: float = 0.001
    ):
        self.prices = prices
        self.micro_features = micro_features
        self.initial_capital = initial_capital
        self.commission = commission
        self.symbol = "UNKNOWN"
        
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
            self.step_idx = 0
            self.cash = self.initial_capital
            self.position = 0.0
            self.portfolio_value = self.initial_capital
        
        return {
            'micro_features': self.micro_features[self.step_idx],
            'price': self.prices[self.step_idx],
            'position': self.position,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value
        }
    
    def step(self, action: int, prediction_value: float = 0.0) -> Tuple[dict, float, bool]:
        """
        Execute action com fórmula quadrática de fitness.
        """
        if self.step_idx >= len(self.prices) - 1:
            self.step_idx = 0
            self.cash = self.initial_capital
            self.position = 0.0
            self.portfolio_value = self.initial_capital
            return self._get_state(), 0.0, False
        
        current_price = self.prices[self.step_idx]
        next_price = self.prices[self.step_idx + 1]
        
        # Mudança de preço
        price_change_pct = (next_price - current_price) / current_price
        
        # Fórmula quadrática com bonus de confiança (vetorizada)
        pred_norm = np.clip(prediction_value, -1, 1)
        confidence = abs(pred_norm)
        
        direction_multiplier = 2 * (np.sign(pred_norm) == np.sign(price_change_pct)) - 1
        
        reward = (pred_norm * price_change_pct * 10000) + \
                 ((confidence ** 2) * abs(price_change_pct) * 5000 * direction_multiplier)
        
        # Avançar
        self.step_idx += 1
        self.portfolio_value = self.cash + (self.position * next_price)
        
        done = False
        next_state = self._get_state()
        
        return next_state, reward, done


# ============================================================================
# PREPARAÇÃO DE DADOS
# ============================================================================

def prepare_micro_data(
    df: pd.DataFrame,
    micro_window: int = 120  # 120 steps × 5min = 10 horas de contexto
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepara dados para MicroNet NEAT.
    
    Returns:
        prices, micro_features
    """
    prices_array = df['close'].values.astype(np.float32)
    
    # Filtrar apenas colunas numéricas (excluindo preços, timestamp, symbol, date)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'symbol', 'date']
    feature_cols = [
        col for col in df.columns 
        if col not in exclude_cols and df[col].dtype in [np.float32, np.float64, np.int32, np.int64]
    ]
    
    features_array = df[feature_cols].values.astype(np.float32)
    
    # Normalização robusta
    features_normalized = (features_array - np.median(features_array, axis=0)) / \
                         (np.percentile(features_array, 75, axis=0) - np.percentile(features_array, 25, axis=0) + 1e-8)
    features_normalized = np.clip(features_normalized, -5, 5)
    
    # Criar features com janela micro
    micro_features_list = []
    for i in range(len(features_normalized)):
        start_micro = max(0, i - micro_window + 1)
        micro_window_actual = features_normalized[start_micro:i+1]
        
        if len(micro_window_actual) < micro_window:
            padding = np.zeros((micro_window - len(micro_window_actual), micro_window_actual.shape[1]))
            micro_window_actual = np.vstack([padding, micro_window_actual])
        
        micro_flat = micro_window_actual.flatten()
        micro_features_list.append(micro_flat)
    
    micro_features = np.array(micro_features_list, dtype=np.float32)
    
    return prices_array, micro_features


def create_environments_by_symbol(
    symbols_data: List[Dict],
    initial_capital: float = 10000.0,
    commission: float = 0.001
) -> List[TradingEnvironmentRL]:
    """Cria um ambiente por símbolo."""
    environments = []
    for symbol_data in symbols_data:
        env = TradingEnvironmentRL(
            prices=symbol_data['prices'],
            micro_features=symbol_data['micro_features'],
            initial_capital=initial_capital,
            commission=commission
        )
        env.symbol = symbol_data.get('symbol', 'UNKNOWN')
        environments.append(env)
    return environments


# ============================================================================
# CONFIGURAÇÃO NEAT
# ============================================================================

def create_neat_config_recurrent(
    input_nodes: int,
    output_nodes: int,
    pop_size: int = 225  # 1.5x maior que 150
) -> neat.Config:
    """
    Cria configuração NEAT para rede recorrente maior.
    """
    template_path = Path("neat_config_template.txt")
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template NEAT não encontrado: {template_path.resolve()}")
    
    with open(template_path, 'r') as f:
        config_text = f.read()
    
    # Ajustar inputs/outputs
    config_text = config_text.replace("num_inputs              = 100", f"num_inputs              = {input_nodes}")
    config_text = config_text.replace("num_outputs             = 3", f"num_outputs             = {output_nodes}")
    
    # HABILITAR CONEXÕES RECORRENTES
    config_text = config_text.replace("feed_forward            = True", "feed_forward            = False")
    
    # AUMENTAR POPULAÇÃO 1.5x
    config_text = config_text.replace("pop_size              = 150", f"pop_size              = {pop_size}")
    
    # Escrever em arquivo temporário
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(config_text)
        config_path = f.name
    
    try:
        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
    finally:
        try:
            os.unlink(config_path)
        except:
            pass
    
    return neat_config


# ============================================================================
# TREINADOR NEAT MICRONET RECORRENTE
# ============================================================================

class MicroNetRecurrentTrainer:
    """Treinador focado apenas em MicroNet recorrente."""
    
    def __init__(self, config_micro: neat.Config, device: str = "cpu"):
        self.device = torch.device(device)
        self.config_micro = config_micro
        
        # População NEAT
        self.population = neat.Population(config_micro)
        
        # Histórico
        self.generation = 0
        self.best_fitness = -np.inf
        
        logger.info("micronet_recurrent_trainer_initialized")
    
    def evolve_generation(
        self,
        genomes: dict,
        envs: List[TradingEnvironmentRL],
        use_multiprocessing: bool = True,
        max_steps: int = 150
    ) -> Tuple[float, float]:
        """
        Evolve uma geração.
        
        Returns:
            (best_fitness, eval_time)
        """
        eval_start_time = time.time()
        
        # Preparar dados dos ambientes
        envs_data = []
        for env in envs:
            envs_data.append({
                'prices': env.prices,
                'micro_features': env.micro_features,
                'initial_capital': env.initial_capital,
                'commission': env.commission,
                'step_idx': getattr(env, 'step_idx', 0),
                'symbol': getattr(env, 'symbol', 'UNKNOWN')
            })
        
        # AVALIAR
        if use_multiprocessing and len(genomes) > 1:
            with Pool(processes=min(6, cpu_count())) as pool:
                eval_func = partial(
                    evaluate_genome_worker,
                    config_micro=self.config_micro,
                    envs_data=envs_data,
                    max_steps=max_steps
                )
                results = pool.map(eval_func, list(genomes.items()))
            
            # Atualizar fitness
            for genome_id, genome_result, fitness in results:
                try:
                    self.population.population[genome_id].fitness = fitness
                except Exception:
                    pass
        else:
            # Sequencial
            for genome_id, genome in genomes.items():
                _, _, fitness = evaluate_genome_worker(
                    (genome_id, genome),
                    self.config_micro,
                    envs_data,
                    max_steps
                )
                genome.fitness = fitness
        
        # Calcular melhor fitness
        if genomes:
            self.best_fitness = max(g.fitness for g in genomes.values() if g.fitness is not None)
        
        # REPRODUZIR
        # Limpar espécies vazias
        self.population.species.species = {
            sid: s for sid, s in self.population.species.species.items()
            if len(s.members) > 0
        }
        
        # Reprodução
        new_pop = self.population.reproduction.reproduce(
            self.config_micro,
            self.population.species,
            self.config_micro.pop_size,
            self.generation
        )
        self.population.population = new_pop
        
        # Re-especiar
        self.population.species.speciate(
            self.config_micro,
            self.population.population,
            self.generation
        )
        
        self.generation += 1
        
        # Nota: step_idx é atualizado pelo curriculum learning no loop principal
        # (não mais aqui, pois depende da fase atual)
        
        eval_total_time = time.time() - eval_start_time
        return self.best_fitness, eval_total_time


# ============================================================================
# FUNÇÃO DE TREINAMENTO PRINCIPAL
# ============================================================================

def train_micronet_recurrent(
    duration_minutes: int = 60,
    log_interval_seconds: int = 30,
    max_steps: int = 150,
    pop_size: int = 170
):
    """
    Treina MicroNet recorrente com curriculum learning.
    
    Args:
        duration_minutes: Duração do treinamento em minutos
        log_interval_seconds: Intervalo de log em segundos
        max_steps: Número máximo de steps por episódio (base)
        pop_size: Tamanho da população NEAT
    """
    
    # ════════════════════════════════════════════════════════════════
    # CONFIGURAÇÃO DE FASES (CURRICULUM LEARNING)
    # ════════════════════════════════════════════════════════════════
    # Convergência de 0.85 entre jump_candles: cada novo jump = jump_anterior * 0.85
    # 
    # OVERLAP PROGRESSIVO: jump_candles < max_steps
    # Exemplo: Se max_steps=576 e jump=66:
    #   Geração 1: avalia candles [0 a 576]
    #   Geração 2: avalia candles [66 a 642]  ← overlap de 510 candles (88.5%)
    #   Geração 3: avalia candles [132 a 708] ← overlap de 510 candles (88.5%)
    # 
    # Fases Elite: overlap aumenta (jump diminui, steps aumentam)
    # Elite 20: steps=1728, jump=8 → overlap de 99.5% (quase revisita tudo!)
    # ════════════════════════════════════════════════════════════════
    curriculum_phases = [
        {'name': 'Iniciante',     'generations': 10, 'steps_multiplier': 1.00, 'jump_candles': 66},
        {'name': 'Intermediário', 'generations': 15, 'steps_multiplier': 1.15, 'jump_candles': 80},
        {'name': 'Avançado',      'generations': 20, 'steps_multiplier': 1.30, 'jump_candles': 100},
        {'name': 'Expert',        'generations': 30, 'steps_multiplier': 1.50, 'jump_candles': 120},
        {'name': 'Master',        'generations': 40, 'steps_multiplier': 1.75, 'jump_candles': 150},
        {'name': 'Elite',         'generations': 50, 'steps_multiplier': 2.00, 'jump_candles': 180},
        {'name': 'Elite 1',       'generations': 50, 'steps_multiplier': 2.05, 'jump_candles': 153},
        {'name': 'Elite 2',       'generations': 50, 'steps_multiplier': 2.10, 'jump_candles': 130},
        {'name': 'Elite 3',       'generations': 50, 'steps_multiplier': 2.15, 'jump_candles': 111},
        {'name': 'Elite 4',       'generations': 50, 'steps_multiplier': 2.20, 'jump_candles': 94},
        {'name': 'Elite 5',       'generations': 50, 'steps_multiplier': 2.25, 'jump_candles': 80},
        {'name': 'Elite 6',       'generations': 50, 'steps_multiplier': 2.30, 'jump_candles': 68},
        {'name': 'Elite 7',       'generations': 50, 'steps_multiplier': 2.35, 'jump_candles': 58},
        {'name': 'Elite 8',       'generations': 50, 'steps_multiplier': 2.40, 'jump_candles': 49},
        {'name': 'Elite 9',       'generations': 50, 'steps_multiplier': 2.45, 'jump_candles': 42},
        {'name': 'Elite 10',      'generations': 50, 'steps_multiplier': 2.50, 'jump_candles': 36},
        {'name': 'Elite 11',      'generations': 50, 'steps_multiplier': 2.55, 'jump_candles': 31},
        {'name': 'Elite 12',      'generations': 50, 'steps_multiplier': 2.60, 'jump_candles': 26},
        {'name': 'Elite 13',      'generations': 50, 'steps_multiplier': 2.65, 'jump_candles': 22},
        {'name': 'Elite 14',      'generations': 50, 'steps_multiplier': 2.70, 'jump_candles': 19},
        {'name': 'Elite 15',      'generations': 50, 'steps_multiplier': 2.75, 'jump_candles': 16},
        {'name': 'Elite 16',      'generations': 50, 'steps_multiplier': 2.80, 'jump_candles': 14},
        {'name': 'Elite 17',      'generations': 50, 'steps_multiplier': 2.85, 'jump_candles': 12},
        {'name': 'Elite 18',      'generations': 50, 'steps_multiplier': 2.90, 'jump_candles': 10},
        {'name': 'Elite 19',      'generations': 50, 'steps_multiplier': 2.95, 'jump_candles': 9},
        {'name': 'Elite 20',      'generations': 50, 'steps_multiplier': 3.00, 'jump_candles': 8},
        {'name': 'Elite 21',      'generations': 55, 'steps_multiplier': 3.05, 'jump_candles': 10},
        {'name': 'Elite 22',      'generations': 20, 'steps_multiplier': 1.30, 'jump_candles': 10},
        {'name': 'Elite 23',      'generations': 70, 'steps_multiplier': 3.15, 'jump_candles': 10},
        {'name': 'Elite 24',      'generations': 75, 'steps_multiplier': 3.20, 'jump_candles': 10},
        {'name': 'Elite 25',      'generations': 80, 'steps_multiplier': 3.25, 'jump_candles': 10}
    ]
    
    print("\n" + "="*70)
    print(f"  🧬 TREINAMENTO MICRONET RECORRENTE COM CURRICULUM LEARNING")
    print(f"  População: {pop_size} indivíduos")
    print("  Conexões: RECORRENTES (memória temporal)")
    print("  Símbolos: BTC/ETH/BNB (avaliação balanceada)")
    print("  Estratégia: OVERLAP PROGRESSIVO (jump < steps)")
    print("="*70 + "\n")
    
    # Mostrar fases com cálculo de overlap
    print("📚 FASES DE CURRICULUM LEARNING (COM OVERLAP PROGRESSIVO):")
    print("-" * 110)
    for idx, phase in enumerate(curriculum_phases, 1):
        phase_steps = int(max_steps * phase['steps_multiplier'])
        phase_hours = phase_steps * 5 / 60
        horizon_candles = phase['jump_candles'] * phase['generations']
        horizon_hours = horizon_candles * 5 / 60
        
        # Calcular % de overlap: quanto do episódio é revisitado
        overlap_pct = max(0, (phase_steps - phase['jump_candles']) / phase_steps * 100) if phase_steps > 0 else 0
        
        print(
            f"  Fase {idx:2} ({phase['name']:12}): {phase['generations']:2} gens | "
            f"Steps: {phase_steps:4} ({phase_hours:5.1f}h) | "
            f"Jump: {phase['jump_candles']:3} candles | "
            f"Overlap: {overlap_pct:5.1f}% | "
            f"Horizonte: {horizon_hours:6.1f}h"
        )
    print("-" * 110 + "\n")
    
    # 1. Carregar dados
    print("📅 Carregando dados de 2023-2024...")
    from src.features.builder import FeatureBuilder
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    all_dfs = []
    
    for symbol in symbols:
        data_path = Path(f"data/timeframe=5m/symbol={symbol}/candles.parquet")
        if not data_path.exists():
            print(f"⚠️  Arquivo não encontrado: {data_path}, pulando {symbol}")
            continue
        
        df = pd.read_parquet(data_path, engine="pyarrow")
        if 'timestamp' not in df.columns:
            print(f"⚠️  Coluna 'timestamp' ausente em {symbol}, pulando")
            continue
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        numeric_cols = [
            'open', 'high', 'low', 'close', 'volume', 'quote_volume',
            'trades_count', 'taker_buy_volume'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filtrar 2023-2024
        df_filtered = df[
            (df['timestamp'] >= datetime(2023, 1, 1)) &
            (df['timestamp'] < datetime(2025, 1, 1))
        ].copy()
        
        if df_filtered.empty:
            print(f"⚠️  Dataset vazio para {symbol}, pulando")
            continue
        
        builder = FeatureBuilder()
        df_filtered = builder.add_features(df_filtered)
        df_filtered['symbol'] = symbol
        all_dfs.append(df_filtered)
        
        print(f"✅ {symbol}: {len(df_filtered)} candles")
    
    if not all_dfs:
        raise ValueError("Nenhum símbolo foi carregado")
    
    # 2. Preparar dados por símbolo
    print(f"\n🎯 Preparando dados por símbolo...")
    symbols_data = []
    
    for df_symbol in all_dfs:
        symbol_name = df_symbol['symbol'].iloc[0]
        
        prices, micro_features = prepare_micro_data(
            df_symbol,
            micro_window=60  # 5 horas
        )
        
        symbols_data.append({
            'symbol': symbol_name,
            'prices': prices,
            'micro_features': micro_features
        })
        
        print(f"  ✅ {symbol_name}: {len(prices)} candles processados")
    
    # 3. Criar ambientes
    envs = create_environments_by_symbol(
        symbols_data=symbols_data,
        initial_capital=10000.0,
        commission=0.001
    )
    
    print(f"\n🧪 Ambientes criados: {len(envs)} (1 por símbolo)")
    for env in envs:
        print(f"  📊 {env.symbol}: {len(env.prices)} candles")
    
    # 4. Criar configuração NEAT
    print("\n⚙️  Criando configuração NEAT recorrente...")
    
    sample_micro_features = symbols_data[0]['micro_features']
    
    config_micro = create_neat_config_recurrent(
        input_nodes=sample_micro_features.shape[1],
        output_nodes=3,  # HOLD, BUY, SELL
        pop_size=pop_size
    )
    
    print(f"✅ População: {config_micro.pop_size} indivíduos")
    print(f"✅ Inputs: {config_micro.genome_config.num_inputs}")
    print(f"✅ Outputs: {config_micro.genome_config.num_outputs}")
    print(f"✅ Tipo: RECORRENTE (memória temporal)")
    
    # 5. Criar trainer
    trainer = MicroNetRecurrentTrainer(
        config_micro=config_micro,
        device=config.device
    )
    
    print(f"\n🚀 Iniciando evolução por {duration_minutes} minutos...")
    print(f"📊 Símbolos: {', '.join([env.symbol for env in envs])}")
    print(f"📈 Avaliação balanceada: TODOS os símbolos a cada geração")
    print(f"💰 Capital inicial: $10,000 por símbolo")
    print(f"🧬 População: {config_micro.pop_size} indivíduos")
    print(f"🔄 Conexões: RECORRENTES (feed_forward=False)")
    print(f"⚡ Steps base: {max_steps} (aumenta com curriculum)")
    print(f"🚀 Multiprocessing: {cpu_count()} workers\n")
    
    # 6. Loop de treinamento com curriculum learning
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    last_log_time = start_time
    
    generation = 0
    
    # Controle de curriculum learning
    current_phase_idx = 0
    generations_in_current_phase = 0
    phase_start_step_idx = 0  # step_idx inicial de cada fase
    
    # Histórico expandido
    history = {
        'time_min': [],
        'generation': [],
        'phase': [],
        'phase_generation': [],
        'max_steps': [],
        'jump_candles': [],
        'overlap_pct': [],
        'step_idx': [],
        'best_fitness': [],
        'avg_fitness': [],
        'std_fitness': [],
        'population_size': [],
        'species_count': [],
        'network_width': [],
        'network_depth': [],
        'eval_time_seconds': [],
        'fitness_improvement': []
    }
    
    last_fitness = -np.inf
    table_header_printed = False
    
    # Diretório de resultados
    results_dir = Path("training_results_micronet_recurrent")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "evolution_table.csv"
    
    # Sistema de fases progressivas
    initial_max_steps = max_steps
    
    while time.time() < end_time:
        elapsed = time.time() - start_time
        
        # ════════════════════════════════════════════════════════════════
        # CURRICULUM LEARNING: Controle de fases
        # ════════════════════════════════════════════════════════════════
        current_phase = curriculum_phases[current_phase_idx]
        generations_in_current_phase += 1
        
        # Calcular parâmetros da fase atual
        current_max_steps = int(max_steps * current_phase['steps_multiplier'])
        current_jump_candles = current_phase['jump_candles']
        
        # Verificar se deve avançar para próxima fase
        # (usar '>' para garantir que a geração final da fase seja executada com os parâmetros da fase)
        if generations_in_current_phase > current_phase['generations']:
            # Avançar step_idx com overlap de 80%
            overlap_ratio = 0.8
            phase_advance_candles = int(current_jump_candles * current_phase['generations'] * overlap_ratio)
            phase_start_step_idx += phase_advance_candles

            # Aplicar novo step_idx a todos os ambientes
            for env in envs:
                env.step_idx = min(phase_start_step_idx, len(env.prices) - 1)

            # Avançar para próxima fase se disponível
            if current_phase_idx < len(curriculum_phases) - 1:
                current_phase_idx += 1
                generations_in_current_phase = 0

            else:
                # Última fase: apenas resetar contador
                generations_in_current_phase = 0
        
        # Evoluir geração
        best_fitness, eval_time = trainer.evolve_generation(
            genomes=trainer.population.population,
            envs=envs,
            use_multiprocessing=True,
            max_steps=current_max_steps
        )
        
        generation += 1
        
        # Avançar step_idx após cada geração (jump definido pela fase)
        for env in envs:
            env.step_idx = min(env.step_idx + current_jump_candles, len(env.prices) - 1)
        
        # Log periódico
        current_time = time.time()
        if current_time - last_log_time >= log_interval_seconds or generation % 5 == 0:
            # Calcular dimensões da melhor rede
            best_genome_id = max(
                trainer.population.population,
                key=lambda g: trainer.population.population[g].fitness or -np.inf
            )
            best_genome = trainer.population.population[best_genome_id]
            
            network_depth = calculate_network_depth(best_genome, trainer.config_micro)
            network_width = calculate_network_width(best_genome, trainer.config_micro)
            
            # Estatísticas da população
            fitnesses = [g.fitness for g in trainer.population.population.values() if g.fitness is not None]
            avg_fitness = np.mean(fitnesses) if fitnesses else 0.0
            std_fitness = np.std(fitnesses) if fitnesses else 0.0
            species_count = len(trainer.population.species.species)
            population_size = len(trainer.population.population)
            fitness_improvement = best_fitness - last_fitness if last_fitness != -np.inf else 0.0
            
            # Step_idx médio dos ambientes
            avg_step_idx = np.mean([env.step_idx for env in envs])
            
            # Calcular overlap percentual da fase atual
            current_overlap_pct = max(0, (current_max_steps - current_jump_candles) / current_max_steps * 100) if current_max_steps > 0 else 0
            
            last_fitness = best_fitness
            
            if not table_header_printed:
                print("\nFase         | FaseGen | Tempo(m) | Ger   | Steps | Jump | Overlap% | BestFit     | AvgFit      | StdFit      | Spc | Pop | W | D | Eval(s)")
                print("-" * 170)
                table_header_printed = True
            
            print(
                f"{current_phase['name']:12} | {generations_in_current_phase:>7} | {elapsed/60:>8.1f} | {generation:>5} | {current_max_steps:>5} | "
                f"{current_jump_candles:>4} | {current_overlap_pct:>7.1f}% | "
                f"{best_fitness:>11.6f} | {avg_fitness:>11.6f} | {std_fitness:>11.6f} | "
                f"{species_count:>3} | {population_size:>3} | {network_width:>1} | {network_depth:>1} | {eval_time:>7.2f}"
            )
            
            # Adicionar ao histórico
            history['time_min'].append(elapsed / 60)
            history['generation'].append(generation)
            history['phase'].append(current_phase['name'])
            history['phase_generation'].append(generations_in_current_phase)
            history['max_steps'].append(current_max_steps)
            history['jump_candles'].append(current_jump_candles)
            history['overlap_pct'].append(current_overlap_pct)
            history['step_idx'].append(int(avg_step_idx))
            history['best_fitness'].append(best_fitness)
            history['avg_fitness'].append(avg_fitness)
            history['std_fitness'].append(std_fitness)
            history['population_size'].append(population_size)
            history['species_count'].append(species_count)
            history['network_width'].append(network_width)
            history['network_depth'].append(network_depth)
            history['eval_time_seconds'].append(eval_time)
            history['fitness_improvement'].append(fitness_improvement)
            
            # Salvar CSV a cada 50 gerações
            if generation % 50 == 0:
                df_history = pd.DataFrame(history)
                # Append to existing CSV instead of overwriting. Write header only if file doesn't exist.
                df_history.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
                print(f"💾 Checkpoint salvo: {csv_path}")
                
                # Limpar histórico em memória
                for key in history:
                    history[key] = []
            
            last_log_time = current_time
    
    # 7. Salvar modelo final
    print("-" * 150)
    print(f"\n✅ Treinamento concluído após {generation} gerações!")
    print(f"🏆 Melhor fitness: {trainer.best_fitness:.6f}")
    print(f"📊 Fase final: {current_phase['name']} (geração {generations_in_current_phase}/{current_phase['generations']})")
    print(f"⚡ Steps finais: {current_max_steps} ({current_max_steps*5/60:.1f}h de dados)")
    print(f"📍 Step_idx médio: {int(avg_step_idx)}")
    
    # Salvar CSV final
    if history['generation']:
        df_history = pd.DataFrame(history)
        # Append final rows to CSV; write header only if file missing
        df_history.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
    
    # Salvar melhor genoma
    best_genome_id = max(
        trainer.population.population,
        key=lambda g: trainer.population.population[g].fitness or -np.inf
    )
    best_genome = trainer.population.population[best_genome_id]
    
    model_path = results_dir / f"best_genome_gen{generation}.pkl"
    with open(model_path, 'wb') as f:
        import pickle
        pickle.dump(best_genome, f)
    
    print(f"💾 Modelo salvo: {model_path}")
    print(f"📊 Resultados salvos: {csv_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    train_micronet_recurrent(
        duration_minutes=60*6,  # 6 horas de treinamento
        log_interval_seconds=120,
        max_steps=20,  # 144 steps × 5min = 720min = 12 horas
        pop_size=100  # População de 170 indivíduos
    )
