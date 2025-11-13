"""
Treinamento com Reinforcement Learning (Policy Gradient)

Pipeline RL:
1. Agente (MacroNet + MicroNet) gera ação (long/short/hold)
2. Ambiente simula trade e retorna reward (lucro/prejuízo)
3. Política é atualizada com gradiente para maximizar reward acumulado
4. Usa dados de 2024 completo para treinamento robusto

Vantagens:
- Otimiza DIRETAMENTE o lucro (não MSE)
- Aprende política ótima de trading
- Considera custo de transação e slippage
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
from typing import List, Tuple
import matplotlib.pyplot as plt

from src.pipeline import TradingPipeline
from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class TradingEnvironment:
    """Ambiente de trading para RL"""
    
    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1%
        max_position: float = 1.0
    ):
        self.prices = prices
        self.features = features
        self.initial_capital = initial_capital
        self.commission = commission
        self.max_position = max_position
        
        self.reset()
    
    def reset(self):
        """Resetar ambiente"""
        self.current_step = 0
        self.cash = self.initial_capital
        self.position = 0.0  # -1 a +1 (short, neutro, long)
        self.portfolio_value = self.initial_capital
        self.trade_history = []
        
        return self._get_state()
    
    def _get_state(self):
        """Obter estado atual"""
        if self.current_step >= len(self.features):
            return None
        
        return {
            'features': self.features[self.current_step],
            'price': self.prices[self.current_step],
            'position': self.position,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value
        }
    
    def step(self, action: int) -> Tuple[dict, float, bool]:
        """
        Executar ação
        
        Actions:
            0 = HOLD (manter posição)
            1 = BUY (long)
            2 = SELL (short)
        
        Returns:
            state, reward, done
        """
        if self.current_step >= len(self.prices) - 1:
            return None, 0.0, True
        
        current_price = self.prices[self.current_step]
        next_price = self.prices[self.current_step + 1]
        
        # Calcular reward baseado na ação
        reward = 0.0
        
        # Executar ação
        if action == 1:  # BUY (long)
            if self.position < self.max_position:
                # Aumentar posição long
                trade_size = self.max_position - self.position
                cost = trade_size * current_price * (1 + self.commission)
                
                if cost <= self.cash:
                    self.cash -= cost
                    self.position += trade_size
                    self.trade_history.append(('BUY', self.current_step, current_price, trade_size))
        
        elif action == 2:  # SELL (short)
            if self.position > -self.max_position:
                # Aumentar posição short
                trade_size = self.position - (-self.max_position)
                proceeds = trade_size * current_price * (1 - self.commission)
                
                self.cash += proceeds
                self.position -= trade_size
                self.trade_history.append(('SELL', self.current_step, current_price, trade_size))
        
        # else: action == 0 (HOLD) - não faz nada
        
        # Avançar step
        self.current_step += 1
        
        # Calcular P&L da posição
        price_change = next_price - current_price
        position_pnl = self.position * price_change
        
        # Reward = P&L percentual
        reward = (position_pnl / self.initial_capital) * 100  # Em %
        
        # Atualizar portfolio value
        position_value = self.position * next_price
        self.portfolio_value = self.cash + position_value
        
        # Penalidade por trades excessivos (reduz overtrading)
        if action != 0:
            reward -= 0.01  # Pequena penalidade por trade
        
        done = self.current_step >= len(self.prices) - 1
        
        next_state = self._get_state()
        
        return next_state, reward, done


class PolicyNetwork(nn.Module):
    """Rede de política para RL"""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_actions: int = 3  # HOLD, BUY, SELL
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim),  # +2 para position e cash
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state_features, position, cash_ratio):
        """
        Args:
            state_features: (batch, feature_dim)
            position: (batch,) posição atual
            cash_ratio: (batch,) cash / initial_capital
        """
        # Concatenar features com posição e cash
        x = torch.cat([
            state_features,
            position.unsqueeze(-1),
            cash_ratio.unsqueeze(-1)
        ], dim=-1)
        
        return self.network(x)


class RLTrainer:
    """Treinador com Reinforcement Learning"""
    
    def __init__(
        self,
        pipeline: TradingPipeline,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,  # Discount factor
        device: str = "cpu"
    ):
        self.pipeline = pipeline
        self.gamma = gamma
        self.device = torch.device(device)
        
        # Inicializar rede de política
        self.state_dim = None  # Será definido depois
        self.policy_net = None
        self.optimizer = None
        self.learning_rate = learning_rate
        
        # Histórico
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []
    
    def build_policy(self, state_dim: int):
        """Construir rede de política"""
        self.state_dim = state_dim
        self.policy_net = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=128,
            num_actions=3
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )
        
        logger.info("policy_network_built", state_dim=state_dim)
    
    def select_action(self, state: dict) -> Tuple[int, torch.Tensor]:
        """
        Selecionar ação usando política
        
        Returns:
            action, log_prob
        """
        # Preparar estado
        features = torch.FloatTensor(state['features']).unsqueeze(0).to(self.device)
        position = torch.FloatTensor([state['position']]).to(self.device)
        cash_ratio = torch.FloatTensor([state['cash'] / 10000.0]).to(self.device)
        
        # Forward
        action_probs = self.policy_net(features, position, cash_ratio)
        
        # Sample ação
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def train_episode(self, env: TradingEnvironment) -> float:
        """
        Treinar um episódio completo
        
        Returns:
            total_reward
        """
        state = env.reset()
        
        # Coletar trajetória
        log_probs = []
        rewards = []
        
        while True:
            action, log_prob = self.select_action(state)
            next_state, reward, done = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            if done:
                break
            
            state = next_state
        
        # Calcular returns descontados
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalizar returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calcular loss de política
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Clip gradientes para estabilidade
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        total_reward = sum(rewards)
        
        return total_reward, env.portfolio_value


def prepare_data_2024(symbol: str = "BTCUSDT") -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Baixar dados de 2024 completo
    
    Returns:
        dataframe, prices, features
    """
    print("\n📅 Baixando dados de 2024 completo...")
    
    # Calcular dias de 2024
    start_2024 = datetime(2024, 1, 1)
    end_2024 = datetime(2024, 12, 31)
    days_2024 = (end_2024 - start_2024).days
    
    print(f"   Período: {start_2024.date()} até {end_2024.date()}")
    print(f"   Total: {days_2024} dias")
    
    # Fetch dados
    from src.ingest.binance import BinanceIngestor
    ingestor = BinanceIngestor()
    
    df = ingestor.fetch_candles(symbol, days_back=days_2024)
    
    # Filtrar apenas 2024
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    df_2024 = df[
        (df['datetime'] >= start_2024) &
        (df['datetime'] < datetime(2025, 1, 1))
    ].copy()
    
    print(f"✅ {len(df_2024)} candles de 2024")
    
    # Adicionar features
    from src.features.builder import FeatureBuilder
    builder = FeatureBuilder()
    df_2024 = builder.add_features(df_2024)
    
    # Extrair prices e features numéricas
    prices = df_2024['close'].values.astype(np.float32)
    
    numeric_cols = []
    for col in df_2024.columns:
        if np.issubdtype(df_2024[col].dtype, np.number) and col not in ['open_time']:
            numeric_cols.append(col)
    
    features = df_2024[numeric_cols].fillna(0).values.astype(np.float32)
    
    print(f"📊 Features: {len(numeric_cols)} colunas")
    
    return df_2024, prices, features


def train_rl(
    duration_minutes: int = 10,
    num_episodes_per_update: int = 5,
    log_interval_seconds: int = 30
):
    """
    Treinar com Reinforcement Learning
    
    Args:
        duration_minutes: Tempo total de treinamento
        num_episodes_per_update: Episódios antes de logar progresso
        log_interval_seconds: Intervalo de log
    """
    print("\n" + "="*70)
    print("  🎮 TREINAMENTO COM REINFORCEMENT LEARNING")
    print("  Policy Gradient + Trading Environment")
    print("  Dados: 2024 completo")
    print("="*70 + "\n")
    
    # 1. Preparar dados de 2024
    df_2024, prices, features = prepare_data_2024("BTCUSDT")
    
    # 2. Criar pipeline
    pipeline = TradingPipeline()
    
    # 3. Criar trainer RL
    trainer = RLTrainer(
        pipeline=pipeline,
        learning_rate=0.0003,
        gamma=0.99,
        device=config.device
    )
    
    # Build policy network
    trainer.build_policy(state_dim=features.shape[1])
    
    # 4. Criar ambiente
    env = TradingEnvironment(
        prices=prices,
        features=features,
        initial_capital=10000.0,
        commission=0.001
    )
    
    print(f"\n🚀 Iniciando treinamento RL por {duration_minutes} minutos...")
    print(f"📊 Dataset: {len(prices)} candles de 2024")
    print(f"🎯 Objetivo: Maximizar lucro acumulado")
    print(f"💰 Capital inicial: $10,000")
    print(f"⏰ Log a cada {log_interval_seconds}s\n")
    
    # Histórico
    history = {
        'time_min': [],
        'episode': [],
        'avg_reward': [],
        'best_reward': [],
        'avg_portfolio': [],
        'best_portfolio': []
    }
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    last_log_time = start_time
    
    episode = 0
    best_reward = float('-inf')
    best_portfolio = 0.0
    
    recent_rewards = []
    recent_portfolios = []
    
    while time.time() < end_time:
        # Treinar episódio
        total_reward, final_portfolio = trainer.train_episode(env)
        
        episode += 1
        recent_rewards.append(total_reward)
        recent_portfolios.append(final_portfolio)
        
        # Track best
        if total_reward > best_reward:
            best_reward = total_reward
        if final_portfolio > best_portfolio:
            best_portfolio = final_portfolio
        
        # Log periódico
        current_time = time.time()
        if current_time - last_log_time >= log_interval_seconds:
            elapsed = current_time - start_time
            remaining = end_time - current_time
            progress = (elapsed / (duration_minutes * 60)) * 100
            
            # Calcular médias
            avg_reward = np.mean(recent_rewards[-20:]) if recent_rewards else 0
            avg_portfolio = np.mean(recent_portfolios[-20:]) if recent_portfolios else 10000
            
            # Return percentual
            avg_return_pct = ((avg_portfolio - 10000) / 10000) * 100
            best_return_pct = ((best_portfolio - 10000) / 10000) * 100
            
            print(f"\n{'─'*70}")
            print(f"⏱️  Tempo: {elapsed/60:.1f}min / {duration_minutes}min ({progress:.1f}%)")
            print(f"🎮 Episódio: {episode}")
            print(f"💰 Portfolio (médio últimos 20): ${avg_portfolio:,.2f} ({avg_return_pct:+.2f}%)")
            print(f"🏆 Melhor Portfolio: ${best_portfolio:,.2f} ({best_return_pct:+.2f}%)")
            print(f"📈 Reward Médio: {avg_reward:.2f} | Melhor: {best_reward:.2f}")
            print(f"⏳ Restante: {remaining/60:.1f}min")
            print(f"{'─'*70}")
            
            # Salvar histórico
            history['time_min'].append(elapsed / 60)
            history['episode'].append(episode)
            history['avg_reward'].append(avg_reward)
            history['best_reward'].append(best_reward)
            history['avg_portfolio'].append(avg_portfolio)
            history['best_portfolio'].append(best_portfolio)
            
            last_log_time = current_time
    
    # Final
    total_time = time.time() - start_time
    
    # Salvar modelo
    output_dir = Path("./training_results_rl")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    policy_path = output_dir / "policy_network.pt"
    torch.save(trainer.policy_net.state_dict(), policy_path)
    
    # Plotar gráfico
    plot_rl_history(history, output_dir, total_time, episode, best_portfolio)
    
    print(f"\n✅ Treinamento RL completo: {episode} episódios em {total_time/60:.1f}min")
    print(f"🏆 Melhor portfolio: ${best_portfolio:,.2f} ({((best_portfolio-10000)/10000)*100:+.2f}%)")
    print(f"💾 Modelo e gráficos salvos em: {output_dir}/")


def plot_rl_history(history: dict, output_dir: Path, total_time: float, episodes: int, best_portfolio: float):
    """Plotar histórico de treinamento RL"""
    
    if len(history['time_min']) == 0:
        print("⚠️  Sem dados para plotar")
        return
    
    print(f"\n📊 Gerando gráfico de evolução RL...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Reinforcement Learning - {total_time/60:.1f} min, {episodes} episódios', 
                 fontsize=14, fontweight='bold')
    
    time_axis = history['time_min']
    
    # 1. Portfolio Value
    ax1 = axes[0, 0]
    ax1.plot(time_axis, history['avg_portfolio'], 'b-', linewidth=2, label='Avg Portfolio')
    ax1.plot(time_axis, history['best_portfolio'], 'g--', linewidth=2, label='Best Portfolio')
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial ($10k)')
    ax1.set_xlabel('Tempo (minutos)')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Evolução do Portfolio')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.1f}k'))
    
    # 2. Return %
    ax2 = axes[0, 1]
    avg_returns = [((p - 10000) / 10000) * 100 for p in history['avg_portfolio']]
    best_returns = [((p - 10000) / 10000) * 100 for p in history['best_portfolio']]
    ax2.plot(time_axis, avg_returns, 'b-', linewidth=2, label='Avg Return')
    ax2.plot(time_axis, best_returns, 'g--', linewidth=2, label='Best Return')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Tempo (minutos)')
    ax2.set_ylabel('Return (%)')
    ax2.set_title('Retorno Percentual')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Rewards
    ax3 = axes[1, 0]
    ax3.plot(time_axis, history['avg_reward'], 'purple', linewidth=2, label='Avg Reward')
    ax3.plot(time_axis, history['best_reward'], 'orange', linewidth=2, label='Best Reward')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Tempo (minutos)')
    ax3.set_ylabel('Reward')
    ax3.set_title('Recompensas Acumuladas')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Episodes
    ax4 = axes[1, 1]
    ax4.plot(time_axis, history['episode'], 'red', linewidth=2)
    ax4.set_xlabel('Tempo (minutos)')
    ax4.set_ylabel('Episódios Completados')
    ax4.set_title('Progresso de Episódios')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'rl_training_evolution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico salvo: {plot_path}")
    
    plt.close()


if __name__ == "__main__":
    train_rl(
        duration_minutes=10,
        num_episodes_per_update=5,
        log_interval_seconds=30
    )
