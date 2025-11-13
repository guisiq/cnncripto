# 🎮 Treinamento com Reinforcement Learning

## 🚀 O Que Mudou?

### ❌ **Antes: Supervised Learning**
```
Input → Rede Neural → Predição de Retorno
Loss = MSE(predição, retorno_real)
```

**Problema**: Otimiza MSE, não lucro!

### ✅ **Agora: Reinforcement Learning**
```
Estado → Política (RL Agent) → Ação (Buy/Sell/Hold)
Ambiente → Executa Trade → Reward (Lucro/Prejuízo)
Política → Atualizada com Gradient Ascent
```

**Vantagem**: Otimiza DIRETAMENTE o lucro!

---

## 🧠 Arquitetura RL

### 1. **Agente (Policy Network)**
```
Input: [Features, Position, Cash]
         ↓
    Linear(128)
         ↓
      ReLU
         ↓
   Linear(64)
         ↓
      ReLU
         ↓
   Linear(3)  → Softmax
         ↓
  [P(HOLD), P(BUY), P(SELL)]
```

### 2. **Ambiente de Trading**
```python
class TradingEnvironment:
    - Estado: Features + Posição + Cash
    - Ações: 0=HOLD, 1=BUY (long), 2=SELL (short)
    - Reward: Lucro percentual - Custo de transação
```

### 3. **Algoritmo: Policy Gradient**
```
1. Coletar episódio: (s₀, a₀, r₀), (s₁, a₁, r₁), ...
2. Calcular returns: G_t = Σ γᵏ r_{t+k}
3. Loss: -Σ log π(a_t|s_t) * G_t
4. Backprop e atualizar pesos
5. Repetir
```

---

## 📊 Dados: 2024 Completo

### Antes vs Agora

| Métrica | Antes | Agora |
|---------|-------|-------|
| **Período** | 5-30 dias | 365 dias (2024) |
| **Candles** | ~500-8,000 | ~105,120 (ano todo) |
| **Timeframe** | 5min | 5min |
| **Robustez** | Baixa | Alta ✅ |

### Download Automático
```python
# Baixa automaticamente todos os dados de 2024
start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)
df = fetch_candles("BTCUSDT", days_back=365)
```

---

## 🎯 Vantagens do RL

### 1. **Otimização Direta**
- ✅ Maximiza lucro real (não MSE)
- ✅ Considera custos de transação
- ✅ Aprende política ótima de trading

### 2. **Exploration vs Exploitation**
- ✅ Explora diferentes estratégias
- ✅ Descobre padrões não óbvios
- ✅ Não fica preso em mínimos locais

### 3. **Aprendizado Contínuo**
- ✅ Pesos evoluem a cada episódio
- ✅ Adaptação a diferentes mercados
- ✅ Melhora com mais dados

### 4. **Métricas Realistas**
- ✅ Portfolio value
- ✅ Return percentual
- ✅ Sharpe ratio (pode ser adicionado)

---

## 📈 Métricas Monitoradas

### Durante Treinamento

```
⏱️  Tempo: 2.5min / 10min (25.0%)
🎮 Episódio: 127
💰 Portfolio (médio últimos 20): $10,450.23 (+4.50%)
🏆 Melhor Portfolio: $11,234.56 (+12.35%)
📈 Reward Médio: 8.45 | Melhor: 23.67
⏳ Restante: 7.5min
```

### Gráfico Final (4 painéis)

1. **Portfolio Value** 📊
   - Evolução do capital
   - Linha base: $10,000 inicial
   - Avg vs Best

2. **Return %** 📈
   - Retorno percentual
   - Positivo = lucro, Negativo = prejuízo

3. **Rewards** 🎁
   - Recompensas acumuladas
   - Indica aprendizado

4. **Episódios** 🎮
   - Progresso de treinamento
   - Episódios completados

---

## 🔧 Parâmetros Importantes

### Learning Rate
```python
learning_rate = 0.0003  # Baixo para estabilidade
```

### Gamma (Discount Factor)
```python
gamma = 0.99  # Valoriza recompensas futuras
```

### Comissão
```python
commission = 0.001  # 0.1% por trade
```

### Capital Inicial
```python
initial_capital = 10000.0  # $10k
```

---

## 🚀 Como Usar

### Executar Treinamento
```bash
conda run -n cnncripto python train_reinforcement_learning.py
```

### Ajustar Tempo
```python
train_rl(
    duration_minutes=10,  # Mudar aqui
    log_interval_seconds=30
)
```

### Usar Modelo Treinado
```python
import torch
from train_reinforcement_learning import PolicyNetwork

# Carregar modelo
policy = PolicyNetwork(state_dim=13)
policy.load_state_dict(torch.load('training_results_rl/policy_network.pt'))

# Predizer ação
action_probs = policy(features, position, cash_ratio)
action = action_probs.argmax()  # 0=HOLD, 1=BUY, 2=SELL
```

---

## 📁 Arquivos Gerados

```
training_results_rl/
├── policy_network.pt              # Modelo treinado
└── rl_training_evolution.png      # Gráfico de evolução
```

---

## 🎓 Diferença vs Supervised Learning

| Aspecto | Supervised | Reinforcement |
|---------|------------|---------------|
| **Objetivo** | Minimizar MSE | Maximizar Lucro |
| **Target** | Retorno futuro | Não tem (descobre) |
| **Feedback** | Imediato | Delayed reward |
| **Exploração** | Não há | Sim (via sampling) |
| **Adaptação** | Estática | Dinâmica ✅ |

---

## 🔮 Próximos Passos

### 1. **A2C/A3C** (Actor-Critic)
- Duas redes: Actor (política) + Critic (valor)
- Mais estável que Policy Gradient puro
- Convergência mais rápida

### 2. **PPO** (Proximal Policy Optimization)
- SOTA em RL
- Usado por OpenAI
- Muito estável

### 3. **Replay Buffer**
- Armazenar experiências passadas
- Treinar com mini-batches
- Off-policy learning

### 4. **Multi-Asset**
- Treinar em múltiplos pares (BTC, ETH, BNB)
- Generalização melhor
- Portfolio diversificado

---

## 🐛 Troubleshooting

### Problema: Portfolio sempre perdendo
**Solução:**
- Reduzir learning rate
- Aumentar exploração inicial
- Verificar comissões muito altas

### Problema: Muitos trades (overtrading)
**Solução:**
- Aumentar penalidade por trade
- Ajustar reward function

### Problema: Não aprende (estagnado)
**Solução:**
- Aumentar learning rate
- Reduzir gamma (focar em recompensas imediatas)
- Verificar normalização de features

---

**Data:** 13 de novembro de 2025  
**Versão:** 3.0 - Reinforcement Learning com dados de 2024
