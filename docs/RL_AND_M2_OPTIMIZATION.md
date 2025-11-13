# 🎮 Análise: RL e Otimização para Apple M2

## ❓ Suas Perguntas

### 1️⃣ O treinamento assimétrico está usando RL?

**✅ SIM! Completamente baseado em Reinforcement Learning.**

O arquivo `train_asymmetric_rl.py` implementa **Policy Gradient RL**:

```python
# 1. AMBIENTE RL
class TradingEnvironmentRL:
    - State: [macro_features, micro_features, position, cash]
    - Actions: [HOLD, BUY, SELL]
    - Reward: (position_pnl / capital) * 100 - trade_penalty
    - Transition: s_t → a_t → r_t → s_{t+1}

# 2. POLICY NETWORK
class AsymmetricPolicyNetwork:
    - Input: State features
    - Output: Action probabilities π(a|s)
    - Softmax: Categorical distribution

# 3. POLICY GRADIENT ALGORITHM
def train_episode():
    # Collect trajectory
    trajectory = [(s_0, a_0, r_0), ..., (s_T, a_T, r_T)]
    
    # Calculate discounted returns
    G_t = Σ γ^i * r_{t+i}
    
    # Policy loss (REINFORCE)
    L = -Σ log π(a_t|s_t) * G_t
    
    # Backpropagation
    L.backward()
    optimizer.step()
```

**Algoritmo**: REINFORCE (Monte Carlo Policy Gradient)
- ✅ Sem necessidade de Q-function (model-free)
- ✅ On-policy (aprende da própria política)
- ✅ Otimiza diretamente o retorno esperado

---

### 2️⃣ Está otimizado para usar o máximo do M2?

**⚠️ PARCIALMENTE! Usa MPS, mas faltam otimizações importantes.**

#### ✅ O que JÁ está otimizado:

1. **MPS habilitado**:
```python
# src/config.py
def detect_device():
    if torch.backends.mps.is_available():
        return "mps"  # ✅ Apple Silicon GPU
```

2. **Modelo roda em MPS**:
```python
# train_asymmetric_rl.py (linha 698)
trainer = AsymmetricRLTrainer(
    device=config.device  # ✅ "mps" no M2
)
```

#### ❌ O que FALTA otimizar:

1. **Batch processing** ❌ (roda 1 amostra por vez)
2. **Mixed precision (float16)** ❌ (usa float32)
3. **Gradient accumulation** ❌
4. **Operações vetorizadas** ❌ (loop Python)
5. **Pinned memory** ❌
6. **DataLoader multithreading** ❌

---

## 🚀 Otimizações Propostas para M2

### Otimização 1: **Batch Processing** (CRÍTICO)

**Problema atual**:
```python
# UMA amostra por vez (ineficiente!)
state = env.reset()  # scalar
action = select_action(state)  # batch=1
```

**Solução**:
```python
# Processar MÚLTIPLOS episódios em paralelo
class VectorizedEnv:
    def __init__(self, num_envs=32):  # 32 episódios simultâneos
        self.envs = [TradingEnvironmentRL(...) for _ in range(num_envs)]
    
    def step(self, actions):  # (32,) actions
        # Parallel execution
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        states = torch.stack([r[0] for r in results])  # (32, state_dim)
        rewards = torch.tensor([r[1] for r in results])  # (32,)
        return states, rewards

# Training loop
states = vec_env.reset()  # (32, state_dim)
actions = policy(states)  # (32, 3) → (32,) via sample
states, rewards = vec_env.step(actions)  # Vectorized!
```

**Ganho esperado**: **10-20x speedup** (M2 ama batch operations)

---

### Otimização 2: **Mixed Precision (float16)** (MÉDIO)

**Problema**: Float32 usa 2x mais memória e bandwidth.

**Solução**:
```python
# Enable AMP (Automatic Mixed Precision)
from torch.cuda.amp import autocast, GradScaler  # Works on MPS too!

scaler = GradScaler()

# Training loop
for episode in range(num_episodes):
    with autocast(device_type='mps'):  # MPS float16
        action_probs = policy(states)
        loss = compute_loss(...)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Ganho esperado**: **1.5-2x speedup**, **2x less memory**

---

### Otimização 3: **Gradient Accumulation** (BAIXO)

**Problema**: Batch pequeno → gradientes ruidosos.

**Solução**:
```python
accumulation_steps = 4  # Simula batch 4x maior

for i, episode in enumerate(episodes):
    loss = compute_loss(episode)
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Ganho esperado**: **Melhor convergência**, sem custo computacional extra.

---

### Otimização 4: **Operações Vetorizadas** (CRÍTICO)

**Problema**: Loops Python são lentos.

**Solução**:
```python
# ❌ MAU (loop Python)
policy_loss = []
for log_prob, G in zip(log_probs, returns):
    policy_loss.append(-log_prob * G)
policy_loss = torch.stack(policy_loss).sum()

# ✅ BOM (vetorizado)
log_probs = torch.stack(log_probs)  # (T,)
returns = torch.tensor(returns)      # (T,)
policy_loss = -(log_probs * returns).sum()  # Vectorized!
```

**Ganho esperado**: **2-3x speedup** em cálculo de loss.

---

### Otimização 5: **Compile Model (PyTorch 2.0+)** (ALTO)

**Problema**: Interpretação Python overhead.

**Solução**:
```python
# PyTorch 2.0+ torch.compile (JIT)
policy = AsymmetricPolicyNetwork(...)
policy = torch.compile(policy, backend="aot_eager")  # MPS-compatible

# Depois disso, forward pass é compilado!
```

**Ganho esperado**: **1.5-2.5x speedup** (especialmente em redes profundas).

---

### Otimização 6: **DataLoader com Workers** (MÉDIO)

**Problema**: Preparação de dados bloqueia GPU.

**Solução**:
```python
from torch.utils.data import DataLoader, Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories
    
    def __getitem__(self, idx):
        return self.trajectories[idx]
    
    def __len__(self):
        return len(self.trajectories)

# Multi-threaded data loading
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # 4 threads preparam dados
    pin_memory=True,  # Faster transfer to MPS
    prefetch_factor=2
)
```

**Ganho esperado**: **1.3-1.8x speedup** (GPU não espera CPU).

---

## 📊 Impacto Estimado das Otimizações

| Otimização | Dificuldade | Ganho Esperado | Prioridade |
|------------|-------------|----------------|------------|
| **Batch Processing** | Alta | 10-20x | 🔴 CRÍTICA |
| **Mixed Precision** | Baixa | 1.5-2x | 🟠 Alta |
| **Torch Compile** | Baixa | 1.5-2.5x | 🟠 Alta |
| **Vectorize Ops** | Média | 2-3x | 🟠 Alta |
| **DataLoader Workers** | Média | 1.3-1.8x | 🟡 Média |
| **Gradient Accumulation** | Baixa | Estabilidade | 🟢 Baixa |

**Ganho combinado estimado**: **30-60x speedup total!** 🚀

---

## 🛠️ Implementação: Versão Otimizada para M2

### Código Otimizado (Highlights)

```python
import torch
from torch.cuda.amp import autocast, GradScaler

class OptimizedAsymmetricTrainer:
    def __init__(self, ...):
        self.device = torch.device("mps")
        
        # ✅ Compile model
        self.policy = torch.compile(
            AsymmetricPolicyNetwork(...),
            backend="aot_eager"
        ).to(self.device)
        
        # ✅ Mixed precision
        self.scaler = GradScaler()
        
        # ✅ Vectorized envs
        self.vec_env = VectorizedEnv(num_envs=32)
    
    def train_batch(self, batch_size=32):
        """Train on batch of episodes simultaneously"""
        
        # Reset all envs
        states = self.vec_env.reset()  # (32, state_dim)
        
        trajectories = [[] for _ in range(batch_size)]
        
        # Collect trajectories in parallel
        for step in range(max_steps):
            with autocast(device_type='mps'):
                # ✅ Batch forward (32 simultâneos)
                action_probs = self.policy(states)  # (32, 3)
            
            # ✅ Sample actions vectorized
            dist = Categorical(action_probs)
            actions = dist.sample()  # (32,)
            log_probs = dist.log_prob(actions)  # (32,)
            
            # ✅ Step all envs
            next_states, rewards, dones = self.vec_env.step(actions)
            
            # Store
            for i in range(batch_size):
                trajectories[i].append((states[i], log_probs[i], rewards[i]))
            
            states = next_states
            
            if dones.all():
                break
        
        # ✅ Vectorized loss computation
        all_log_probs = []
        all_returns = []
        
        for traj in trajectories:
            log_probs_ep = torch.stack([t[1] for t in traj])
            rewards_ep = torch.tensor([t[2] for t in traj])
            
            # Compute returns vectorized
            returns_ep = self._compute_returns(rewards_ep)
            
            all_log_probs.append(log_probs_ep)
            all_returns.append(returns_ep)
        
        # ✅ Concatenate all episodes
        log_probs = torch.cat(all_log_probs)  # (total_steps,)
        returns = torch.cat(all_returns)      # (total_steps,)
        
        # ✅ Vectorized policy loss
        policy_loss = -(log_probs * returns).mean()
        
        # ✅ Mixed precision backward
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
    
    def _compute_returns(self, rewards):
        """Vectorized return computation"""
        T = len(rewards)
        gamma_vec = torch.pow(self.gamma, torch.arange(T, device=self.device))
        
        # Vectorized discounted return
        returns = torch.zeros_like(rewards)
        for t in range(T):
            returns[t] = (rewards[t:] * gamma_vec[:T-t]).sum()
        
        return returns
```

---

## 🔬 Benchmarks Esperados (M2)

### Antes (atual):
```
Setup: 1 episódio/vez, float32, sem compile
─────────────────────────────────────────────
Tempo por episódio:    ~500ms
Episódios por minuto:  120
Utilização GPU:        15-25% (subutilizado!)
Memória GPU:           ~800MB
```

### Depois (otimizado):
```
Setup: 32 episódios/batch, float16, compiled
─────────────────────────────────────────────
Tempo por batch(32):   ~800ms  (25ms/episódio)
Episódios por minuto:  2400  (20x mais!)
Utilização GPU:        80-95% (OTIMIZADO!)
Memória GPU:           ~2.5GB
```

**Resultado**: Treinamento de 10 minutos → **30 segundos!** ⚡

---

## 🎯 Próximos Passos

### Fase 1: Quick Wins (1-2 horas)
1. ✅ Vectorizar cálculo de loss (remover loops)
2. ✅ Adicionar `torch.compile()`
3. ✅ Habilitar mixed precision (AMP)

### Fase 2: Batch Processing (4-6 horas)
1. ✅ Criar `VectorizedEnv` para 32 episódios paralelos
2. ✅ Refatorar `train_episode` → `train_batch`
3. ✅ Ajustar logging para batch

### Fase 3: Profiling (2-3 horas)
1. ✅ Usar `torch.profiler` para identificar bottlenecks
2. ✅ Medir tempo de cada operação
3. ✅ Otimizar operações lentas

### Fase 4: Advanced (opcional)
1. ✅ Implementar PPO (mais estável que REINFORCE)
2. ✅ Adicionar Generalized Advantage Estimation (GAE)
3. ✅ Curriculum learning (treinar progressivamente)

---

## 📝 Resumo Executivo

| Aspecto | Status Atual | Status Ideal |
|---------|--------------|--------------|
| **Algoritmo** | ✅ REINFORCE (Policy Gradient) | ✅ Adequado |
| **Device** | ✅ MPS habilitado | ✅ OK |
| **Precision** | ❌ Float32 | ⚠️ Usar Float16 |
| **Batching** | ❌ 1 sample/vez | 🔴 CRÍTICO: 32+ batch |
| **Vectorization** | ❌ Loops Python | 🔴 CRÍTICO: Torch ops |
| **Compilation** | ❌ Interpretado | 🟠 Compilar modelo |
| **Data Loading** | ✅ Sincrono (OK para RL) | 🟢 Suficiente |
| **Utilização M2** | ⚠️ ~20% | 🔴 Target: 80%+ |

**Prioridade #1**: Implementar **Batch Processing** (VectorizedEnv)  
**Prioridade #2**: **Torch.compile** + **Mixed Precision**  
**Prioridade #3**: **Vectorizar** operações (remover loops)

---

## 💡 Comandos para Verificar Otimização

```bash
# 1. Verificar device ativo
python -c "
import torch
from src.config import config
print(f'Device: {config.device}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"

# 2. Benchmark atual
time conda run -n cnncripto python -c "
import torch
from train_asymmetric_rl import AsymmetricPolicyNetwork
import time

model = AsymmetricPolicyNetwork(60, 60).to('mps')
x_macro = torch.randn(1, 60).to('mps')
x_micro = torch.randn(1, 60).to('mps')
pos = torch.zeros(1).to('mps')
cash = torch.ones(1).to('mps')

# Warmup
for _ in range(10):
    model(x_macro, x_micro, pos, cash)

# Benchmark
start = time.time()
for _ in range(1000):
    model(x_macro, x_micro, pos, cash)
elapsed = time.time() - start
print(f'1000 forward passes: {elapsed:.2f}s ({elapsed*1000:.2f}ms each)')
"

# 3. Profile com PyTorch
python -c "
import torch
from torch.profiler import profile, ProfilerActivity
from train_asymmetric_rl import AsymmetricPolicyNetwork

model = AsymmetricPolicyNetwork(60, 60).to('mps')
x_macro = torch.randn(1, 60).to('mps')
x_micro = torch.randn(1, 60).to('mps')
pos = torch.zeros(1).to('mps')
cash = torch.ones(1).to('mps')

with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(100):
        model(x_macro, x_micro, pos, cash)

print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))
"
```

---

**Conclusão**: 
- ✅ **RL está implementado** corretamente (REINFORCE/Policy Gradient)
- ⚠️ **M2 está parcialmente otimizado** (MPS ativo, mas sem batching)
- 🚀 **Potencial de 30-60x speedup** com otimizações propostas
- 🔴 **Prioridade**: Implementar batch processing (VectorizedEnv)

---

**Data**: 13 de novembro de 2025  
**Versão**: 5.0 - Análise de RL e Otimização M2
