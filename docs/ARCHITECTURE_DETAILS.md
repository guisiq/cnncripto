# 🏗️ Arquitetura Detalhada: Treinamento Assimétrico (1:10)

## 📐 Estrutura Completa da Rede

### Visão Geral
```
Input Features → [MacroNet] → Macro Embedding (128)
                                      ↓
Input Features → [MicroNet] → Micro Features (32)
                                      ↓
Position + Cash → [Context] → State (2)
                                      ↓
                    [Concatenate: 128 + 32 + 2 = 162]
                                      ↓
                          [Decision Head]
                                      ↓
                      [Softmax: 3 actions]
                      [HOLD, BUY, SELL]
```

---

## 🔵 MacroNet (Encoder de Longo Prazo)

### Objetivo
Captura tendências de **longo prazo** (41 horas de contexto, 492 candles)

### Arquitetura

#### Entrada
- **Dimensão**: Variável (depende das features agregadas)
- **Features agregadas de 492 candles**:
  - Mean de todas features: N features
  - Std de todas features: N features  
  - Último valor: N features
  - **Total**: 3N features (exemplo: se 20 features → 60 inputs)

#### Camadas

| Camada | Tipo | Entrada | Saída | Ativação | Dropout |
|--------|------|---------|-------|----------|---------|
| **Layer 1** | Linear | 3N (ex: 60) | 256 | ReLU | - |
| **Dropout 1** | Dropout | 256 | 256 | - | 20% |
| **Layer 2** | Linear | 256 | 128 | ReLU | - |

#### Saída
- **Dimensão**: 128 (macro embedding)
- **Significado**: Representação compacta do contexto de longo prazo

### Parâmetros (exemplo com N=20 features)

```python
# Input: 60 features (20 * 3)

Layer 1: 60 → 256
  Weights: 60 × 256 = 15,360
  Biases:  256 = 256
  Subtotal: 15,616

Layer 2: 256 → 128
  Weights: 256 × 128 = 32,768
  Biases:  128 = 128
  Subtotal: 32,896

TOTAL MacroNet: 48,512 parâmetros
```

### Características
- ✅ **Updates**: 1x por ciclo (a cada 11 episódios)
- ✅ **Learning Rate**: 0.0001 (baixo para estabilidade)
- ✅ **Propósito**: Define direção estratégica (bull/bear/sideways)
- ✅ **Não reage a ruído**: Treina menos → generaliza melhor

---

## 🟢 MicroNet (Processor de Curto Prazo)

### Objetivo
Captura padrões de **curto prazo** (5 horas de contexto, 60 candles)

### Arquitetura

#### Entrada
- **Dimensão**: Variável (depende das features agregadas)
- **Features agregadas de 60 candles**:
  - Mean de todas features: N features
  - Std de todas features: N features
  - Último valor: N features
  - **Total**: 3N features (exemplo: se 20 features → 60 inputs)

#### Camadas

| Camada | Tipo | Entrada | Saída | Ativação | Dropout |
|--------|------|---------|-------|----------|---------|
| **Layer 1** | Linear | 3N (ex: 60) | 64 | ReLU | - |
| **Dropout 1** | Dropout | 64 | 64 | - | 20% |
| **Layer 2** | Linear | 64 | 32 | ReLU | - |

#### Saída
- **Dimensão**: 32 (micro features)
- **Significado**: Padrões táticos de curto prazo

### Parâmetros (exemplo com N=20 features)

```python
# Input: 60 features (20 * 3)

Layer 1: 60 → 64
  Weights: 60 × 64 = 3,840
  Biases:  64 = 64
  Subtotal: 3,904

Layer 2: 64 → 32
  Weights: 64 × 32 = 2,048
  Biases:  32 = 32
  Subtotal: 2,080

TOTAL MicroNet: 5,984 parâmetros
```

### Características
- ✅ **Updates**: 10x por ciclo (a cada episódio exceto quando MacroNet atualiza)
- ✅ **Learning Rate**: 0.0005 (alto para agilidade)
- ✅ **Propósito**: Define timing preciso de entrada/saída
- ✅ **Alta reatividade**: Treina muito → adapta-se rápido

---

## 🟡 Decision Head (Cabeça de Decisão)

### Objetivo
Combina contexto macro + micro + estado atual → decisão de ação

### Arquitetura

#### Entrada
- **Macro Embedding**: 128 dim (de MacroNet)
- **Micro Features**: 32 dim (de MicroNet)
- **Position**: 1 dim (-1.0 a +1.0, posição atual)
- **Cash Ratio**: 1 dim (0.0 a 1.0, cash/capital)
- **Total**: 128 + 32 + 2 = **162 dim**

#### Camadas

| Camada | Tipo | Entrada | Saída | Ativação | Dropout |
|--------|------|---------|-------|----------|---------|
| **Layer 1** | Linear | 162 | 128 | ReLU | - |
| **Dropout 1** | Dropout | 128 | 128 | - | 20% |
| **Layer 2** | Linear | 128 | 64 | ReLU | - |
| **Dropout 2** | Dropout | 64 | 64 | - | 20% |
| **Layer 3** | Linear | 64 | 3 | - | - |
| **Softmax** | Softmax | 3 | 3 | Softmax | - |

#### Saída
- **Dimensão**: 3
- **Significado**: Probabilidades de ações
  - `action[0]`: P(HOLD) - Manter posição
  - `action[1]`: P(BUY) - Comprar (long)
  - `action[2]`: P(SELL) - Vender (short)

### Parâmetros

```python
Layer 1: 162 → 128
  Weights: 162 × 128 = 20,736
  Biases:  128 = 128
  Subtotal: 20,864

Layer 2: 128 → 64
  Weights: 128 × 64 = 8,192
  Biases:  64 = 64
  Subtotal: 8,256

Layer 3: 64 → 3
  Weights: 64 × 3 = 192
  Biases:  3 = 3
  Subtotal: 195

TOTAL Decision Head: 29,315 parâmetros
```

### Características
- ✅ **Updates**: 10x por ciclo (junto com MicroNet)
- ✅ **Learning Rate**: 0.0005 (mesmo da MicroNet)
- ✅ **Propósito**: Combinar informações e tomar decisão final
- ✅ **Dropout**: 20% para regularização

---

## 📊 Resumo Total

### Contagem de Parâmetros (N=20 features)

| Componente | Parâmetros | % Total | Updates/Ciclo | LR |
|------------|------------|---------|---------------|-----|
| **MacroNet** | 48,512 | 57.8% | **1x** | 0.0001 |
| **MicroNet** | 5,984 | 7.1% | **10x** | 0.0005 |
| **Decision Head** | 29,315 | 35.0% | **10x** | 0.0005 |
| **TOTAL** | **83,811** | 100% | - | - |

### Workload por Ciclo (11 episódios)

```
Episódio 0:  [MacroNet ✓] + [MicroNet ✓] + [Decision ✓]  → 83,811 params
Episódio 1:                 [MicroNet ✓] + [Decision ✓]  → 35,299 params
Episódio 2:                 [MicroNet ✓] + [Decision ✓]  → 35,299 params
Episódio 3:                 [MicroNet ✓] + [Decision ✓]  → 35,299 params
Episódio 4:                 [MicroNet ✓] + [Decision ✓]  → 35,299 params
Episódio 5:                 [MicroNet ✓] + [Decision ✓]  → 35,299 params
Episódio 6:                 [MicroNet ✓] + [Decision ✓]  → 35,299 params
Episódio 7:                 [MicroNet ✓] + [Decision ✓]  → 35,299 params
Episódio 8:                 [MicroNet ✓] + [Decision ✓]  → 35,299 params
Episódio 9:                 [MicroNet ✓] + [Decision ✓]  → 35,299 params
Episódio 10:                [MicroNet ✓] + [Decision ✓]  → 35,299 params

Total por ciclo: 401,502 operações de parâmetros
Média por episódio: 36,500 params
```

### Comparação com Simétrico

| Abordagem | Macro Updates | Micro Updates | Params/Ciclo | Eficiência |
|-----------|---------------|---------------|--------------|------------|
| **Simétrico** | 11x | 11x | 921,921 | Baseline |
| **Assimétrico 1:2** | 4x | 8x | 529,156 | 1.74x faster |
| **Assimétrico 1:10** | 1x | 10x | 401,502 | **2.30x faster** |

---

## 🎯 Fluxo de Dados Completo

### Fase 1: Feature Extraction
```
Raw OHLCV Data (2024)
   ↓
Feature Builder (20+ indicators)
   ↓
Numeric Features Array (N × M)
   N = candles, M = features
```

### Fase 2: Window Aggregation
```
For each timestep t:

  Macro Window [t-492:t]:
    → Mean, Std, Last → (60 features)
  
  Micro Window [t-60:t]:
    → Mean, Std, Last → (60 features)
  
  State:
    → Position (1)
    → Cash Ratio (1)
```

### Fase 3: Forward Pass
```
Macro Features (60) → MacroNet → Macro Embedding (128)
                                         ↓
Micro Features (60) → MicroNet → Micro Features (32)
                                         ↓
                    [Concatenate with Position + Cash]
                                         ↓
                        Combined (162)
                                         ↓
                      Decision Head
                                         ↓
                  Action Probabilities (3)
                  [P(HOLD), P(BUY), P(SELL)]
```

### Fase 4: Action Selection
```
Action Probs → Categorical Distribution → Sample Action
                                              ↓
                                        Execute in Env
                                              ↓
                                          Get Reward
```

### Fase 5: Policy Gradient Update
```
Collect Trajectory: [(s₀,a₀,r₀), (s₁,a₁,r₁), ..., (sₜ,aₜ,rₜ)]
                            ↓
        Calculate Discounted Returns: G = Σ γⁱ·rᵢ
                            ↓
              Normalize Returns: Ĝ = (G - μ) / σ
                            ↓
          Policy Loss: L = -Σ log π(aᵢ|sᵢ) · Ĝᵢ
                            ↓
                  Backpropagation
                            ↓
        Selective Update (cycle position):
        
        Episode % 11 == 0:
          → Update MacroNet (LR=0.0001)
          → Update MicroNet (LR=0.0005)
          → Update Decision (LR=0.0005)
        
        Episode % 11 != 0:
          → Update MicroNet (LR=0.0005)
          → Update Decision (LR=0.0005)
          → Freeze MacroNet
```

---

## 🧮 Cálculo de Parâmetros (Fórmula Geral)

Para entender como calculei:

### Linear Layer
```
Parâmetros = (input_dim × output_dim) + output_dim
           = weights + biases

Exemplo: Linear(256, 128)
  Weights: 256 × 128 = 32,768
  Biases:  128
  Total:   32,896
```

### Dropout Layer
```
Parâmetros = 0 (apenas máscara durante treinamento)
```

### ReLU Activation
```
Parâmetros = 0 (função pura)
```

### Softmax
```
Parâmetros = 0 (função pura)
```

---

## 🔬 Análise de Complexidade

### Computacional (FLOPs por Forward Pass)

| Componente | FLOPs | % Total |
|------------|-------|---------|
| MacroNet Layer 1 | 60 × 256 × 2 = 30,720 | 36.5% |
| MacroNet Layer 2 | 256 × 128 × 2 = 65,536 | 77.8% (acumulado) |
| MicroNet Layer 1 | 60 × 64 × 2 = 7,680 | 87.0% |
| MicroNet Layer 2 | 64 × 32 × 2 = 4,096 | 91.8% |
| Decision Layer 1 | 162 × 128 × 2 = 41,472 | 100% |
| Decision Layer 2 | 128 × 64 × 2 = 16,384 | - |
| Decision Layer 3 | 64 × 3 × 2 = 384 | - |
| **TOTAL** | **166,272 FLOPs** | - |

### Memória (Tensors)

| Tensor | Shape | Size (float32) |
|--------|-------|----------------|
| Macro Input | (batch, 60) | 240 bytes |
| Macro Hidden | (batch, 256) | 1,024 bytes |
| Macro Output | (batch, 128) | 512 bytes |
| Micro Input | (batch, 60) | 240 bytes |
| Micro Hidden | (batch, 64) | 256 bytes |
| Micro Output | (batch, 32) | 128 bytes |
| Decision Input | (batch, 162) | 648 bytes |
| Decision Hidden 1 | (batch, 128) | 512 bytes |
| Decision Hidden 2 | (batch, 64) | 256 bytes |
| Action Probs | (batch, 3) | 12 bytes |
| **TOTAL (batch=1)** | - | **3,828 bytes ≈ 3.7 KB** |

---

## 💡 Design Rationale

### Por que 128 dim para Macro?
- ✅ Espaço suficiente para representar tendências complexas
- ✅ Não muito grande (evita overfitting)
- ✅ Potência de 2 (eficiente em GPU)

### Por que 32 dim para Micro?
- ✅ Menor que Macro (contexto mais simples)
- ✅ Suficiente para padrões de curto prazo
- ✅ Mais leve → updates 10x mais rápidos

### Por que 162 → 128 → 64 → 3?
- ✅ Redução gradual (smooth)
- ✅ 128 → 64: redução de 2x (padrão)
- ✅ 64 → 3: bottleneck final força compressão

### Por que Dropout 20%?
- ✅ Não muito alto (não perde informação)
- ✅ Não muito baixo (ainda regulariza)
- ✅ Padrão da literatura (0.2-0.5)

---

## 📈 Vantagens da Arquitetura Assimétrica

### 1. Eficiência Computacional
```
Simétrico:    921,921 params/ciclo
Assimétrico:  401,502 params/ciclo
Economia:     56.5% menos operações!
```

### 2. Separação de Concerns
```
MacroNet:  "Devemos comprar ou vender?" (estratégia)
           ↓ (atualiza 1x, estável)
MicroNet:  "Exatamente quando entrar/sair?" (tática)
           ↓ (atualiza 10x, ágil)
Decision:  "Qual ação tomar agora?" (execução)
```

### 3. Estabilidade + Agilidade
```
Macro LR = 0.0001 → Mudanças lentas e estáveis
Micro LR = 0.0005 → Adaptação rápida

Resultado: Estratégia sólida + Tática flexível
```

---

## 🎛️ Hyperparameters Summary

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| **Macro Window** | 492 candles (41h) | Captura tendências diárias |
| **Micro Window** | 60 candles (5h) | Captura padrões intraday |
| **Macro Embedding** | 128 dim | Balanço capacidade/overfitting |
| **Micro Features** | 32 dim | Leve e suficiente |
| **Macro LR** | 0.0001 | Estabilidade |
| **Micro LR** | 0.0005 | Agilidade |
| **Gamma** | 0.99 | Valoriza recompensas futuras |
| **Commission** | 0.1% | Realista (Binance) |
| **Dropout** | 20% | Regularização padrão |
| **Update Ratio** | 1:10 | Máxima assimetria |

---

**Resumo Final:**
- **83,811 parâmetros totais**
- **MacroNet**: 48,512 params (57.8%), atualiza 1x/ciclo
- **MicroNet**: 5,984 params (7.1%), atualiza 10x/ciclo
- **Decision**: 29,315 params (35.0%), atualiza 10x/ciclo
- **Eficiência**: 2.30x mais rápida que simétrico
- **Ratio**: 1:10 (extremamente assimétrico)

---

**Data:** 13 de novembro de 2025  
**Versão:** 4.0 - Treinamento Assimétrico (1:10)
