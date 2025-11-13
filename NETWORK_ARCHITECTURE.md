# Arquitetura das Redes Neurais - cnncripto

## Visão Geral do Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE DE TRADING                          │
│                                                                     │
│  Dados Binance → Feature Engineering → MacroNet → MicroNet → Signal │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. MacroNet - Encoder de Padrões de Longo Prazo

### Entrada
- **Shape**: (batch_size, seq_len=492, num_features=10)
- **Descrição**: 2 dias de dados de 5 minutos (492 candles) com 10 features numéricas

### Arquitetura

```
INPUT
  │
  └──→ (batch=1, seq_len=492, features=10)
       │
       ├──────────────────────────────────────────────────────────────┐
       │                    TEMPORAL ENCODER                          │
       │                                                              │
       │  Transpose: (batch, features, seq_len)                      │
       │     ↓                                                        │
       │  ┌─────────────────────────────────────────────────┐        │
       │  │  Conv1d Layer 1 (Dilated, dilation=1)         │        │
       │  │  in: 10 → out: 256                            │        │
       │  │  kernel_size=3, padding=adaptivo              │        │
       │  └──→ BatchNorm1d → ReLU → Dropout(0.1)         │        │
       │     ↓                                             │        │
       │  ┌─────────────────────────────────────────────────┐        │
       │  │  Conv1d Layer 2 (Dilated, dilation=2)         │        │
       │  │  in: 256 → out: 256                           │        │
       │  └──→ BatchNorm1d → ReLU → Dropout(0.1)         │        │
       │     ↓                                             │        │
       │  ┌─────────────────────────────────────────────────┐        │
       │  │  Conv1d Layer 3 (Dilated, dilation=4)         │        │
       │  │  in: 256 → out: 256                           │        │
       │  └──→ BatchNorm1d → ReLU → Dropout(0.1)         │        │
       │     ↓                                             │        │
       │  Transpose back: (batch, seq_len, 256)           │        │
       └──────────────────────────────────────────────────────────────┘
       │
       ├──────────────────────────────────────────────────────────────┐
       │              ATTENTION-BASED POOLING                         │
       │                                                              │
       │  (batch, seq_len=492, hidden=256)                           │
       │     ↓                                                        │
       │  ┌─────────────────────────────┐                            │
       │  │ Attention Weights Generator │  Linear + ReLU + Linear    │
       │  │ 256 → 64 → 1               │  + Softmax                  │
       │  └─────────────────────────────┘                            │
       │     ↓                                                        │
       │  (batch, seq_len=492, 1) [pesos de atenção]                │
       │     ↓                                                        │
       │  Weighted Sum: Σ(seq * weights) → (batch, hidden=256)      │
       └──────────────────────────────────────────────────────────────┘
       │
       ├──────────────────────────────────────────────────────────────┐
       │                  EMBEDDING LAYER                             │
       │                                                              │
       │  (batch, hidden=256)                                        │
       │     ↓                                                        │
       │  Linear(256 → 128) → ReLU → Dropout(0.1) → Linear(128→128) │
       │     ↓                                                        │
       │  (batch, embedding_dim=128)                                 │
       └──────────────────────────────────────────────────────────────┘
       │
       └──→ EMBEDDING OUTPUT
            │
            ├─→ Shape: (batch=1, embedding_dim=128)
            │
            └─→ Armazenado em: pipeline.current_macro_embedding
```

### Saída
- **Shape**: (batch_size, embedding_dim=128)
- **Descrição**: Vetor de embedding comprimido capturando padrões macro (longo prazo)

### Treinamento
- **Loss**: MSELoss (Autoencoder - reconstrução)
- **Epochs**: 50 (padrão)
- **Learning Rate**: 0.001
- **Otimizador**: Adam

---

## 2. MicroNet - Decision Head de Curto Prazo

### Entrada
- **Entrada 1** (Short-term features): (batch_size, seq_len=60, num_features=10)
  - Últimos 60 candles (5 horas em timeframe 5m)
- **Entrada 2** (Macro embedding): (batch_size, embedding_dim=128)
  - Embedding do MacroNet

### Arquitetura

```
INPUT 1: Short-Term Features          INPUT 2: Macro Embedding
  │                                       │
  └──→ (batch, 60, 10)                    └──→ (batch, 128)
       │                                       │
       ├────────────────────────────┐          │
       │  SHORT-TERM PROCESSOR      │          │
       │                            │          │
       │  Linear(60*10 → 64)        │          │
       │  (after flatten: 600 → 64) │          │
       │     ↓                      │          │
       │  ReLU → Dropout(0.1)       │          │
       │     ↓                      │          │
       │  Linear(64 → 32)           │          │
       │     ↓                      │          │
       │  ReLU                      │          │
       │     ↓                      │          │
       │  (batch, short_features=32)│          │
       └────────────────────────────┘          │
       │                                       │
       └──────────────────┬────────────────────┘
                          │
              ┌───────────┴───────────┐
              │   CONCATENATION       │
              │  (batch, 32+128=160)  │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────────────────┐
              │  DECISION HEAD                    │
              │                                  │
              │  Linear(160 → 64)                │
              │     ↓                            │
              │  ReLU → Dropout(0.1)             │
              │     ↓                            │
              │  Linear(64 → 32)                 │
              │     ↓                            │
              │  ReLU                            │
              │     ↓                            │
              │  Linear(32 → 1)                  │
              │     ↓                            │
              │  Tanh  (output ∈ [-1, 1])        │
              │                                  │
              └───────────┬───────────────────────┘
                          │
                          └──→ SIGNAL OUTPUT
                               │
                               ├─→ Shape: (batch=1,)
                               │
                               ├─→ Range: [-1.0, 1.0]
                               │   • -1.0 = Strong SELL signal
                               │    0.0 = Neutral
                               │   +1.0 = Strong BUY signal
                               │
                               └─→ Usado para trading decisions
```

### Saída
- **Shape**: (batch_size,) — escalar por amostra
- **Range**: [-1.0, 1.0]
- **Interpretação**:
  - **-1.0**: Sinal forte de venda/short
  - **0.0**: Neutro
  - **+1.0**: Sinal forte de compra/long

### Treinamento
- **Loss**: MSELoss
- **Epochs**: 100 (padrão)
- **Learning Rate**: 0.001
- **Otimizador**: Adam

---

## 3. Fluxo Completo de Dados

```
┌────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLETO                                   │
└────────────────────────────────────────────────────────────────────────┘

STEP 1: COLETA DE DADOS
┌─────────────┐
│  Binance    │ → fetch_and_prepare_data("BTCUSDT", days_back=2)
│  REST API   │
└─────────────┘
       │
       └──→ long_data:  (492, 21)   [2 dias de OHLCV + raw features]
       │
       └──→ short_data: (60, 21)    [últimas 5 horas]

STEP 2: ENGENHARIA DE FEATURES
┌──────────────────┐
│ FeatureBuilder   │ → build_features() + normalize_features()
│ (src/features)   │
└──────────────────┘
       │
       └──→ long_data:  (492, 21)  → Cálculo de indicadores técnicos
       │    [open, high, low, close, volume, quote_volume, 
       │     log_return, volatility_12/24/48, volume_zscore, ...]
       │
       └──→ short_data: (60, 21)

STEP 3: EXTRAÇÃO DE FEATURES NUMÉRICAS
┌───────────────────────────┐
│ extract_feature_arrays()  │ → Filtra apenas colunas numéricas
│ (remove: timestamp, date, │    remove: '5m' (timeframe string)
│  open, high, low, close,  │
│  volume, quote_volume)    │
└───────────────────────────┘
       │
       └──→ X_long:  (492, 10)   [10 features numéricas]
       │
       └──→ X_short: (60, 10)

STEP 4: TREINAMENTO MACRONET
┌──────────────────┐
│ MacroNet.train() │ ← X: (1, 492, 10)  [batch de 1, seq_len=492]
│                  │
│ Objetivo:        │ Aprender padrões de longo prazo
│ Reconstruir X    │ Compressão: 492*10 → 128 dimensões
└──────────────────┘
       │
       └──→ Embedding: (1, 128)
            Armazenado em: self.current_macro_embedding

STEP 5: TREINAMENTO MICRONET
┌──────────────────┐
│MicroNet.train()  │ ← X_short: (10, 60, 10)  [10 amostras de treino]
│                  │   X_macro: (10, 128)
│ Objetivo:        │   y: (10, 1)             [labels de mercado]
│ Aprender signal  │
│ combinando short │ Converte 3D → 2D (flatten timeframes)
│ + long term      │ Combina short features + macro embedding
└──────────────────┘
       │
       └──→ Model weights salvos

STEP 6: PREDIÇÃO DE SINAL
┌────────────────────────────┐
│ predict_signal(symbol)     │
│                            │
│ 1. Fetch novos dados       │
│    ↓                       │
│ 2. Extract features        │
│    ↓                       │
│ 3. Gerar novo embedding    │
│    (ou usar anterior)      │
│    ↓                       │
│ 4. X_short: últimos 60     │ → Shape: (1, 60, 10)
│    candles extraídos       │   (com padding se necessário)
│    ↓                       │
│ 5. MicroNet.predict()      │ Combina com macro_embedding
│    ↓                       │
│ 6. Output signal           │ Range: [-1.0, 1.0]
└────────────────────────────┘
       │
       └──→ TRADING DECISION
            (-1 = SELL, 0 = HOLD, +1 = BUY)
```

---

## 4. Mapeamento de Dimensões

```
┌─────────────────────────────────────────────────────────────┐
│                    SHAPE EVOLUTION                          │
└─────────────────────────────────────────────────────────────┘

MacroNet Training Flow:
  Raw Data        → (1, 492, 21)    [Binance OHLCV + temps]
  Extract Features→ (1, 492, 10)    [Apenas numéricas]
  Transpose       → (1, 10, 492)    [Para Conv1d]
  Conv Layers     → (1, 256, 492)   [After 3x Conv1d]
  Transpose Back  → (1, 492, 256)   [Volta a seq_len]
  Attention Pool  → (1, 256)        [Redução: seq_len → 1]
  Embedding Layer → (1, 128)        [Compressão final]

MicroNet Training Flow:
  X_short (raw)   → (10, 60, 10)    [10 samples × 60 candles × 10 features]
  Flatten (in forward) → (10, 600)  [60*10 dimensões]
  Short Processor → (10, 32)        [Redução via Linear layers]
  X_macro (input) → (10, 128)       [Embedding pré-calculado]
  Concatenate     → (10, 160)       [32 + 128]
  Decision Head   → (10, 1)         [Output: score para cada sample]

MicroNet Prediction Flow:
  X_short (real)  → (1, 60, 10)     [1 sample × 60 candles × 10 features]
  Flatten (in forward) → (1, 600)
  Short Processor → (1, 32)
  X_macro (input) → (1, 128)        [Embedding atual]
  Concatenate     → (1, 160)
  Decision Head   → (1, 1)
  Final Output    → (1,)            [Escalar: signal score]
```

---

## 5. Parâmetros das Redes

```
┌────────────────────────────────────────────────────────────┐
│              MacroNet Configuration                        │
├────────────────────────────────────────────────────────────┤
│  embedding_dim         128      Dimensão de saída         │
│  encoder_layers        3        Número de Conv1d layers   │
│  hidden_dim            256      Dimensão dos conv layers  │
│  dropout               0.1      Taxa de dropout           │
│  learning_rate         0.001    Learning rate do Adam     │
│  batch_size            32       Batch para treinamento    │
│  epochs                50       Épocas de treino          │
│  lookback_days         5        Dias de dados históricos  │
│  model_path            ./models/macronet                  │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│              MicroNet Configuration                        │
├────────────────────────────────────────────────────────────┤
│  hidden_dim            64       Dimensão dos layers       │
│  dropout               0.1      Taxa de dropout           │
│  learning_rate         0.001    Learning rate do Adam     │
│  batch_size            32       Batch para treinamento    │
│  epochs                100      Épocas de treino          │
│  lookback_candles      60       Candles para predict      │
│  decision_threshold    0.5      Threshold para signals    │
│  model_path            ./models/micronet                  │
└────────────────────────────────────────────────────────────┘
```

---

## 6. Conexão entre MacroNet e MicroNet

```
┌─────────────────────────────────────────────────────────────┐
│          INTEGRAÇÃO: Como MacroNet alimenta MicroNet       │
└─────────────────────────────────────────────────────────────┘

Timeline:
├─ T-48h: Coleta de dados históricos de 2 dias (492 candles)
│
├─ T-48h: MacroNet.train()
│         Processa 492 candles → embedding de 128 dimensões
│         Captura padrões de longo prazo (macro)
│         Salva modelo
│
├─ T-24h: Treina MicroNet
│         Recebe:
│         • X_short: janelas de 60 candles (5h) com 10 features
│         • X_macro: embedding macro (128-dim)
│         • y: labels/targets de trading
│         
│         Combina informações:
│         short_feats(32d) + macro_embedding(128d) → decision(signal)
│
└─ T-0: Predição em tempo real
        1. Fetch dados recentes
        2. Gera novo embedding macro (atualizado)
        3. X_short = últimos 60 candles
        4. MicroNet.predict(X_short, macro_embedding)
        5. Output: signal [-1, 1]
```

---

## 7. Exemplo de Execução

```
# Terminal
$ python quick_tests.py

STEP 1: Fetch Data
  → Binance API: 2 dias BTCUSDT 5m
  → 492 candles, 21 colunas

STEP 2: Extract Features
  → Remove não-numéricas (timestamp, date, '5m')
  → Resultado: 10 features numéricas

STEP 3: Train MacroNet
  → Input:  (1, 492, 10)
  → Output: (1, 128) embedding

STEP 4: Generate Embedding
  → Usar embedding macro para próximo passo

STEP 5: Train MicroNet
  → Short input: (10, 60, 10)
  → Macro input: (10, 128)
  → Aprende relação entre short+macro → signal

STEP 6: Predict Signal
  → Input: (1, 60, 10) short + (1, 128) macro
  → Output: 0.95 (STRONG BUY signal)

RESULT: ✅ PASS - Pipeline completo funcionando!
```

---

## Resumo Visual Final

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│                    SISTEMA DE TRADING                            │
│                                                                  │
│  ┌─────────────────────┐          ┌──────────────────────┐      │
│  │  DADOS BINANCE      │          │   MACRO EMBEDDING    │      │
│  │  (492 candles)      │          │   (128 dimensões)    │      │
│  └──────────┬──────────┘          └──────────┬───────────┘      │
│             │                                 │                  │
│             │              MACRONET           │                  │
│             │          ┌─────────────┐        │                  │
│             └─────────→│  Encoder    │────────┘                  │
│                        │  Conv1d x3  │                           │
│                        │  Attention  │                           │
│                        │  Embedding  │                           │
│                        └─────────────┘                           │
│                                                                  │
│                        MICRONET                                  │
│            ┌───────────────────────────────┐                     │
│            │  Decision Head                │                     │
│            │  (Short features → decision)  │                     │
│            │  + Macro embedding context    │                     │
│            └───────────────┬────────────────┘                     │
│                            │                                     │
│                    ┌───────▼────────┐                            │
│                    │   SIGNAL       │                            │
│                    │  [-1.0, +1.0]  │                            │
│                    │                │                            │
│                    │ -1.0 → SELL    │                            │
│                    │  0.0 → HOLD    │                            │
│                    │ +1.0 → BUY     │                            │
│                    └────────────────┘                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

**Documentação criada em**: 13 de novembro de 2025
**Versão do sistema**: cnncripto com MacroNet + MicroNet
**Status**: ✅ Todos os testes passando (7/7)
