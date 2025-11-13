# 🚀 Como Executar o Sistema - Guia Prático

## 📋 Índice
1. [Verificação Inicial](#verificação-inicial)
2. [Testes Rápidos](#testes-rápidos)
3. [Análise Interativa](#análise-interativa)
4. [Demo Completo](#demo-completo)
5. [Uso em Produção](#uso-em-produção)

---

## ✅ Verificação Inicial

### 1. Verificar Python e Dependências

```bash
# Verificar Python 3.12+
python --version

# Verificar dependências principais
python -c "import torch, polars, pandas; print('✓ Core OK')"
```

**Resultado esperado:**
```
Python 3.12.10
Core packages OK
Torch 2.2.0+cpu, Polars 0.20.3, Pandas 2.1.3
```

### 2. Testar Import dos Módulos

```bash
python -c "
import sys
sys.path.insert(0, '.')
from src.pipeline import TradingPipeline
from src.config import config
print('✓ Importações OK')
print(f'Device: {config.device}')
"
```

---

## 🧪 Testes Rápidos (Recomendado para começar)

### Executar Todos os Testes

```bash
python quick_tests.py
```

**O que testa:**
- ✅ Configurações
- ✅ Ingestão de dados (Binance API)
- ✅ Feature engineering
- ✅ MacroNet training
- ✅ MicroNet training
- ✅ Backtesting
- ✅ Pipeline completo

**Tempo esperado:** 3-5 minutos

**Resultado esperado:**
```
🧪 QUICK TESTS - Validar Componentes
═══════════════════════════════════════

TEST: 1. Verificar Configurações
✓ Device: cpu
✓ MacroNet embedding_dim: 128
✓ MicroNet short_lookback: 60
✅ Config: PASSED

TEST: 2. Ingestão de Dados (Binance)
⏳ Coletando 2 dias de BTCUSDT...
✓ Total de candles: 576
✓ Features: 21
✅ Data Ingestion: PASSED

... (mais testes)

RESUMO DOS TESTES
═══════════════════════════════════════
Config                         ✅ PASS
Data Ingestion                 ✅ PASS
Feature Engineering            ✅ PASS
MacroNet                       ✅ PASS
MicroNet                       ✅ PASS
Backtest                       ✅ PASS
Pipeline                       ✅ PASS

Total: 7/7 ✅
🎉 Todos os testes passaram!
```

---

## 📊 Análise Interativa

### Menu Interativo com Dados Reais

```bash
python interactive_analysis.py
```

**Menu disponível:**
```
📋 Escolha uma opção:
  1. Analisar Dados Coletados
  2. Analisar Features e Correlações
  3. Ver Arquitetura dos Modelos
  4. Executar Demo Completo (longo)
  5. Sair
```

### Exemplos de Uso

#### Opção 1: Analisar Dados Coletados
```bash
# Escolher opção 1
# Resultado:
#   📊 Resumo dos Datasets
#   📈 Dados Brutos
#   📊 Estatísticas de Preço
#   🔧 Features Engenheiradas
#   🔗 Matriz de Correlação
```

#### Opção 2: Analisar Features
```bash
# Escolher opção 2
# Resultado:
#   📊 Distribuição das Features
#   🔗 Top 10 Correlações
```

#### Opção 3: Ver Arquitetura
```bash
# Escolher opção 3
# Resultado:
#   🧠 MacroNet Architecture
#   🎯 MicroNet Architecture
#   📊 Feature Engineering Pipeline
```

---

## 🎬 Demo Completo

### Executar Workflow Completo com Explicações

```bash
python demo_complete_workflow.py
```

**O que inclui:**

1. **Passo 1:** Coleta de Dados
   - Baixa do Binance
   - Cálculo de features
   - Estatísticas

2. **Passo 2:** Feature Engineering
   - 13 features técnicas
   - Correlações
   - Normalização

3. **Passo 3:** Treinamento MacroNet
   - Dados de entrada
   - Processo de treinamento
   - Geração de embedding

4. **Passo 4:** Geração de Embedding
   - Compressão de 5 dias
   - Dimensionalidade (128)
   - Cache

5. **Passo 5:** Treinamento MicroNet
   - Combinação de contextos
   - Treinamento histórico

6. **Passo 6:** Geração de Sinal
   - Score de -1 a +1
   - Interpretação

7. **Passo 7:** Backtesting
   - Simulação de 30 dias
   - Métricas de performance
   - Análise

**Tempo esperado:** 5-10 minutos

**Resultado esperado:**
```
══════════════════════════════════════════════════════════════
  PASSO 1: COLETA DE DADOS DO BINANCE
══════════════════════════════════════════════════════════════

📥 Coletando últimos 5 dias de BTCUSDT (5m candles)...
✓ Total de candles: 1440
✓ Features calculadas: 21
✓ Long window (últimos 5d): 1440 candles
✓ Short window (últimas 5h): 60 candles

📊 Estatísticas:
  Close (últimas 5h): min=43500.00, max=43700.00
  Volume médio: 850000
  Retorno médio: 0.000015

... (próximos passos)

✅ DEMO COMPLETO FINALIZADO COM SUCESSO!

📊 Resumo final:
  • Retorno total: +2.35%
  • Sharpe ratio: 0.85
  • Drawdown máximo: -8.30%
  • Taxa de acerto: 53.20%
  • Total de trades: 156
```

---

## 💼 Uso em Produção

### Exemplo 1: Previsão Simples

```python
from src.pipeline import TradingPipeline

# Inicializar
pipeline = TradingPipeline()

# Gerar sinal para hoje
signal = pipeline.predict_signal("BTCUSDT")

if signal > 0.5:
    print("🟢 COMPRA")
elif signal < -0.5:
    print("🔴 VENDA")
else:
    print("⚪ NEUTRO")
```

### Exemplo 2: Backtesting de 30 dias

```python
from src.pipeline import TradingPipeline

pipeline = TradingPipeline()

# Treinar macronet
pipeline.train_macronet("BTCUSDT", days_back=30)

# Backtest
results = pipeline.backtest_strategy("BTCUSDT", days_back=30)

print(f"Retorno: {results['total_return']*100:.2f}%")
print(f"Sharpe: {results['sharpe']:.2f}")
print(f"Drawdown: {results['max_drawdown']*100:.2f}%")
```

### Exemplo 3: Múltiplos Símbolos

```python
from src.pipeline import TradingPipeline

pipeline = TradingPipeline()
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

for symbol in symbols:
    signal = pipeline.predict_signal(symbol)
    print(f"{symbol}: {signal:.4f}")
```

---

## 🔧 Configuração

### Arquivos de Configuração

**`src/config.py`**: Define todos os hiperparâmetros

```python
# Editar para ajustar:
config.macro.embedding_dim = 128    # Dimensão do embedding
config.micro.short_lookback = 60    # Candles de curto prazo
config.backtest.commission = 0.001  # 0.1%
config.backtest.slippage = 0.0005   # 0.05%
```

**`.env`**: Variáveis de ambiente

```
# Configurar Binance API (opcional)
BINANCE_API_KEY=sua_chave_aqui
BINANCE_API_SECRET=seu_secret_aqui

# Dados cacheados em:
DATA_DIR=data/
MODELS_DIR=models/
```

---

## 📈 Interpretando Resultados

### Métricas de Performance

| Métrica | Bom | Aceitável | Ruim |
|---------|-----|-----------|------|
| **Sharpe Ratio** | > 1.0 | 0.5 - 1.0 | < 0.5 |
| **Sortino Ratio** | > 1.5 | 0.8 - 1.5 | < 0.8 |
| **Max Drawdown** | < 10% | 10% - 20% | > 20% |
| **Win Rate** | > 55% | 50% - 55% | < 50% |
| **Total Return** | > 20%/ano | 5% - 20% | < 5% |

### Interpretação de Sinais

```
Signal Range    Interpretação           Ação
═══════════════════════════════════════════════════════
+1.0 a +0.7    🟢 Compra Muito Forte   → Comprar
+0.7 a +0.3    🟢 Compra Moderada      → Comprar
+0.3 a -0.3    ⚪ Neutro               → Manter/Hold
-0.3 a -0.7    🔴 Venda Moderada       → Vender
-0.7 a -1.0    🔴 Venda Muito Forte    → Vender
```

---

## ⚠️ Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'torch'"

**Solução:**
```bash
pip install -r requirements.txt
```

### Problema: "FileNotFoundError: data/..."

**Solução:**
```bash
# Criar diretórios
mkdir -p data/timeframe=5m/symbol=BTCUSDT/
mkdir -p models/
```

### Problema: Conexão Binance recusada

**Solução:**
```bash
# Verificar conexão
python -c "import requests; print(requests.get('https://api.binance.com/api/v3/time').json())"

# Usar dados cacheados
python -c "from src.ingest.binance import BinanceIngestor; BinanceIngestor.load_from_parquet('BTCUSDT', '5m')"
```

### Problema: Tempo de execução muito longo

**Solução:**
```python
# Reduzir período
pipeline.fetch_and_prepare_data("BTCUSDT", days_back=2)  # Instead of 30

# Reduzir epochs
pipeline.macronet.train(X, epochs=2)  # Instead of 20
```

---

## 📚 Próximos Passos

1. **Fase 1 (Atual):** Validar PoC
   - ✅ Dados coletando
   - ✅ Modelos treinando
   - ✅ Sinais gerando
   - ✅ Backtest rodando

2. **Fase 2:** Otimização
   - [ ] Drift detection
   - [ ] Auto-retraining
   - [ ] Multi-símbolo
   - [ ] Risk management

3. **Fase 3:** CPPN/HyperNEAT
   - [ ] Evolução de arquitetura
   - [ ] Neuroevolution
   - [ ] Multiobjetiva

4. **Fase 4:** Produção
   - [ ] API REST
   - [ ] Dashboard
   - [ ] Docker
   - [ ] CI/CD

---

## 📞 Suporte

Para debug, adicione logs estruturados:

```python
from src.logger import get_logger

logger = get_logger(__name__)
logger.info("meu_evento", valor=123, outro="teste")
```

Logs estão em formato JSON estruturado para análise fácil.

---

**Última atualização:** Novembro 2025
