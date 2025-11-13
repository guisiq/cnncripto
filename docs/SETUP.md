# Guia de Instalação e Execução

## Pré-requisitos
- Python 3.9+
- pip ou conda
- Conta Binance (opcional, para API real)

## Instalação

### 1. Clone ou navegue para o repositório
```bash
cd cppncripto
```

### 2. Criar ambiente virtual (recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente
```bash
# Copiar arquivo de exemplo
cp .env.example .env

# Editar .env com suas credenciais (opcional)
# BINANCE_API_KEY=your_key
# BINANCE_API_SECRET=your_secret
```

## Uso Rápido

### Execução do Exemplo Básico
```bash
python examples/basic_example.py
```

Este script demonstra:
1. Ingestão de dados da Binance
2. Preparação de features
3. Treinamento da MacroNet
4. Geração de embeddings
5. Geração de sinais com MicroNet

### Teste de Módulos
```bash
pytest tests/test_core.py -v
```

## Estrutura de Diretórios

```
cppncripto/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configurações globais
│   ├── logger.py              # Sistema de logging
│   ├── pipeline.py            # Pipeline completo
│   ├── ingest/
│   │   └── binance.py         # Ingestão de dados Binance
│   ├── features/
│   │   └── builder.py         # Engenharia de features
│   ├── macronet/
│   │   └── model.py           # Modelo MacroNet (Encoder)
│   ├── micronet/
│   │   └── model.py           # Modelo MicroNet (Decision Head)
│   └── evaluation/
│       └── backtest.py        # Backtesting e métricas
├── examples/
│   └── basic_example.py       # Exemplo básico
├── tests/
│   └── test_core.py           # Testes unitários
├── data/                      # Dados (Parquet)
├── models/                    # Modelos treinados (PT)
├── embeddings/                # Cache de embeddings
├── backtests/                 # Relatórios backtest
├── configs/                   # Configurações YAML
├── requirements.txt           # Dependências
├── .env.example               # Exemplo de variáveis
├── .gitignore
└── README.md
```

## Workflow Típico

### 1. Preparar Dados
```python
from src.pipeline import TradingPipeline

pipeline = TradingPipeline()

# Baixar e preparar dados (últimos 10 dias)
long_data, short_data, full_df = pipeline.fetch_and_prepare_data(
    "BTCUSDT",
    days_back=10,
    lookback_days=5
)
```

### 2. Treinar MacroNet
```python
# Treina rede grande uma vez por dia/semana
pipeline.train_macronet("BTCUSDT", days_back=30)
```

### 3. Gerar Embedding Macro (1x por dia)
```python
macro_embedding = pipeline.generate_macro_embedding("BTCUSDT")
```

### 4. Treinar MicroNet
```python
# Treina rede pequena com dados históricos
pipeline.train_micronet("BTCUSDT", days_back=30)
```

### 5. Gerar Sinais (a cada candle 5m)
```python
# Intradiário: gera score [-1, 1] a cada 5 minutos
signal_score = pipeline.predict_signal("BTCUSDT")

if signal_score > 0.5:
    print("COMPRAR")
elif signal_score < -0.5:
    print("VENDER")
else:
    print("NEUTRO")
```

### 6. Executar Backtest
```python
results = pipeline.backtest_strategy(
    "BTCUSDT",
    days_back=30,
    signal_threshold=0.5
)

print(f"Retorno total: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## Configurações Personalizadas

Editar `src/config.py` ou criar arquivo YAML:

```yaml
# configs/custom_config.yaml
data:
  symbols:
    - BTCUSDT
    - ETHUSDT
  timeframe: "5m"
  max_candles_per_request: 1000

macronet:
  embedding_dim: 128
  hidden_dim: 256
  epochs: 50
  lookback_days: 5

micronet:
  hidden_dim: 64
  epochs: 100
  lookback_candles: 60
  decision_threshold: 0.5

backtest:
  initial_cash: 10000.0
  commission: 0.001
```

Carregar com:
```python
from src.config import Config
config = Config.from_yaml("configs/custom_config.yaml")
```

## Troubleshooting

### "No module named 'binance'"
```bash
pip install python-binance
```

### "CUDA not available"
Certifique-se de ter PyTorch com suporte CUDA se tiver GPU. Senão, o código rodará em CPU.

### "Empty cache" ao iniciar
A primeira execução baixará dados da Binance. Pode levar alguns minutos conforme rate limits.

## Logging

Logs estruturados em JSON são salvos em:
- `cppncripto.log` (ou definir via `LOG_LEVEL`)

Ver logs:
```python
from src.logger import get_logger
logger = get_logger(__name__)
logger.info("mensagem", key="value")
```

## Próximos Passos

1. Ajustar `decision_threshold` conforme estatísticas de sinais
2. Monitorar drift: se Sharpe cair >20%, re-treinar
3. Testar multi-símbolo: paralelizar ingestão
4. Implementar CPPN/HyperNEAT para evolução de MacroNet
5. Integrar com API live (FastAPI)

## Suporte

Para problemas:
1. Verificar logs em `cppncripto.log`
2. Executar `pytest tests/ -v` para diagnosticar
3. Confirmar variáveis de ambiente em `.env`

---
**Status**: Versão 0.1 (PoC)  
**Data**: Novembro 2025
