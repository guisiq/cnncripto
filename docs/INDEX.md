# 🗂️ INDEX - Guia de Navegação do Projeto

## 🎯 Comece Aqui

1. **Primeira execução?**
   - Leia: `RESUMO_IMPLEMENTACAO.md`
   - Execute: `python run_quick_start.py`

2. **Tem GPU/CPU e quer otimizar?**
   - Leia: `GPU_CPU_SUPPORT.md`
   - Execute: `python check_device.py`

3. **Quer entender como instalar?**
   - Leia: `SETUP.md`
   - Leia: `COMO_EXECUTAR.md`

---

## 📂 Estrutura de Arquivos

### 📚 Documentação

| Arquivo | Propósito | Quando Ler |
|---------|-----------|-----------|
| **RESUMO_IMPLEMENTACAO.md** | Visão geral do que foi feito | Primeiro |
| **COMO_EXECUTAR.md** | Guia prático de uso | Segundo |
| **GPU_CPU_SUPPORT.md** | Suporte de hardware (CPU/GPU/MPS) | Se tem GPU |
| **README.md** | Visão geral técnica | Para entender arquitetura |
| **SETUP.md** | Instalação e ambiente | Se tiver problemas |
| **ROADMAP.md** | Plano futuro (5 fases) | Para ver próximos passos |

### 🚀 Scripts Executáveis

| Script | Tempo | Propósito |
|--------|-------|-----------|
| **run_quick_start.py** | 5-10 min | COMEÇAR AQUI - Demo completo |
| **quick_tests.py** | 3-5 min | Validar cada componente |
| **interactive_analysis.py** | 10 min | Menu interativo com análise |
| **demo_complete_workflow.py** | 10 min | Explicação detalhada de cada passo |
| **check_device.py** | 1 min | Diagnosticar hardware |
| **test_device_override.py** | 5 min | Testar/forçar diferentes devices |
| **intel-extension-for-pytorch** | 1 min | Para habilitar Intel Arc/Iris Xe (veja `GPU_CPU_SUPPORT.md`) |

### 📁 Código-Fonte

| Diretório | Conteúdo |
|-----------|----------|
| **src/config.py** | Configurações globais (+ novo: detect_device()) |
| **src/logger.py** | Sistema de logging estruturado |
| **src/pipeline.py** | Orquestrador principal |
| **src/ingest/** | Coleta de dados (Binance API) |
| **src/features/** | Engenharia de features (13+ indicadores) |
| **src/macronet/** | Rede MacroNet (encoder) |
| **src/micronet/** | Rede MicroNet (decision head) |
| **src/evaluation/** | Backtesting e métricas |
| **examples/basic_example.py** | Exemplo de uso |
| **tests/test_core.py** | Suite de testes (13 testes) |

### 📊 Dados e Modelos

| Diretório | Propósito |
|-----------|-----------|
| **data/** | Dados de candles em Parquet |
| **models/** | Pesos salvos das redes neurais |
| **embeddings/** | Cache de embeddings diários |
| **backtests/** | Resultados de backtests |

### ⚙️ Configuração

| Arquivo | Propósito |
|---------|-----------|
| **requirements.txt** | Dependências Python |
| **.env.example** | Variáveis de ambiente (template) |
| **.gitignore** | Arquivos ignorados pelo Git |

---

## 🎓 Ordem de Aprendizado Recomendada

### Nível 1: Entender o Projeto (30 minutos)
1. Ler `RESUMO_IMPLEMENTACAO.md`
2. Ler `README.md`
3. Executar `python run_quick_start.py`

### Nível 2: Usar o Sistema (1 hora)
1. Ler `COMO_EXECUTAR.md`
2. Executar `python quick_tests.py`
3. Executar `python interactive_analysis.py`
4. Explorar dados em `data/`

### Nível 3: Hardware e Performance (30 minutos)
1. Ler `GPU_CPU_SUPPORT.md`
2. Executar `python check_device.py`
3. Executar `python test_device_override.py`

### Nível 4: Desenvolvimento (2+ horas)
1. Ler `SETUP.md`
2. Ler `ROADMAP.md`
3. Explorar código em `src/`
4. Modificar `src/config.py` para ajustar hiperparâmetros

---

## 🚀 Guia Rápido por Objetivo

### "Quero ver tudo funcionando"
```bash
python run_quick_start.py
```

### "Quero testar cada parte"
```bash
python quick_tests.py
```

### "Tenho GPU e quero otimizar"
```bash
python check_device.py
python test_device_override.py
```

### "Quero analisar dados"
```bash
python interactive_analysis.py
# Escolher opção 1 ou 2
```

### "Quero usar em meu código"
```python
from src.pipeline import TradingPipeline

pipeline = TradingPipeline()
signal = pipeline.predict_signal("BTCUSDT")
print(f"Sinal: {signal:.4f}")
```

---

## 🔧 Troubleshooting Rápido

| Problema | Solução |
|----------|---------|
| "Module not found" | `pip install -r requirements.txt` |
| "Data not found" | Execute uma vez: `python run_quick_start.py` |
| "CUDA not available" | Ler `GPU_CPU_SUPPORT.md` seção NVIDIA |
| "Lento demais" | Executar: `python check_device.py` (verificar device) |
| "Out of memory" | Reduzir: `config.macronet.embedding_dim = 64` |

---

## 📞 Documentação Técnica

### Arquitetura do Modelo
- Ver: `README.md` seção "Arquitetura"
- Detalhes: `src/macronet/model.py` e `src/micronet/model.py`

### Features Técnicas
- Ver: `COMO_EXECUTAR.md` seção "Features"
- Implementação: `src/features/builder.py`

### Configuração
- Arquivo: `src/config.py`
- Novo: Método `detect_device()` para auto-detectar GPU

### Pipeline Completo
- Arquivo: `src/pipeline.py`
- Método: `fetch_and_prepare_data()`, `train_macronet()`, `predict_signal()`, etc.

---

## 🎯 Checklist antes de Começar

- [ ] Python 3.12 instalado (`python --version`)
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Hardware detectado (`python check_device.py`)
- [ ] Quick tests passando (`python quick_tests.py`)
- [ ] Quick start funcionando (`python run_quick_start.py`)
- [ ] Leu `COMO_EXECUTAR.md`

---

## 🌟 Arquivos Principais por Tipo de Usuário

### Para Iniciantes
1. `RESUMO_IMPLEMENTACAO.md` - Resumo
2. `run_quick_start.py` - Demo
3. `COMO_EXECUTAR.md` - Guia

### Para Desenvolvedores
1. `README.md` - Arquitetura
2. `src/pipeline.py` - Código principal
3. `GPU_CPU_SUPPORT.md` - Performance

### Para DevOps/MLOps
1. `SETUP.md` - Instalação
2. `GPU_CPU_SUPPORT.md` - Hardware
3. `requirements.txt` - Dependências

### Para Traders/Quants
1. `README.md` - Estratégia
2. `COMO_EXECUTAR.md` - Uso prático
3. `src/evaluation/backtest.py` - Métricas

---

## 🔄 Fluxo de Trabalho Típico

```
1. Setup
   └─ Ler SETUP.md
   └─ pip install -r requirements.txt
   └─ python check_device.py

2. Aprender
   └─ Ler RESUMO_IMPLEMENTACAO.md
   └─ Executar run_quick_start.py
   └─ Ler COMO_EXECUTAR.md

3. Experimentar
   └─ Executar quick_tests.py
   └─ Executar interactive_analysis.py
   └─ Modificar src/config.py

4. Otimizar
   └─ Ler GPU_CPU_SUPPORT.md
   └─ Executar test_device_override.py
   └─ Usar GPU se disponível

5. Produção
   └─ Ler ROADMAP.md (Fases 2-4)
   └─ Integrar em seu sistema
   └─ Monitorar performance
```

---

## 📊 Estatísticas do Projeto

- **Linhas de Código:** ~2000 em src/
- **Scripts Demo:** 6 executáveis
- **Documentação:** 6 arquivos Markdown (~3000 linhas)
- **Testes:** 13 unit tests
- **Suporte Hardware:** 4 tipos (CUDA, MPS, XPU, CPU)
- **Features Técnicas:** 13+ indicadores
- **Tempo para Setup:** ~5 minutos
- **Tempo para Entender:** ~1 hora

---

## 🎉 Você Tem

✅ Sistema completo de trading com Deep Learning
✅ Suporte para CPU, NVIDIA GPU, Intel GPU, Apple Silicon
✅ Pipeline end-to-end: dados → features → modelos → sinais
✅ Backtesting com métricas financeiras
✅ Documentação completa
✅ Scripts executáveis com demos
✅ Código pronto para produção

---

**Próximo passo:** `python run_quick_start.py` 🚀

---

**Última atualização:** Novembro 13, 2025
