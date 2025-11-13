# cppncripto

## 1. Visão Geral
Este repositório implementa uma primeira prova de conceito (PoC) para previsão e geração de sinais de compra/venda em cripto ativos usando **duas redes**:

1. **MacroNet** (rede grande, executada uma vez por dia): processa um intervalo longo de candles de 5 minutos (ex: últimos 5 dias até 1 mês) e gera um _embedding macro_ que resume padrões temporais, de volatilidade e estrutura de mercado.
2. **MicroNet** (rede pequena, executada a cada avaliação intradiária): recebe as últimas horas (5h-10h) de candles + o embedding macro gerado pela MacroNet e produz uma decisão binária ou score contínuo (comprar / vender / neutro) em horizontes de 1m a 5m.

Esta arquitetura simplificada serve como base para futura evolução para um modelo com CPPN/HyperNEAT que gera topologias ou pesos da MacroNet.

## 2. Objetivos da PoC
- Reduzir custo computacional diário executando a rede mais pesada apenas uma vez.
- Validar se o embedding macro melhora a precisão dos sinais de curta duração.
- Estabelecer pipeline de dados, treinamento, inferência e avaliação contínua.
- Criar base modular para substituição futura da MacroNet por uma geração evolutiva (CPPN / HyperNEAT).

## 3. Escopo Inicial
| Item | Incluído | Justificativa |
|------|----------|---------------|
| Ingestão candles (Binance spot) | Sim | Dados intradiários 5m para múltiplos símbolos. |
| Features básicas | Sim | Retornos log, volatilidade rolling, volume normalizado. |
| MacroNet (Encoder Temporal) | Sim | Extração embedding macro diário. |
| MicroNet (Head de Decisão) | Sim | Score de sinal em tempo quase real. |
| Backtesting vetorizado (vectorbt) | Sim | Avaliar estratégia diariamente. |
| CPPN / NEAT | Não (fase posterior) | Primeiro validar arquitetura 2 redes. |
| Multi-exchange (CCXT) | Não | Foca em Binance para reduzir variáveis. |

## 4. Arquitetura Simplificada
```
+------------------+              +--------------------+                  +-----------------------+
| Ingestão & Cache | -- diário -> | MacroNet (Encoder) | -- embedding --> | MicroNet (Decision Head) |
+------------------+              +--------------------+                  +-----------------------+
       | (Parquet / DuckDB)                 |                                  |
       v                                    |                                  v
  Pré-processamento ----> Dataset Longo ----+                       Sinais / Métricas / Logs
```

### 4.1 MacroNet
- Entrada: sequência longa de candles (T_long x F_features). Ex: T_long = 1440 * 5 dias / 5m ≈ 1440 candles/dia * 5 = 7200.
- Função: Comprimi-los em um vetor de dimensão fixa `embedding_dim` (ex: 128).
- Arquiteturas candidatas: 
  - Temporal CNN dilatada.
  - Transformer leve (atenção reduzida com janela local + pooling).
  - GRU empilhada com atenção simples.
- Saída: `macro_embedding` (shape [embedding_dim]).

### 4.2 MicroNet
- Entrada: janela curta intradiária (T_short x F_features) + `macro_embedding` concatenado ou usado como contexto.
  - Ex: T_short = últimas 60 candles (5h em 5m = 60) → shape [60, F].
- Combinação: 
  1. Processar janela curta via CNN/GRU → vetor local.
  2. Concatenar com `macro_embedding` → MLP final → score.
- Saída: Score contínuo `s` ∈ [-1,1] ou probabilidade `p_buy`, `p_sell`.

## 5. Fluxo Diário
1. 00:00 UTC (ou início da sessão definida): carregar últimos N dias de dados → MacroNet → gerar `macro_embedding` e armazenar em cache.
2. Durante o dia, a cada novo candle de 5m: 
   - Atualizar janela T_short.
   - Rodar MicroNet com embedding macro fixo.
   - Gerar sinal (se abs(score) > threshold adaptativo).
3. Coletar performance em backtest (vectorbt) ao final do dia.
4. Atualizar métricas e decidir se retreinar (ex: semanal ou drift). 

## 6. Dados e Pré-processamento
- Fonte: REST API Binance (endpoint klines 5m) + WebSocket para último dia (incremental).
- Colunas: `open, high, low, close, volume, quote_volume, trades_count, taker_buy_volume`.
- Features derivadas:
  - Retorno log: `log(close_t / close_{t-1})`.
  - Volatilidade: std rolling (janelas 12, 24, 48).
  - Volume normalizado (z-score nos últimos X candles).
  - High-Low range normalizado.
- Normalização: MinMax ou RobustScaler por símbolo diário (evitar vazamento entre dias). Ideal: calcular estatísticas do período usado pela MacroNet, aplicar mesma escala na MicroNet.

## 7. Treinamento
### 7.1 MacroNet
- Dataset: várias janelas longas históricas (deslizantes por dia). Targets indiretos: não precisa rótulo direto; pode ser auto-supervisionado.
- Estratégias: 
  - Autoencoder temporal (reconstrução). 
  - Contrastive (SimCLR temporal entre segmentos do mesmo dia). 
  - Máscara (masked modeling). 
- Resultado: embedding robusto que preserva padrões intradiários.

### 7.2 MicroNet
- Rótulos: sinais derivados de regra heurística inicial (ex: retorno futuro em 5m > threshold → class 1; < -threshold → class -1; caso contrário 0) OU usar simulação de estratégia (entry/exit) para gerar labels.
- Loss: 
  - Classificação multi-classe (CrossEntropy) OU regressão (MSE com retorno futuro). 
  - Penalização de overconfidence (regularização L2). 

### 7.3 Ciclo
1. Treinar MacroNet offline (diário/semanal). 
2. Congelar MacroNet → gerar embeddings para histórico.
3. Treinar MicroNet com embeddings + janelas curtas. 
4. Validar em conjunto hold-out (últimos dias não vistos).

## 8. Inferência
Pseudo-código:
```python
macro_embedding = run_macronet(long_sequence)  # executado 1x por dia

for each new 5m candle:
    short_window = get_last_short_window()
    x_short = build_features(short_window)
    score = micronet(x_short, macro_embedding)
    if abs(score) > threshold:
        emit_signal(score)
```

Threshold adaptativo: quantil dos scores históricos (ex: top 15% compra, bottom 15% venda). Ajustar dinamicamente se número de sinais estiver muito alto/baixo.

## 9. Métricas de Avaliação
| Categoria | Métrica |
|-----------|--------|
| Financeira | Retorno diário, Sharpe, Sortino, Max Drawdown |
| Precisão | Precision/Recall de sinais positivos, Hit Ratio (direção correta) |
| Operacional | Latência média inferência MicroNet, tempo geração MacroNet |
| Robustez | Deriva de embedding (distância média entre dias) |

## 10. Estrutura de Pastas Proposta
```
cppncripto/
  data/                 # Parquet / brutos / features
  notebooks/            # Exploração inicial
  src/
    ingest/             # Scripts de coleta Binance
    features/           # Transformações e normalização
    macronet/           # Modelo + treinamento
    micronet/           # Modelo + treinamento
    evaluation/         # Backtesting e métricas
    api/                # FastAPI endpoints (opcional)
  models/
    macronet/           # Pesos salvos
    micronet/           # Pesos salvos
  embeddings/           # Cache diário de macro embeddings
  backtests/            # Relatórios JSON/CSV
  configs/              # YAML/JSON hiperparâmetros
  README.md
```

## 11. Roadmap Incremental
| Fase | Entrega |
|------|---------|
| 1 | Ingestão + armazenamento candles 5m + features básicos |
| 2 | Protótipo MacroNet (autoencoder simples) + geração embedding diário |
| 3 | Protótipo MicroNet (MLP com janela curta + embedding) |
| 4 | Pipeline inferência contínua + threshold adaptativo + logs |
| 5 | Backtesting vectorbt integrado + métricas |
| 6 | Refinar MacroNet (Transformer leve / CNN dilatada) |
| 7 | Regularização, tuning hiperparâmetros, early stopping |
| 8 | Introduzir evolução CPPN para substituir MacroNet |
| 9 | Multi-símbolo e generalização transversal |
| 10 | Segurança, API pública e monitoramento avançado |

## 12. Plano de Experimentos
| Experimento | Objetivo | Critério Sucesso |
|-------------|----------|------------------|
| E1: Macro vs Sem Macro | Ver se embedding melhora hit ratio | +5% hit ratio sobre baseline sem embedding |
| E2: Autoencoder vs Contrastive | Escolher melhor pré-treinamento | Melhor Sharpe + menor reconstruction loss |
| E3: Janela curta 2h vs 5h | Ajustar horizonte ótimo | Maior retorno ajustado ao risco |
| E4: Threshold estático vs adaptativo | Calibrar disparo de sinais | Menor drawdown e alta precisão |

## 13. Evolução Futura (CPPN / HyperNEAT)
Após validar ganho do embedding macro, substituir MacroNet por geração evolutiva:
- CPPN gera pesos/esparsidade de uma CNN temporal.
- Fitness: performance + simplicidade (número de parâmetros). 
- Etapa híbrida: fine-tune rápido pós-evolução com PyTorch.

## 14. Riscos e Mitigações (PoC)
| Risco | Mitigação |
|-------|-----------|
| Overfitting MicroNet | Regularização + validação temporal separada |
| Latência durante pico | Pré-carregar embedding + evitar recomputar features pesadas |
| Falhas ingestão API | Retry + cache local + verificação completude |
| Padrões não estáveis | Re-treino periódico + monitor drift |

## 15. Próximos Passos Imediatos
1. Implementar módulo de ingestão (Binance klines 5m) + persistência Parquet.
2. Criar script de geração de features básicas (retorno, volatilidade, volume). 
3. Prototipar MacroNet autoencoder (PyTorch) e salvar primeiro embedding.
4. Criar MicroNet simples (MLP) e script de inferência por loop temporal.
5. Integrar backtest inicial (vectorbt) comparando “com embedding” vs “sem embedding”.

## 16. Glossário
- **Embedding Macro**: Vetor denso representando longo histórico intradiário.
- **Janela Curta (T_short)**: Intervalo recente usado para decisão (últimas horas).
- **Autoencoder Temporal**: Rede que comprime sequência e tenta reconstruir, aprendendo representação útil.
- **Threshold Adaptativo**: Limite dinâmico baseado em distribuição recente do score.

## 17. Licença / Uso
Definir licença futuramente (ex: MIT) após validação interna.

---
Este documento deverá ser revisado conforme resultados dos primeiros experimentos e antes da introdução de CPPN/HyperNEAT.
