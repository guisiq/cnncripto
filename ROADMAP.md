# Desenvolvimento Futuro

## Fase 1: Validação PoC (Atual)
- ✅ Ingestão básica Binance
- ✅ Features simples
- ✅ MacroNet autoencoder (encoder temporal)
- ✅ MicroNet (MLP + embedding)
- ✅ Backtesting básico
- [ ] Testes em dados ao vivo (papel trading)

## Fase 2: Refinamento (2-3 semanas)
### Arquitetura MacroNet
- [ ] Substituir CNN simples por Transformer leve (attention local)
- [ ] Implementar pré-treinamento contrastivo (SimCLR temporal)
- [ ] Adicionar regularização por complexidade

### MicroNet
- [ ] Threshold adaptativo inteligente (quantil dinâmico)
- [ ] Combinação multiobjetivo (Sharpe + Win Rate + Complexidade)
- [ ] Dropout/batch normalization tuning

### Pipeline
- [ ] Cache Redis para embeddings diários
- [ ] Monitoramento de drift automático
- [ ] Re-treino programado (semanal/adaptativo)

## Fase 3: Evolução Neuroevolutiva (4-6 semanas)
### Implementar CPPN/HyperNEAT
- [ ] Definir substrate geométrico para MacroNet
- [ ] Integrar NEAT-Python para evolução de CPPN
- [ ] Gerar topologias via evolução (fitness: Sharpe + penalidade complexidade)
- [ ] Híbrido: evolução + fine-tuning PyTorch pós-evolução

### Multiobjetivo
- [ ] Integrar DEAP para otimização multiobjetivo
- [ ] Frente Pareto: performance vs latência vs complexidade
- [ ] Seleção de melhor genoma por trade-off

## Fase 4: Escalabilidade (6-8 semanas)
### Multi-Símbolo
- [ ] Generalização transversal entre pares
- [ ] Embedding compartilhado (feature space unificado)
- [ ] Clusterização de símbolos por correlação

### Persistência Avançada
- [ ] Migração para DuckDB particionado
- [ ] Compactação automática de histórico antigo
- [ ] Retenção por política (30d online, 2y archive)

### API & Serving
- [ ] FastAPI endpoints: `/predict`, `/metrics`, `/status`
- [ ] Authentication (JWT tokens)
- [ ] Rate limiting e circuit breaker

## Fase 5: Robustez Operacional (8-10 semanas)
### Risk Management
- [ ] Position sizing dinâmico (volatilidade)
- [ ] Stop loss adaptativo
- [ ] Limite diário de drawdown

### Monitoramento
- [ ] Prometheus metrics
- [ ] Grafana dashboard
- [ ] Alertas de anomalias

### CI/CD
- [ ] GitHub Actions pipeline
- [ ] Testes automáticos + cobertura
- [ ] Containerização (Docker)

## Experiências Planejadas

| ID | Experimento | Objetivo | Critério Sucesso |
|----|-------------|----------|------------------|
| E1 | Macro vs Sem Macro | Validar embedding melhora | +5% hit ratio |
| E2 | MacroNet: Autoencoder vs Contrastive | Melhor pré-treino | Maior Sharpe + menor loss |
| E3 | MicroNet: Janela 2h vs 5h vs 10h | Horizonte ótimo | Máximo Sharpe e hit rate |
| E4 | Threshold estático vs adaptativo | Calibração sinais | Menos drawdown, mais precisão |
| E5 | CPPN: topologia vs rede fixa | Evolução vs baseline | Maior Sharpe com menos parâmetros |
| E6 | Multi-símbolo: individual vs compartilhado | Generalização | +3% performance avg. |

## Métricas de Sucesso (Roadmap)
| Milestone | Métrica Alvo | Data Alvo |
|-----------|-------------|----------|
| Baseline PoC | Sharpe > 0.5 | Nov 2025 |
| Refinamento | Sharpe > 1.5 | Dez 2025 |
| CPPN/NEAT | Sharpe > 2.0 + 50% menos params | Jan 2026 |
| Produção | Sharpe > 2.5 + multi-símbolo + <100ms latência | Fev 2026 |

## Riscos Identificados

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|--------|-----------|
| Overfitting | Alta | Alto | Cross-val rolling, regularização, penalidade complexidade |
| Drift de mercado | Alta | Médio | Re-treino automático por drift detector, portfólio robusto |
| Latência API | Média | Médio | Cache + GPU batch, pre-compute embeddings |
| Falhas Binance | Baixa | Médio | Fallback CCXT, cache offline, alertas |
| Custo computacional | Médio | Médio | Quantização modelos, pruning, distilação |

## Débitos Técnicos a Resolver

1. **Feature Store**: Migrar de Parquet simples para feature store (Feast/Tecton)
2. **Orquestração**: Airflow/Prefect para pipelines complexos
3. **Versionamento**: DVC para modelos e datasets
4. **Documentação**: Docstrings completas, type hints
5. **Testes**: Aumentar cobertura para >80%

## Contribuições Bem-Vindas

- [ ] Otimizações CUDA/TensorRT
- [ ] Novos tipos de features (orderbook, funding rates)
- [ ] Integração com mais exchanges
- [ ] Dashboard interativo (Streamlit/Dash)
- [ ] Comparação com outros modelos (LSTM, Transformer standalone)

---
**Última atualização**: Novembro 2025  
**Responsável**: Trading Research Team
