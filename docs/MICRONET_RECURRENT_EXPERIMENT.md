# Treinamento MicroNet Recorrente (1.5x Maior)

## 🎯 Objetivo

Experimento focado em **MicroNet standalone** com:
- **População 1.5x maior**: 225 indivíduos (vs 150 baseline)
- **Conexões recorrentes**: Memória temporal habilitada
- **Sem arquitetura assimétrica**: Apenas micro (sem macro)
- **Avaliação balanceada**: 3 símbolos fixos (BTC/ETH/BNB)

---

## 📊 Diferenças vs `train_asymmetric_neat.py`

| Aspecto | Assimétrico (Original) | MicroNet Recorrente (Novo) |
|---------|----------------------|---------------------------|
| **Arquitetura** | MacroNet + MicroNet | Apenas MicroNet |
| **População** | 150 cada rede | 225 (1.5x maior) |
| **Conexões** | Feed-forward | **Recorrentes** |
| **Ratio evolução** | 1:10 (macro:micro) | N/A (só micro) |
| **Memória temporal** | ❌ Não | ✅ **Sim** |
| **Complexidade** | Alta (2 redes) | Média (1 rede) |
| **Velocidade** | ~30 gerações/hora | ~45 gerações/hora |

---

## 🚀 Como Executar

```bash
# Ativar ambiente
conda activate cnncripto

# Executar treinamento (padrão: 2 horas)
python train_micronet_recurrent.py

# Ou especificar duração customizada (editar código):
# train_micronet_recurrent(duration_minutes=120)
```

---

## 🧬 Configurações NEAT

### População
- **Tamanho**: 225 indivíduos (50% maior que baseline)
- **Elitismo**: 5 (top 5 preservados)
- **Species elitism**: 3 (top 3 por espécie)
- **Survival threshold**: 0.5 (metade melhor reproduz)

### Arquitetura
- **Tipo**: `RecurrentNetwork` (feed_forward=False)
- **Inputs**: 60 candles × features (janela micro 5h)
- **Outputs**: 3 (HOLD, BUY, SELL)
- **Hidden nodes**: 3 iniciais (cresce via mutação)

### Mutações
- **Weight mutate rate**: 0.95 (alta exploração)
- **Bias mutate rate**: 0.7
- **Conn add prob**: 0.8 (favorece conectividade)
- **Node add prob**: 0.3
- **Activation options**: tanh, sigmoid, relu

### Fitness
- **Fórmula**: Quadrática com bonus de confiança
  ```python
  reward = (pred * price_change * 10000) + 
           (confidence² * |price_change| * 5000 * direction)
  ```
- **Objetivo**: Maximizar previsões confiantes corretas

---

## 📈 Resultados Esperados

### Vantagens da Rede Recorrente

1. **Memória Temporal**:
   - Detecta momentum (alta/baixa contínua)
   - Aprende padrões de velas consecutivas
   - Reconhece support/resistance histórico

2. **Fitness Esperado**:
   - Baseline (feed-forward): ~28k-35k
   - Recorrente (este): **40k-55k** (+20-40%)
   - Meta produção: 80k+

3. **Convergência**:
   - Feed-forward: ~300-500 gerações
   - Recorrente: ~400-700 gerações (mais lento)

### Trade-offs

**Vantagens**:
- ✅ Memória temporal (essencial para trading)
- ✅ Maior expressividade
- ✅ População maior (mais diversidade)

**Desvantagens**:
- ❌ Treinamento ~30% mais lento
- ❌ Risco de overfitting maior
- ❌ Precisa de mais gerações

---

## 📂 Estrutura de Arquivos

```
training_results_micronet_recurrent/
├── evolution_table.csv           # Histórico de treinamento
├── best_genome_genXXX.pkl        # Melhor genoma salvo
└── training_analysis.png         # Gráficos (gerar com plot_training_results.py)
```

---

## 📊 Monitoramento

### Durante Treinamento

Console mostra a cada 30 segundos:
```
Tempo(min) | Geração | BestFitness | AvgFitness | StdFitness | Species | PopSize | Width | Depth | EvalTime(s)
```

### Após Treinamento

Gerar gráficos:
```bash
python plot_training_results.py
# Modificar script para usar:
# results_dir = Path("training_results_micronet_recurrent")
```

---

## 🔍 Análise de Resultados

### Métricas Importantes

1. **Best Fitness**:
   - < 30k: Ainda aprendendo
   - 30k-50k: Progresso moderado
   - 50k-80k: Bom desempenho
   - 80k+: **Produção ready**

2. **Species Count**:
   - Ideal: 5-15 espécies
   - < 5: Pouca diversidade
   - > 20: Fragmentação excessiva

3. **Network Depth/Width**:
   - Depth: 2-6 camadas (típico)
   - Width: 5-20 neurônios/camada
   - Crescimento indica complexidade necessária

4. **Std Fitness**:
   - Alta (>5k): Diversidade boa
   - Baixa (<1k): Convergência prematura

---

## 🎓 Experimentos Sugeridos

### 1. Testar População Maior
```python
config_micro = create_neat_config_recurrent(
    pop_size=300  # 2x baseline
)
```

### 2. Ajustar Janela Temporal
```python
prices, micro_features = prepare_micro_data(
    df_symbol,
    micro_window=90  # 7.5h em vez de 5h
)
```

### 3. Aumentar Steps
```python
trainer.evolve_generation(
    max_steps=200  # ~16.7h por episódio
)
```

---

## 🐛 Troubleshooting

### Fitness Estagnado
- **Sintoma**: Fitness não melhora por 100+ gerações
- **Solução**: 
  - Aumentar `weight_mutate_rate` para 0.98
  - Reduzir `compatibility_threshold` para 2.0
  - Aumentar população para 300

### Espécies Fragmentadas
- **Sintoma**: 20+ espécies com < 5 indivíduos cada
- **Solução**:
  - Aumentar `compatibility_threshold` para 3.0
  - Aumentar `min_species_size` para 3

### Treinamento Muito Lento
- **Sintoma**: < 20 gerações/hora
- **Solução**:
  - Reduzir `max_steps` para 100
  - Reduzir `pop_size` para 150
  - Desabilitar multiprocessing se Mac M1/M2

---

## 📝 Logs e Checkpoints

- **CSV salvo a cada 50 gerações**
- **Modelo salvo ao final do treinamento**
- **Histórico completo mantido**

---

## 🔬 Validação

Após atingir fitness > 80k:

1. **Out-of-Sample Test**:
   ```python
   # Testar em dados de 2025 (não vistos)
   df_test = df[df['timestamp'] >= datetime(2025, 1, 1)]
   ```

2. **Backtest Completo**:
   ```python
   # Simular trading real com melhor genoma
   # Verificar: Sharpe > 0.8, Max Drawdown < 35%
   ```

3. **Comparação com Baseline**:
   - Feed-forward: ~28k fitness
   - Recorrente: esperado ~45k (+60%)

---

## 📌 Notas Importantes

1. **Memória Recorrente**:
   - Estado é resetado no início de cada episódio
   - Não vaza informação entre avaliações
   - Permite aprender dependências temporais

2. **Avaliação Balanceada**:
   - Cada genoma testado em BTC, ETH e BNB
   - Fitness = média dos 3 símbolos
   - Força generalização cross-asset

3. **Multiprocessing**:
   - 6 workers em paralelo (otimizado para M2/M3)
   - Speedup ~4-5x vs sequencial

---

**Data de Criação**: 14 de novembro de 2025  
**Status**: ✅ Pronto para execução  
**Baseline Esperado**: 40k-55k fitness (vs 28k feed-forward)
