# 🎯 Threshold de Fitness para Produção

## 📐 Fórmula de Fitness

```python
reward = (prediction_value * 100) * (price_change_pct * 100)
fitness = sum(rewards) / num_steps
```

Onde:
- `prediction_value`: saída da rede (-1 a +1, tipicamente)
- `price_change_pct`: mudança real do preço em porcentagem

## 🧮 Análise Matemática

### Exemplos de Cálculo

#### Caso 1: Previsão Perfeita de Alta
```
Previsão: +0.5 (prevê alta moderada)
Preço real: +0.3% (subiu 0.3%)
Reward: (0.5 × 100) × (0.3 × 100) = 50 × 30 = +1,500
```

#### Caso 2: Previsão Perfeita de Queda
```
Previsão: -0.5 (prevê queda moderada)
Preço real: -0.3% (caiu 0.3%)
Reward: (-0.5 × 100) × (-0.3 × 100) = -50 × -30 = +1,500
```

#### Caso 3: Previsão Errada (Pior Caso)
```
Previsão: +0.5 (prevê alta)
Preço real: -0.3% (mas caiu)
Reward: (0.5 × 100) × (-0.3 × 100) = 50 × -30 = -1,500
```

#### Caso 4: Previsão Neutra (Hold)
```
Previsão: 0.0 (indeciso)
Preço real: +0.3%
Reward: (0.0 × 100) × (0.3 × 100) = 0 × 30 = 0
```

## 📊 Benchmarks de Mercado

### Cripto (5 minutos)
- **Volatilidade média**: ±0.1% a ±0.5% por candle
- **Movimento extremo**: ±1% a ±3% por candle
- **Tendência forte**: ±0.3% consistente por 10+ candles

### Reward por Step
```
Volatilidade baixa (±0.1%):
- Previsão correta (0.5): 50 × 10 = +500
- Previsão errada (0.5): 50 × -10 = -500

Volatilidade média (±0.3%):
- Previsão correta (0.5): 50 × 30 = +1,500
- Previsão errada (0.5): 50 × -30 = -1,500

Volatilidade alta (±1%):
- Previsão correta (0.5): 50 × 100 = +5,000
- Previsão errada (0.5): 50 × -100 = -5,000
```

## 🎯 Valores de Fitness Esperados

### Por 150 Steps (~12.5 horas)

#### Modelo Aleatório (Baseline)
```
Taxa de acerto: 50%
Fitness médio: ~0 (±500 por step)
Fitness esperado: 0 ± 10,000
```

#### Modelo Fraco (Não Recomendado)
```
Taxa de acerto: 50-55%
Fitness médio: +200 a +500 por step
Fitness esperado: +30,000 a +75,000
Interpretação: Levemente melhor que aleatório
```

#### Modelo Aceitável (Mínimo para Produção)
```
Taxa de acerto: 55-60%
Fitness médio: +500 a +1,000 por step
Fitness esperado: +75,000 a +150,000
Interpretação: Consistentemente lucrativo
```

#### Modelo Bom (Produção Confiável)
```
Taxa de acerto: 60-65%
Fitness médio: +1,000 a +1,500 por step
Fitness esperado: +150,000 a +225,000
Interpretação: Forte edge no mercado
```

#### Modelo Excelente (Alta Performance)
```
Taxa de acerto: 65-70%
Fitness médio: +1,500 a +2,000 por step
Fitness esperado: +225,000 a +300,000
Interpretação: Performance profissional
```

#### Modelo Suspeito (Provavelmente Overfitting)
```
Taxa de acerto: >75%
Fitness médio: >+2,500 por step
Fitness esperado: >+375,000
⚠️ ALERTA: Provavelmente overfitting, validar em out-of-sample!
```

## 🚦 Critérios para Produção

### ✅ THRESHOLD MÍNIMO RECOMENDADO

#### Para MacroNet (long-term):
```python
MACRO_MIN_FITNESS = 100_000  # +100k em 150 steps
MACRO_GOOD_FITNESS = 200_000  # +200k em 150 steps
MACRO_EXCELLENT_FITNESS = 300_000  # +300k em 150 steps
```

**Justificativa**:
- 100k / 150 steps = ~667 reward/step
- Equivale a ~55-60% de acerto com volatilidade média
- Supera estratégia buy-and-hold em períodos laterais

#### Para MicroNet (short-term):
```python
MICRO_MIN_FITNESS = 80_000  # +80k em 150 steps
MICRO_GOOD_FITNESS = 150_000  # +150k em 150 steps
MICRO_EXCELLENT_FITNESS = 250_000  # +250k em 150 steps
```

**Justificativa**:
- 80k / 150 steps = ~533 reward/step
- Micro tem mais noise, threshold menor
- Foca em movimentos rápidos de curto prazo

### 📋 Checklist de Validação

Antes de colocar em produção, verificar:

#### 1. Fitness Consistente
```python
# Fitness deve ser positivo em MÚLTIPLOS períodos diferentes
fitness_by_month = [
    eval_period(model, jan_2024),
    eval_period(model, feb_2024),
    eval_period(model, mar_2024),
    # ...
]

# Todos devem ser > threshold
all_above_threshold = all(f > MACRO_MIN_FITNESS for f in fitness_by_month)
```

#### 2. Out-of-Sample Test
```python
# Testar em dados NÃO vistos no treinamento
# Ex: Treinou em 2023-2024, testar em Jan-Mar 2025
fitness_oos = eval_model(model, data_2025)

# Deve manter >80% do fitness de treinamento
acceptable_oos = fitness_oos > (MACRO_MIN_FITNESS * 0.8)
```

#### 3. Diferentes Condições de Mercado
```python
# Testar em:
# - Mercado em alta (bull)
# - Mercado em baixa (bear)
# - Mercado lateral (sideways)

fitness_bull = eval_market_condition(model, bull_period)
fitness_bear = eval_market_condition(model, bear_period)
fitness_sideways = eval_market_condition(model, sideways_period)

# Deve funcionar em TODOS os cenários
robust_model = all(f > MACRO_MIN_FITNESS * 0.7 for f in [fitness_bull, fitness_bear, fitness_sideways])
```

#### 4. Sharpe Ratio do Fitness
```python
# Fitness deve ter baixa volatilidade (consistente)
import numpy as np

fitness_history = [eval_window(model, i) for i in range(100)]
sharpe = np.mean(fitness_history) / (np.std(fitness_history) + 1e-8)

# Sharpe > 1.0 indica consistência
good_sharpe = sharpe > 1.0
```

#### 5. Drawdown Máximo
```python
# Pior sequência de rewards negativos
cumulative_rewards = np.cumsum(reward_history)
running_max = np.maximum.accumulate(cumulative_rewards)
drawdown = running_max - cumulative_rewards
max_drawdown = np.max(drawdown)

# Drawdown não deve exceder 30% do fitness total
acceptable_dd = max_drawdown < (total_fitness * 0.3)
```

## 🎯 Threshold Final Recomendado

### Produção Conservadora (Baixo Risco)
```python
PRODUCTION_CRITERIA = {
    'macro_fitness': {
        'min': 150_000,      # Fitness mínimo
        'oos_retention': 0.85,  # Manter 85% em out-of-sample
        'sharpe_ratio': 1.2,    # Alta consistência
        'max_drawdown': 0.25    # Drawdown máximo 25%
    },
    'micro_fitness': {
        'min': 120_000,
        'oos_retention': 0.80,
        'sharpe_ratio': 1.0,
        'max_drawdown': 0.30
    }
}
```

### Produção Moderada (Risco Médio)
```python
PRODUCTION_CRITERIA = {
    'macro_fitness': {
        'min': 100_000,
        'oos_retention': 0.75,
        'sharpe_ratio': 0.8,
        'max_drawdown': 0.35
    },
    'micro_fitness': {
        'min': 80_000,
        'oos_retention': 0.70,
        'sharpe_ratio': 0.7,
        'max_drawdown': 0.40
    }
}
```

### Produção Agressiva (Alto Risco) ⚠️
```python
PRODUCTION_CRITERIA = {
    'macro_fitness': {
        'min': 75_000,
        'oos_retention': 0.65,
        'sharpe_ratio': 0.5,
        'max_drawdown': 0.45
    },
    'micro_fitness': {
        'min': 60_000,
        'oos_retention': 0.60,
        'sharpe_ratio': 0.5,
        'max_drawdown': 0.50
    }
}
```

## 💡 Recomendação Final

### Para Ir para Produção:

#### Cenário Ideal (Recomendado)
```
✅ MacroNet fitness > 150,000 (em training)
✅ MicroNet fitness > 120,000 (em training)
✅ Out-of-sample fitness > 120,000 (macro) e > 95,000 (micro)
✅ Testado em 3+ meses diferentes
✅ Sharpe ratio > 1.0
✅ Max drawdown < 30%
✅ Lucrativo em bull, bear E sideways
```

#### Cenário Mínimo Aceitável
```
⚠️ MacroNet fitness > 100,000
⚠️ MicroNet fitness > 80,000
⚠️ Out-of-sample > 75,000 (macro) e > 60,000 (micro)
⚠️ Testado em 2+ meses
⚠️ Sharpe ratio > 0.7
⚠️ Max drawdown < 40%
⚠️ Lucrativo em pelo menos 2/3 condições de mercado

🚨 Usar apenas com capital de teste limitado!
```

#### Cenário de Rejeição ❌
```
❌ Fitness < 75,000
❌ Out-of-sample fitness < 50,000
❌ Sharpe ratio < 0.5
❌ Max drawdown > 50%
❌ Não lucrativo em mercado lateral
❌ Performance instável entre períodos

🛑 NÃO colocar em produção!
```

## 📈 Exemplo de Código de Validação

```python
def validate_for_production(model, train_data, test_data):
    """
    Valida se modelo está pronto para produção.
    
    Returns:
        (is_ready, report)
    """
    import numpy as np
    
    # 1. Fitness em training
    train_fitness = evaluate_model(model, train_data, steps=150)
    
    # 2. Fitness em test (out-of-sample)
    test_fitness = evaluate_model(model, test_data, steps=150)
    
    # 3. Múltiplos períodos
    monthly_fitness = []
    for month_data in split_by_month(test_data):
        fitness = evaluate_model(model, month_data, steps=150)
        monthly_fitness.append(fitness)
    
    # 4. Sharpe ratio
    sharpe = np.mean(monthly_fitness) / (np.std(monthly_fitness) + 1e-8)
    
    # 5. Drawdown
    rewards = get_reward_history(model, test_data)
    cumsum = np.cumsum(rewards)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = (running_max - cumsum) / (running_max + 1e-8)
    max_dd = np.max(drawdown)
    
    # Critérios
    checks = {
        'train_fitness': train_fitness >= 100_000,
        'test_fitness': test_fitness >= 75_000,
        'oos_retention': test_fitness >= train_fitness * 0.75,
        'sharpe_ratio': sharpe >= 0.8,
        'max_drawdown': max_dd <= 0.35,
        'consistent': all(f > 50_000 for f in monthly_fitness)
    }
    
    passed = sum(checks.values())
    total = len(checks)
    
    report = f"""
    ═══════════════════════════════════════════
    VALIDAÇÃO PARA PRODUÇÃO
    ═══════════════════════════════════════════
    
    📊 Métricas:
    - Train Fitness:     {train_fitness:>12,.0f} {'✅' if checks['train_fitness'] else '❌'}
    - Test Fitness:      {test_fitness:>12,.0f} {'✅' if checks['test_fitness'] else '❌'}
    - OOS Retention:     {test_fitness/train_fitness:>12.1%} {'✅' if checks['oos_retention'] else '❌'}
    - Sharpe Ratio:      {sharpe:>12.2f} {'✅' if checks['sharpe_ratio'] else '❌'}
    - Max Drawdown:      {max_dd:>12.1%} {'✅' if checks['max_drawdown'] else '❌'}
    - Consistência:      {'✅' if checks['consistent'] else '❌'}
    
    📈 Resultado:
    {passed}/{total} critérios atendidos
    
    {'🟢 APROVADO PARA PRODUÇÃO' if passed >= 5 else '🟡 PRODUÇÃO COM CAUTELA' if passed >= 4 else '🔴 NÃO APROVADO'}
    """
    
    return passed >= 5, report


# Uso
is_ready, report = validate_for_production(
    model=best_micro_genome,
    train_data=df_2023_2024,
    test_data=df_2025_q1
)

print(report)

if is_ready:
    print("\n✅ Modelo aprovado! Pode ir para produção.")
else:
    print("\n❌ Modelo precisa melhorar antes de produção.")
```

---

## 🎯 TL;DR - Resposta Rápida

**Valor mínimo aceitável para produção:**

- **MacroNet**: `fitness >= 100,000` (conservador: 150,000)
- **MicroNet**: `fitness >= 80,000` (conservador: 120,000)

**Com validação out-of-sample retendo pelo menos 75% do fitness.**

Isso garante que o modelo supera estratégias aleatórias e buy-and-hold com margem de segurança! 🚀
