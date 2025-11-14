# 🧬 Treinamento Assimétrico com NEAT

Implementação de RL assimétrico usando **NEAT (NeuroEvolution of Augmenting Topologies)** em vez de redes PyTorch pré-definidas.

## 📁 Arquivos Criados

### `train_asymmetric_neat.py` (Principal)
Arquivo principal que implementa evolução assimétrica com duas populações NEAT:
- **MacroNet**: Evolui a cada 2 gerações (topologia para contexto de longo prazo)
- **MicroNet**: Evolui a cada geração (topologia para contexto de curto prazo)

**Classes principais:**
- `NEATNetworkAdapter`: Converte genomas NEAT para forward pass
- `TradingEnvironmentRL`: Ambiente para avaliar fitness dos genomas
- `AsymmetricNEATTrainer`: Gerencia populações NEAT e evolução

**Funções principais:**
- `create_neat_config(input_nodes, output_nodes)`: Cria config NEAT customizada
- `train_asymmetric_neat(duration_minutes, log_interval_seconds, num_envs, population_size)`: Executa treinamento

### `neat_config_template.txt` (Configuração)
Template NEAT com todos parâmetros obrigatórios. Usado como base para criar configs customizadas dinamicamente.

## 🚀 Como Usar

### Execução Básica
```bash
cd /Users/vlngroup/Desktop/cnncripto
python train_asymmetric_neat.py
```

### Execução com Parâmetros Customizados
```python
from train_asymmetric_neat import train_asymmetric_neat

# Treinar por 5 minutos com 8 ambientes
train_asymmetric_neat(
    duration_minutes=5,
    log_interval_seconds=30,
    num_envs=8,
    population_size=50
)
```

## 📊 O que Esperar

1. **Carregamento de dados**: Lê parquet local (2024 data)
2. **Preparação de features**: Cria janelas assimétricas (macro=492 candles, micro=60 candles)
3. **Criação de ambientes**: Cria N_envs ambientes independentes
4. **Evolução**: Gerações de NEAT com avaliação de fitness sobre episódios de trading
5. **Log**: Tabela periódica com:
   - Gen: Número da geração
   - Tempo(min): Tempo decorrido
   - MacroFit: Fitness do melhor genoma MacroNet
   - MicroFit: Fitness do melhor genoma MicroNet
   - PopMacro: Tamanho população MacroNet
   - PopMicro: Tamanho população MicroNet

## 🔧 Parâmetros NEAT (no template)

### Especiação
- `compatibility_threshold = 3.0`: Distância máxima para mesma espécie
- `compatibility_disjoint_coefficient = 1.0`: Peso de genes disjuntos
- `compatibility_weight_coefficient = 0.5`: Peso de diferença de pesos

### Mutação
- `conn_add_prob = 0.5`: Probabilidade de adicionar conexão
- `conn_delete_prob = 0.5`: Probabilidade de remover conexão
- `node_add_prob = 0.2`: Probabilidade de adicionar nó
- `node_delete_prob = 0.2`: Probabilidade de remover nó
- `weight_mutate_rate = 0.8`: Taxa de mutação de pesos

### Reprodução
- `elitism = 2`: Melhores indivíduos preservados
- `survival_threshold = 0.2`: % da população que reproduz
- `max_stagnation = 20`: Gerações máximas sem melhoria antes de reset

## 🎯 Diferenças entre Abordagens

### RL com PyTorch (`train_asymmetric_rl.py`)
✅ Arquitetura controlada (sabemos exatamente quantas camadas)
✅ Treinamento mais rápido (gradient descent)
✅ Determinístico (dado o seed)
❌ Requer design manual de rede
❌ Pode subaprender ou overfitar

### NEAT (`train_asymmetric_neat.py`)
✅ Topologia evolui automaticamente
✅ Encontra arquitetura ótima para o problema
✅ Menos risco de overfitting (especiação preserva diversidade)
❌ Mais lento (avaliação de múltiplos genomas)
❌ Menos determinístico (crossover + mutação)
❌ Não usa GPU eficientemente (apenas forward pass)

## 💡 Próximos Passos Recomendados

1. **Testar performance**: Rodar ambos os scripts por mesmo tempo e comparar convergência
2. **Hybrid approach**: Usar NEAT para encontrar topologia, depois treinar com PyTorch
3. **Paralelização**: Implementar avaliação paralela de genomas em múltiplas CPUs
4. **Especiação aprimorada**: Ajustar thresholds de compatibilidade conforme dados

## 📈 Métricas de Sucesso

- **MacroFit > 0**: Rede ganhando dinheiro em média
- **MicroFit > MacroFit**: Micro se especializando bem (esperado)
- **PopMacro/PopMicro crescendo**: Diversidade aumentando
- **Convergência**: Fitness melhorando ao longo do tempo (não estagnando)

## 🐛 Troubleshooting

**Erro: "Arquivo de template NEAT não encontrado"**
- Certifique-se que `neat_config_template.txt` está no diretório raiz do projeto

**Evolução muito lenta**
- Reduzir `population_size` (mas manter ≥30)
- Aumentar `conn_add_prob` e `node_add_prob` para exploração
- Reduzir `max_stagnation` para permitir resets

**Fitness negativo em todas as gerações**
- Dataset pode ser insuficiente
- Aumentar `duration_minutes` para dar mais tempo
- Verificar se parquet tem dados válidos de 2024

---

**Versão**: 1.0 - NEAT Assimétrico  
**Data**: 13 de novembro de 2025
