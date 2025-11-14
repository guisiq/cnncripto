#!/usr/bin/env python3
"""
Gera gráficos de análise dos resultados de treinamento NEAT assimétrico.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend para salvar sem display
from pathlib import Path
import numpy as np

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# Carregar dados
results_dir = Path("training_results_asymmetric")

# Tentar carregar o arquivo mais recente
csv_files = list(results_dir.glob("evolution_table_*.csv"))
if not csv_files:
    print("❌ Nenhum arquivo CSV encontrado em training_results_asymmetric/")
    exit(1)

# Ordenar por nome e pegar o mais recente
latest_csv = sorted(csv_files)[-1]
print(f"📊 Carregando dados de: {latest_csv}")

df = pd.read_csv(latest_csv)
print(f"✅ {len(df)} registros carregados")
print(f"\nColunas disponíveis: {df.columns.tolist()}")
print(f"\nPrimeiras linhas:\n{df.head()}")

# Criar figura com múltiplos subplots
fig = plt.figure(figsize=(20, 14))

# 1. Fitness ao longo do tempo (Macro e Micro) - SE DISPONÍVEL
ax1 = plt.subplot(4, 4, 1)
if 'best_macro_fitness' in df.columns and 'best_micro_fitness' in df.columns:
    ax1.plot(df['episode'], df['best_macro_fitness'], label='MacroNet', linewidth=2, marker='o', markersize=2, alpha=0.7)
    ax1.plot(df['episode'], df['best_micro_fitness'], label='MicroNet', linewidth=2, marker='s', markersize=2, alpha=0.7)
    ax1.set_xlabel('Episódio', fontsize=10)
    ax1.set_ylabel('Best Fitness', fontsize=10)
    ax1.set_title('Evolução do Fitness', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
elif 'avg_reward' in df.columns:
    ax1.plot(df['episode'], df['avg_reward'], linewidth=2, marker='o', markersize=3, color='#2E86AB')
    ax1.set_xlabel('Episódio', fontsize=10)
    ax1.set_ylabel('Reward Médio', fontsize=10)
    ax1.set_title('Evolução do Reward', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)

# 2. Fitness médio da população (SE DISPONÍVEL)
ax2 = plt.subplot(4, 4, 2)
if 'macro_avg_fitness' in df.columns and 'micro_avg_fitness' in df.columns:
    ax2.plot(df['episode'], df['macro_avg_fitness'], label='Macro Avg', linewidth=2, color='#457B9D', alpha=0.7)
    ax2.plot(df['episode'], df['micro_avg_fitness'], label='Micro Avg', linewidth=2, color='#E63946', alpha=0.7)
    ax2.set_xlabel('Episódio', fontsize=10)
    ax2.set_ylabel('Fitness Médio', fontsize=10)
    ax2.set_title('Fitness Médio Populacional', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
elif 'avg_portfolio' in df.columns:
    ax2.plot(df['episode'], df['avg_portfolio'], linewidth=2, marker='s', markersize=3, color='#06A77D')
    ax2.axhline(y=10000, color='orange', linestyle='--', alpha=0.7, label='Capital Inicial')
    ax2.set_xlabel('Episódio', fontsize=10)
    ax2.set_ylabel('Portfolio ($)', fontsize=10)
    ax2.set_title('Portfolio Médio', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

# 3. Desvio padrão de fitness (diversidade populacional)
ax3 = plt.subplot(4, 4, 3)
if 'macro_std_fitness' in df.columns and 'micro_std_fitness' in df.columns:
    ax3.plot(df['episode'], df['macro_std_fitness'], label='Macro StdDev', linewidth=2, color='#023047', alpha=0.7)
    ax3.plot(df['episode'], df['micro_std_fitness'], label='Micro StdDev', linewidth=2, color='#FB8500', alpha=0.7)
    ax3.set_xlabel('Episódio', fontsize=10)
    ax3.set_ylabel('Desvio Padrão', fontsize=10)
    ax3.set_title('Diversidade Populacional (StdDev)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
elif 'avg_return_pct' in df.columns:
    ax3.plot(df['episode'], df['avg_return_pct'], linewidth=2, marker='d', markersize=3, color='#D62828')
    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Episódio', fontsize=10)
    ax3.set_ylabel('Retorno (%)', fontsize=10)
    ax3.set_title('Retorno Percentual', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

# 4. Número de espécies
ax4 = plt.subplot(4, 4, 4)
if 'macro_species_count' in df.columns and 'micro_species_count' in df.columns:
    ax4.plot(df['episode'], df['macro_species_count'], label='Macro Species', linewidth=2, marker='o', markersize=2, color='#9B59B6')
    ax4.plot(df['episode'], df['micro_species_count'], label='Micro Species', linewidth=2, marker='s', markersize=2, color='#3498DB')
    ax4.set_xlabel('Episódio', fontsize=10)
    ax4.set_ylabel('Número de Espécies', fontsize=10)
    ax4.set_title('Especiação ao Longo do Tempo', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

# 5. Número de atualizações
ax5 = plt.subplot(4, 4, 5)
if 'macro_updates' in df.columns and 'micro_updates' in df.columns:
    ax5.plot(df['episode'], df['macro_updates'], label='Macro Updates', linewidth=2.5, color='#023047')
    ax5.plot(df['episode'], df['micro_updates'], label='Micro Updates', linewidth=2.5, color='#FB8500')
    ax5.set_xlabel('Episódio', fontsize=10)
    ax5.set_ylabel('Número de Updates', fontsize=10)
    ax5.set_title('Atualizações de Rede (Ratio 1:10)', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

# 6. Melhoria de fitness por geração
ax6 = plt.subplot(4, 4, 6)
if 'macro_fitness_improvement' in df.columns and 'micro_fitness_improvement' in df.columns:
    ax6.plot(df['episode'], df['macro_fitness_improvement'], label='Macro', linewidth=2, alpha=0.7, color='#2C3E50')
    ax6.plot(df['episode'], df['micro_fitness_improvement'], label='Micro', linewidth=2, alpha=0.7, color='#E74C3C')
    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Episódio', fontsize=10)
    ax6.set_ylabel('Melhoria de Fitness', fontsize=10)
    ax6.set_title('Melhoria por Geração', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

# 7. Dimensões de rede - Largura (Width)
ax7 = plt.subplot(4, 4, 7)
if 'macro_width' in df.columns and 'micro_width' in df.columns:
    ax7.plot(df['episode'], df['macro_width'], label='Macro Width', linewidth=2, marker='o', markersize=2, alpha=0.7)
    ax7.plot(df['episode'], df['micro_width'], label='Micro Width', linewidth=2, marker='s', markersize=2, alpha=0.7)
    ax7.set_xlabel('Episódio', fontsize=10)
    ax7.set_ylabel('Largura (neurônios)', fontsize=10)
    ax7.set_title('Largura da Rede', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

# 8. Dimensões de rede - Profundidade (Depth)
ax8 = plt.subplot(4, 4, 8)
if 'macro_depth' in df.columns and 'micro_depth' in df.columns:
    ax8.plot(df['episode'], df['macro_depth'], label='Macro Depth', linewidth=2, marker='o', markersize=2, alpha=0.7)
    ax8.plot(df['episode'], df['micro_depth'], label='Micro Depth', linewidth=2, marker='s', markersize=2, alpha=0.7)
    ax8.set_xlabel('Episódio', fontsize=10)
    ax8.set_ylabel('Profundidade (camadas)', fontsize=10)
    ax8.set_title('Profundidade da Rede', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

# 9. Step_idx (posição no dataset)
ax9 = plt.subplot(4, 4, 9)
if 'step_idx' in df.columns:
    ax9.plot(df['episode'], df['step_idx'], linewidth=2, color='#16A085', marker='o', markersize=2)
    ax9.set_xlabel('Episódio', fontsize=10)
    ax9.set_ylabel('Step Index', fontsize=10)
    ax9.set_title('Progressão no Dataset', fontsize=11, fontweight='bold')
    ax9.grid(True, alpha=0.3)

# 10. Tempo de avaliação
ax10 = plt.subplot(4, 4, 10)
if 'eval_time_seconds' in df.columns:
    ax10.plot(df['episode'], df['eval_time_seconds'], linewidth=2, color='#8E44AD', marker='d', markersize=2)
    ax10.set_xlabel('Episódio', fontsize=10)
    ax10.set_ylabel('Tempo (s)', fontsize=10)
    ax10.set_title('Tempo de Avaliação por Geração', fontsize=11, fontweight='bold')
    ax10.grid(True, alpha=0.3)

# 11. Tamanho da população
ax11 = plt.subplot(4, 4, 11)
if 'macro_population_size' in df.columns and 'micro_population_size' in df.columns:
    ax11.plot(df['episode'], df['macro_population_size'], label='Macro Pop', linewidth=2, alpha=0.7)
    ax11.plot(df['episode'], df['micro_population_size'], label='Micro Pop', linewidth=2, alpha=0.7)
    ax11.set_xlabel('Episódio', fontsize=10)
    ax11.set_ylabel('Tamanho da População', fontsize=10)
    ax11.set_title('Tamanho Populacional', fontsize=11, fontweight='bold')
    ax11.legend(fontsize=9)
    ax11.grid(True, alpha=0.3)

# 12. Fase de curriculum learning
ax12 = plt.subplot(4, 4, 12)
if 'phase' in df.columns:
    # Mapear fases para números
    unique_phases = df['phase'].unique()
    phase_to_num = {phase: i for i, phase in enumerate(unique_phases)}
    df['phase_num'] = df['phase'].map(phase_to_num)
    
    ax12.plot(df['episode'], df['phase_num'], linewidth=2, marker='o', markersize=3, color='#E67E22')
    ax12.set_xlabel('Episódio', fontsize=10)
    ax12.set_ylabel('Fase', fontsize=10)
    ax12.set_title('Curriculum Learning - Progressão', fontsize=11, fontweight='bold')
    ax12.set_yticks(range(len(unique_phases)))
    ax12.set_yticklabels(unique_phases, fontsize=8)
    ax12.grid(True, alpha=0.3)

# 13. Média móvel de fitness (suavização)
ax13 = plt.subplot(4, 4, 13)
if 'best_macro_fitness' in df.columns and 'best_micro_fitness' in df.columns:
    window = min(20, len(df) // 10)
    if window >= 2:
        macro_ma = df['best_macro_fitness'].rolling(window=window).mean()
        micro_ma = df['best_micro_fitness'].rolling(window=window).mean()
        ax13.plot(df['episode'], macro_ma, linewidth=2.5, label=f'Macro MA({window})', alpha=0.8)
        ax13.plot(df['episode'], micro_ma, linewidth=2.5, label=f'Micro MA({window})', alpha=0.8)
        ax13.set_xlabel('Episódio', fontsize=10)
        ax13.set_ylabel('Fitness (MA)', fontsize=10)
        ax13.set_title(f'Tendência de Fitness (MA{window})', fontsize=11, fontweight='bold')
        ax13.legend(fontsize=9)
        ax13.grid(True, alpha=0.3)

# 14. Histograma de melhoria de fitness
ax14 = plt.subplot(4, 4, 14)
if 'macro_fitness_improvement' in df.columns:
    improvements = df['macro_fitness_improvement'][df['macro_fitness_improvement'] != 0]
    if len(improvements) > 0:
        ax14.hist(improvements, bins=30, color='#34495E', alpha=0.7, edgecolor='black')
        ax14.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax14.set_xlabel('Melhoria de Fitness', fontsize=10)
        ax14.set_ylabel('Frequência', fontsize=10)
        ax14.set_title('Distribuição de Melhorias', fontsize=11, fontweight='bold')
        ax14.grid(True, alpha=0.3, axis='y')

# 15. Scatter: Fitness vs Espécies
ax15 = plt.subplot(4, 4, 15)
if 'macro_species_count' in df.columns and 'best_macro_fitness' in df.columns:
    scatter = ax15.scatter(df['macro_species_count'], df['best_macro_fitness'], 
                          c=df['episode'], cmap='viridis', s=20, alpha=0.6)
    plt.colorbar(scatter, ax=ax15, label='Episódio')
    ax15.set_xlabel('Número de Espécies', fontsize=10)
    ax15.set_ylabel('Best Fitness', fontsize=10)
    ax15.set_title('Fitness vs Especiação', fontsize=11, fontweight='bold')
    ax15.grid(True, alpha=0.3)

# 16. Estatísticas finais (texto)
ax16 = plt.subplot(4, 4, 16)
ax16.axis('off')

stats_text = "ESTATISTICAS FINAIS\n" + "="*42 + "\n\n"

if 'best_macro_fitness' in df.columns:
    macro_final = df['best_macro_fitness'].iloc[-1]
    macro_max = df['best_macro_fitness'].max()
    macro_mean = df['best_macro_fitness'].mean()
    stats_text += f"MacroNet:\n"
    stats_text += f"  Final: {macro_final:,.1f}\n"
    stats_text += f"  Maximo: {macro_max:,.1f}\n"
    stats_text += f"  Media: {macro_mean:,.1f}\n\n"

if 'best_micro_fitness' in df.columns:
    micro_final = df['best_micro_fitness'].iloc[-1]
    micro_max = df['best_micro_fitness'].max()
    micro_mean = df['best_micro_fitness'].mean()
    stats_text += f"MicroNet:\n"
    stats_text += f"  Final: {micro_final:,.1f}\n"
    stats_text += f"  Maximo: {micro_max:,.1f}\n"
    stats_text += f"  Media: {micro_mean:,.1f}\n\n"

if 'macro_species_count' in df.columns:
    species_final = df['macro_species_count'].iloc[-1]
    species_avg = df['macro_species_count'].mean()
    stats_text += f"Especies (Macro):\n"
    stats_text += f"  Final: {species_final:.0f}\n"
    stats_text += f"  Media: {species_avg:.1f}\n\n"

if 'macro_width' in df.columns and 'macro_depth' in df.columns:
    width_final = df['macro_width'].iloc[-1]
    depth_final = df['macro_depth'].iloc[-1]
    stats_text += f"Rede (Macro):\n"
    stats_text += f"  Largura: {width_final}\n"
    stats_text += f"  Profundidade: {depth_final}\n\n"

if 'macro_updates' in df.columns:
    total_macro = df['macro_updates'].iloc[-1]
    stats_text += f"Total Macro Updates: {total_macro}\n"

if 'micro_updates' in df.columns:
    total_micro = df['micro_updates'].iloc[-1]
    stats_text += f"Total Micro Updates: {total_micro}\n"

stats_text += f"\nTotal Episodios: {len(df)}\n"

if 'time_min' in df.columns:
    total_time = df['time_min'].iloc[-1]
    stats_text += f"Tempo: {total_time:.1f} min ({total_time/60:.1f}h)\n"

stats_text += f"\n{'='*42}\n"
if 'best_macro_fitness' in df.columns:
    stats_text += f"Meta Producao: 80,000+\n"
    progress = (df['best_macro_fitness'].iloc[-1] / 80000) * 100
    stats_text += f"Progresso: {progress:.1f}%"
else:
    stats_text += f"Fitness em evolucao..."

ax16.text(0.05, 0.5, stats_text, fontsize=9, family='monospace',
         verticalalignment='center', 
         bbox=dict(boxstyle='round', facecolor='#F1FAEE', alpha=0.8, edgecolor='#457B9D', linewidth=2))

plt.suptitle('Analise Completa de Treinamento NEAT Assimetrico', 
             fontsize=16, fontweight='bold', y=0.998)

plt.tight_layout()

# Salvar figura
output_path = results_dir / "training_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Gráfico salvo em: {output_path}")

# Mostrar estatísticas no console
print("\n" + "="*60)
print("📊 RESUMO ESTATÍSTICO")
print("="*60)
print(df.describe())

plt.show()
