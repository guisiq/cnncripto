"""
Pipeline de Treinamento Baseado em Tempo (10 minutos)

Pipeline Simplificado e OTIMIZADO:
1. Contexto longo → MacroNet → embedding (PRÉ-TREINADO 1x)
2. Contexto curto + embedding → MicroNet → sinal [-1, 1]
3. Target: Tendência futura 5min (positivo=+1, negativo=-1)
4. Loss: MSE entre confiança do sinal e aumento percentual real
5. Treinar por 10 minutos independente de épocas
6. Log evolução a cada 30 segundos

Otimizações para Velocidade:
- MacroNet treinado UMA VEZ antes do loop (é encoder fixo)
- Embeddings gerados UMA VEZ e reutilizados
- MicroNet treina em mini-batches aleatórios (não dataset completo)
- Batch size aumentado: 64 (convergência mais rápida)
- Métricas calculadas com sample pequeno (50 exemplos)
- Predições com torch.no_grad() para economizar memória
"""

import numpy as np
import pandas as pd
import torch
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.pipeline import TradingPipeline
from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


def prepare_simple_targets(data: pd.DataFrame) -> tuple:
    """
    Prepara dados com targets simples: tendência futura 5min
    
    Returns:
        X_short, X_long, y_targets
    """
    lookback = config.micronet.lookback_candles  # 60 candles de 5min = 5h
    
    X_short_list = []
    X_long_list = []
    y_list = []
    
    # Extrair features numéricas
    numeric_cols = []
    for col in data.columns:
        if np.issubdtype(data[col].dtype, np.number):
            numeric_cols.append(col)
    
    feature_data = data[numeric_cols].values.astype(np.float32)
    prices = data['close'].values
    
    print(f"📊 Preparando dados: {len(data)} candles, {len(numeric_cols)} features")
    
    for i in range(lookback, len(data) - 1):  # -1 para ter futuro
        # Contexto curto (últimos 60 candles)
        X_short = feature_data[i-lookback:i]
        
        # Contexto longo (todos até agora)
        X_long = feature_data[:i]
        
        # Target: variação percentual do próximo candle (5min)
        current_price = prices[i]
        future_price = prices[i + 1]
        pct_change = (future_price - current_price) / current_price
        
        # Converter para target binário com threshold MENOR
        # >0.1% = +1 (long), <-0.1% = -1 (short)
        # REMOVEMOS neutros para forçar rede a escolher direção
        threshold = 0.0005  # 0.05% - mais sensível
        
        if pct_change > threshold:
            target = 1.0
        elif pct_change < -threshold:
            target = -1.0
        else:
            # Neutro: classificar pela tendência geral
            target = 1.0 if pct_change >= 0 else -1.0
        
        X_short_list.append(X_short)
        X_long_list.append(X_long)
        y_list.append(target)
    
    print(f"✅ Preparados {len(y_list)} exemplos de treinamento")
    
    # Estatísticas dos targets
    y_array = np.array(y_list)
    n_long = np.sum(y_array == 1.0)
    n_short = np.sum(y_array == -1.0)
    n_neutral = np.sum(y_array == 0.0)
    
    print(f"   Long (+1):    {n_long} ({n_long/len(y_list)*100:.1f}%)")
    print(f"   Short (-1):   {n_short} ({n_short/len(y_list)*100:.1f}%)")
    print(f"   Neutral (0):  {n_neutral} ({n_neutral/len(y_list)*100:.1f}%)")
    
    return X_short_list, X_long_list, y_list, feature_data.shape[1]


def plot_training_history(history: dict, output_dir: Path, total_time: float, epochs: int, best_loss: float):
    """Plota gráfico com histórico de treinamento"""
    import matplotlib.pyplot as plt
    
    if len(history['time_min']) == 0:
        print("⚠️  Sem dados para plotar")
        return
    
    print(f"\n📊 Gerando gráfico de evolução...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Evolução do Treinamento - {total_time/60:.1f} min, {epochs} épocas', 
                 fontsize=14, fontweight='bold')
    
    time_axis = history['time_min']
    
    # 1. Loss ao longo do tempo
    ax1 = axes[0, 0]
    ax1.plot(time_axis, history['loss'], 'b-', linewidth=2, label='Loss')
    ax1.axhline(y=best_loss, color='r', linestyle='--', label=f'Best: {best_loss:.4f}')
    ax1.set_xlabel('Tempo (minutos)')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Loss ao Longo do Tempo')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Acurácia ao longo do tempo
    ax2 = axes[0, 1]
    accuracy_pct = [a * 100 for a in history['accuracy']]
    ax2.plot(time_axis, accuracy_pct, 'g-', linewidth=2)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    ax2.set_xlabel('Tempo (minutos)')
    ax2.set_ylabel('Acurácia (%)')
    ax2.set_title('Acurácia de Direção')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 100])
    
    # 3. Média das predições
    ax3 = axes[1, 0]
    ax3.plot(time_axis, history['pred_mean'], 'purple', linewidth=2, label='Média')
    ax3.fill_between(
        time_axis,
        [m - s for m, s in zip(history['pred_mean'], history['pred_std'])],
        [m + s for m, s in zip(history['pred_mean'], history['pred_std'])],
        alpha=0.3,
        label='±1 std'
    )
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.3, label='Long (+1)')
    ax3.axhline(y=-1, color='blue', linestyle='--', alpha=0.3, label='Short (-1)')
    ax3.set_xlabel('Tempo (minutos)')
    ax3.set_ylabel('Valor da Predição')
    ax3.set_title('Distribuição das Predições')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([-1.2, 1.2])
    
    # 4. Épocas por minuto (velocidade)
    ax4 = axes[1, 1]
    epochs_per_interval = []
    for i in range(1, len(history['epoch'])):
        epoch_diff = history['epoch'][i] - history['epoch'][i-1]
        time_diff = history['time_min'][i] - history['time_min'][i-1]
        if time_diff > 0:
            epochs_per_interval.append(epoch_diff / time_diff)
        else:
            epochs_per_interval.append(0)
    
    if epochs_per_interval:
        ax4.plot(time_axis[1:], epochs_per_interval, 'orange', linewidth=2)
        ax4.set_xlabel('Tempo (minutos)')
        ax4.set_ylabel('Épocas / minuto')
        ax4.set_title('Velocidade de Treinamento')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center',
                transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # Salvar
    plot_path = output_dir / 'training_evolution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico salvo: {plot_path}")
    
    # Tentar mostrar (se em ambiente interativo)
    try:
        plt.show(block=False)
    except:
        pass
    
    plt.close()


def train_time_based(
    pipeline: TradingPipeline,
    X_short_list: list,
    X_long_list: list,
    y_list: list,
    num_features: int,
    duration_minutes: int = 10,
    log_interval_seconds: int = 30,
    batch_size: int = 64  # Aumentado para 64 (convergência mais rápida)
):
    """
    Treina por tempo fixo (não épocas) com logging de evolução
    
    Args:
        duration_minutes: Tempo total de treinamento
        log_interval_seconds: Intervalo para log de métricas
    """
    
    print(f"\n{'='*70}")
    print(f"  TREINAMENTO POR TEMPO: {duration_minutes} MINUTOS")
    print(f"{'='*70}\n")
    
    # Construir modelos se necessário
    if pipeline.macronet.model is None:
        pipeline.macronet.build_model(input_dim=num_features)
        print(f"✅ MacroNet construída: {num_features} features → 128 embedding")
    
    if pipeline.micronet.model is None:
        short_dim = config.micronet.lookback_candles * num_features
        pipeline.micronet.build_model(
            short_features_dim=short_dim,
            macro_embedding_dim=128
        )
        print(f"✅ MicroNet construída: {short_dim} + 128 → sinal")
    
    # Preparar tensors
    print(f"\n📦 Preparando dados para treinamento...")
    
    # Para MacroNet: usar todos os contextos longos (variável)
    # Para MicroNet: padronizar em array fixo
    X_short_array = np.array(X_short_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    
    n_samples = len(X_short_list)
    print(f"   {n_samples} amostras prontas")
    
    # Controle de tempo
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    last_log_time = start_time
    
    iteration = 0
    epoch = 0
    best_loss = float('inf')
    
    # Histórico para gráfico
    history = {
        'time_min': [],
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'pred_mean': [],
        'pred_std': []
    }
    
    print(f"\n🚀 Iniciando treinamento...")
    print(f"   Início: {datetime.now().strftime('%H:%M:%S')}")
    print(f"   Fim esperado: {(datetime.now() + timedelta(minutes=duration_minutes)).strftime('%H:%M:%S')}")
    print(f"\n{'─'*70}")
    
    # Treinar MacroNet UMA VEZ antes do loop (ele é encoder, não precisa retreinar toda hora)
    print("🔧 Pré-treinando MacroNet (5 épocas)...")
    for X_long in X_long_list[:5]:  # Apenas primeiras 5 amostras para velocidade
        X_long_batch = X_long[np.newaxis, :, :].astype(np.float32)
        pipeline.macronet.train(X_long_batch, epochs=1, batch_size=1)
    print("✅ MacroNet pré-treinado!")
    
    # Gerar todos os embeddings uma vez
    print("🔧 Gerando embeddings macro...")
    macro_embeddings = []
    for X_long in X_long_list:
        X_long_batch = X_long[np.newaxis, :, :].astype(np.float32)
        emb = pipeline.macronet.encode(X_long_batch)[0]
        macro_embeddings.append(emb)
    X_macro_array = np.array(macro_embeddings, dtype=np.float32)
    print(f"✅ {len(macro_embeddings)} embeddings gerados!")
    
    # Adicionar ruído inicial para evitar convergência prematura
    exploration_phase = True
    
    while time.time() < end_time:
        epoch += 1
        epoch_start = time.time()
        
        # Treinar APENAS MicroNet (muito mais rápido!)
        # Usar mini-batch aleatório para velocidade
        batch_indices = np.random.choice(n_samples, min(batch_size * 4, n_samples), replace=False)
        
        # Adicionar ruído aos targets nas primeiras iterações (exploração)
        y_batch = y_array[batch_indices].copy()
        if exploration_phase and iteration < 50:
            noise_scale = 0.3 * (1 - iteration / 50)
            noise = np.random.normal(0, noise_scale, y_batch.shape).astype(np.float32)
            y_batch = np.clip(y_batch + noise, -1, 1)
        elif iteration == 50:
            exploration_phase = False
            print("🎯 Fase de exploração concluída! Iniciando convergência...")
        
        pipeline.micronet.train(
            X_short_array[batch_indices],
            X_macro_array[batch_indices],
            y_batch,
            epochs=1,
            batch_size=batch_size
        )
        
        iteration += 1
        
        # 3. Calcular métricas de evolução (apenas a cada log_interval)
        current_time = time.time()
        
        if current_time - last_log_time >= log_interval_seconds:
            elapsed = current_time - start_time
            remaining = end_time - current_time
            progress = (elapsed / (duration_minutes * 60)) * 100
            
            # Calcular loss atual (sample pequeno de 50 exemplos)
            sample_size = min(50, n_samples)
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            
            X_short_sample = X_short_array[sample_indices]
            X_macro_sample = X_macro_array[sample_indices]
            y_sample = y_array[sample_indices]
            
            # Predict (mais rápido sem gradientes)
            with torch.no_grad():
                predictions = pipeline.micronet.predict(X_short_sample, X_macro_sample)
            
            # Loss MSE
            loss = float(np.mean((predictions - y_sample) ** 2))
            
            # Acurácia (sinal correto)
            pred_sign = np.sign(predictions)
            true_sign = np.sign(y_sample)
            accuracy = float(np.mean(pred_sign == true_sign))
            
            # Estatísticas das predições
            pred_mean = float(np.mean(predictions))
            pred_std = float(np.std(predictions))
            pred_min = float(np.min(predictions))
            pred_max = float(np.max(predictions))
            
            # Update best
            if loss < best_loss:
                best_loss = loss
                improvement = "🔥"
            else:
                improvement = "  "
            
            # Estatísticas dos targets na amostra
            y_sample_pos = np.sum(y_sample > 0.5)
            y_sample_neg = np.sum(y_sample < -0.5)
            
            # Estatísticas das predições
            pred_pos = np.sum(predictions > 0.5)
            pred_neg = np.sum(predictions < -0.5)
            
            # Log
            print(f"\n{'─'*70}")
            print(f"⏱️  Tempo: {elapsed/60:.1f}min / {duration_minutes}min ({progress:.1f}%)")
            print(f"📈 Época: {epoch} | Iterações: {iteration}")
            print(f"📉 Loss: {loss:.6f} {improvement} | Best: {best_loss:.6f}")
            print(f"🎯 Acurácia: {accuracy*100:.1f}%")
            print(f"📊 Predições: μ={pred_mean:.3f}, σ={pred_std:.3f}, [{pred_min:.3f}, {pred_max:.3f}]")
            print(f"   ↑Long: {pred_pos}/{sample_size} | ↓Short: {pred_neg}/{sample_size}")
            print(f"🎲 Targets:   ↑Long: {y_sample_pos}/{sample_size} | ↓Short: {y_sample_neg}/{sample_size}")
            print(f"⏳ Restante: {remaining/60:.1f}min")
            print(f"{'─'*70}")
            
            logger.info(
                "training_progress",
                epoch=epoch,
                iteration=iteration,
                elapsed_min=elapsed/60,
                progress_pct=progress,
                loss=loss,
                best_loss=best_loss,
                accuracy=accuracy,
                pred_mean=pred_mean,
                pred_std=pred_std
            )
            
            # Salvar no histórico
            history['time_min'].append(elapsed / 60)
            history['epoch'].append(epoch)
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)
            history['pred_mean'].append(pred_mean)
            history['pred_std'].append(pred_std)
            
            last_log_time = current_time
        
        epoch_time = time.time() - epoch_start
        
        # Pausa micro se época muito rápida (para não sobrecarregar)
        if epoch_time < 0.1:
            time.sleep(0.05)
    
    # Final
    total_time = time.time() - start_time
    
    # Salvar modelos finais
    output_dir = Path("./training_results_time_based")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    macro_path = str(output_dir / "macronet_final.pt")
    micro_path = str(output_dir / "micronet_final.pt")
    
    pipeline.macronet.save_model(macro_path)
    pipeline.micronet.save_model(micro_path)
    
    # Plotar gráfico de evolução
    plot_training_history(history, output_dir, total_time, epoch, best_loss)
    
    # Print final resumido
    print(f"\n✅ Treinamento completo: {epoch} épocas em {total_time/60:.1f}min")
    print(f"📉 Best loss: {best_loss:.4f}")
    print(f"💾 Modelos e gráficos salvos em: {output_dir}/")
    
    return {
        'epochs': epoch,
        'iterations': iteration,
        'total_time_min': total_time / 60,
        'best_loss': best_loss,
        'final_loss': loss,
        'final_accuracy': accuracy,
        'history': history
    }


def main():
    """Execução principal"""
    
    print("\n" + "="*70)
    print("  PIPELINE DE TREINAMENTO BASEADO EM TEMPO")
    print("  Contexto Longo → MacroNet → Embedding")
    print("  Contexto Curto + Embedding → MicroNet → Sinal")
    print("  Target: Tendência futura 5min (>0.2% = +1, <-0.2% = -1)")
    print("="*70 + "\n")
    
    # 1. Criar pipeline
    pipeline = TradingPipeline()
    
    # 2. Buscar dados
    print("📡 Baixando dados do Binance...")
    symbol = "BTCUSDT"
    days_back = 30
    
    long_data, short_data, full_df = pipeline.fetch_and_prepare_data(
        symbol,
        days_back=days_back
    )
    
    print(f"✅ Dados baixados: {len(full_df)} candles")
    print(f"   Período: {days_back} dias")
    print(f"   Timeframe: 5min")
    
    # 3. Preparar dados com targets simples
    print(f"\n{'─'*70}")
    X_short_list, X_long_list, y_list, num_features = prepare_simple_targets(full_df)
    
    # 4. Treinar por 10 minutos
    print(f"\n{'─'*70}")
    results = train_time_based(
        pipeline=pipeline,
        X_short_list=X_short_list,
        X_long_list=X_long_list,
        y_list=y_list,
        num_features=num_features,
        duration_minutes=10,
        log_interval_seconds=30,
        batch_size=32
    )
    
    # 5. Resumo final
    print(f"\n{'='*70}")
    print("  📊 RESUMO FINAL")
    print(f"{'='*70}")
    print(f"Épocas completadas:    {results['epochs']}")
    print(f"Iterações totais:      {results['iterations']}")
    print(f"Tempo de treinamento:  {results['total_time_min']:.2f} min")
    print(f"Loss inicial:          (não registrado)")
    print(f"Loss final:            {results['final_loss']:.6f}")
    print(f"Melhor loss:           {results['best_loss']:.6f}")
    print(f"Acurácia final:        {results['final_accuracy']*100:.1f}%")
    print(f"\n💡 Modelos prontos para uso!")
    print(f"   Use pipeline.predict_signal() para gerar sinais")


if __name__ == "__main__":
    main()
