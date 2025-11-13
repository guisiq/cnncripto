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
            
            last_log_time = current_time
        
        epoch_time = time.time() - epoch_start
        
        # Pausa micro se época muito rápida (para não sobrecarregar)
        if epoch_time < 0.1:
            time.sleep(0.05)
    
    # Final
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"  ✅ TREINAMENTO COMPLETO!")
    print(f"{'='*70}")
    print(f"⏱️  Tempo total: {total_time/60:.2f} minutos")
    print(f"📈 Épocas completadas: {epoch}")
    print(f"📉 Melhor loss: {best_loss:.6f}")
    print(f"💾 Modelos salvos automaticamente")
    
    # Salvar modelos finais
    output_dir = Path("./training_results_time_based")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    macro_path = str(output_dir / "macronet_final.pt")
    micro_path = str(output_dir / "micronet_final.pt")
    
    pipeline.macronet.save_model(macro_path)
    pipeline.micronet.save_model(micro_path)
    
    print(f"\n📁 Modelos salvos:")
    print(f"   MacroNet: {macro_path}")
    print(f"   MicroNet: {micro_path}")
    
    return {
        'epochs': epoch,
        'iterations': iteration,
        'total_time_min': total_time / 60,
        'best_loss': best_loss,
        'final_loss': loss,
        'final_accuracy': accuracy
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
