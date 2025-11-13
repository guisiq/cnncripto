#!/usr/bin/env python3
"""
Teste simples para verificar se o problema está na importação ou inicialização
"""

print("=== Teste de importações básicas ===")

try:
    import torch
    print(f"✅ PyTorch importado: {torch.__version__}")
except Exception as e:
    print(f"❌ Erro ao importar PyTorch: {e}")
    exit(1)

try:
    import numpy as np
    print(f"✅ NumPy importado: {np.__version__}")
except Exception as e:
    print(f"❌ Erro ao importar NumPy: {e}")
    exit(1)

print("\n=== Teste de detecção de device ===")

# Force CPU
print("Forçando device = 'cpu'")
device = torch.device("cpu")
print(f"Device: {device}")

print("\n=== Teste de tensor simples ===")
try:
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(f"✅ Tensor criado: {x}")
except Exception as e:
    print(f"❌ Erro ao criar tensor: {e}")
    exit(1)

print("\n=== Teste de importação das classes do projeto ===")

try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from src.config import config
    print(f"✅ Config importado")
except Exception as e:
    print(f"❌ Erro ao importar config: {e}")
    print("Tentando configuração manual...")
    
print("\n=== Teste de importações específicas ===")

try:
    from src.ingest.binance import BinanceIngestor
    print(f"✅ BinanceIngestor importado")
except Exception as e:
    print(f"❌ Erro ao importar BinanceIngestor: {e}")

try:
    from src.features.builder import FeatureBuilder
    print(f"✅ FeatureBuilder importado")
except Exception as e:
    print(f"❌ Erro ao importar FeatureBuilder: {e}")

print("\n=== Todos os testes básicos concluídos ===")