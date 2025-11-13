#!/usr/bin/env python3
"""
Teste de diferentes abordagens para contornar o bug do PyTorch
"""

import numpy as np
import torch

print("=== Teste de workarounds para PyTorch ===")

# Create the problematic array
samples = np.random.randn(902, 48).astype(np.float32)
print(f"Array shape: {samples.shape}")

print("\n1. Teste torch.tensor() - DEVE FALHAR")
try:
    tensor1 = torch.tensor(samples, dtype=torch.float32)
    print("✅ torch.tensor() funcionou!")
except Exception as e:
    print(f"❌ torch.tensor() falhou: {e}")

print("\n2. Teste torch.from_numpy() - PODE FUNCIONAR")
try:
    tensor2 = torch.from_numpy(samples)
    print("✅ torch.from_numpy() funcionou!")
except Exception as e:
    print(f"❌ torch.from_numpy() falhou: {e}")

print("\n3. Teste com copy explicit")
try:
    samples_copy = samples.copy()
    tensor3 = torch.from_numpy(samples_copy)
    print("✅ from_numpy com copy funcionou!")
except Exception as e:
    print(f"❌ from_numpy com copy falhou: {e}")

print("\n4. Teste por chunks pequenos")
try:
    chunk_size = 100
    chunks = []
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i+chunk_size]
        tensor_chunk = torch.from_numpy(chunk)
        chunks.append(tensor_chunk)
    
    tensor4 = torch.cat(chunks, dim=0)
    print(f"✅ Chunks funcionou! Shape: {tensor4.shape}")
except Exception as e:
    print(f"❌ Chunks falhou: {e}")

print("\n5. Teste mudando dtype primeiro")
try:
    samples_float64 = samples.astype(np.float64)
    tensor5 = torch.from_numpy(samples_float64).float()
    print("✅ float64->float32 funcionou!")
except Exception as e:
    print(f"❌ float64->float32 falhou: {e}")

print("\n6. Teste com reshape")
try:
    samples_flat = samples.reshape(-1)
    tensor_flat = torch.from_numpy(samples_flat)
    tensor6 = tensor_flat.reshape(samples.shape)
    print(f"✅ Reshape funcionou! Shape: {tensor6.shape}")
except Exception as e:
    print(f"❌ Reshape falhou: {e}")

print("\n=== Testes concluídos ===")