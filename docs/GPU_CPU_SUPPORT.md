# 🖥️ Suporte de Hardware - GPU/CPU/MPS

## 📋 Dispositivos Suportados

Este projeto suporta automaticamente:

| Dispositivo | Tipo | Performance | Status |
|------------|------|-------------|--------|
| **NVIDIA GPU** | CUDA | Máxima (10-30x CPU) | ✅ Suportado |
| **Intel GPU** | XPU (Arc/Iris Xe) | Muito Alta (5-15x CPU) | ✅ Suportado |
| **Apple Silicon** | MPS (M1/M2/M3/M4) | Alta (5-10x CPU) | ✅ Suportado |
| **Intel CPU** | CPU + VNNI | Básica | ✅ Suportado |
| **AMD CPU** | CPU + AVX2 | Básica | ✅ Suportado |
| **Apple CPU** | CPU | Básica | ✅ Suportado |

---

## 🚀 Detectar Hardware Disponível

### 1. Verificar Dispositivos

```bash
python check_device.py
```

**Resultado esperado:**

```
🔧 DIAGNÓSTICO DE HARDWARE - CPPNCRIPTO
════════════════════════════════════════════════════════════════

PYTORCH - VERIFICAÇÃO DE DISPOSITIVOS
════════════════════════════════════════════════════════════════

✓ PyTorch versão: 2.2.0

📌 CPU:
   • Disponível: Sim
   • Cores: 8

🔷 NVIDIA CUDA:
   ✓ Disponível: SIM
   • Versão CUDA: 12.1
   • cuDNN versão: 8804
   • GPUs detectadas: 1
     - 0: NVIDIA GeForce RTX 4090

🍎 APPLE METAL (M1/M2/M3/M4):
   ✗ Disponível: NÃO

💠 INTEL GPU (Arc, Iris Xe):
   ✗ Disponível: NÃO
```

### 2. Testar e Forçar Dispositivos

```bash
python test_device_override.py
```

**Menu interativo:**
```
📋 Menu:
  1. Testar CPU
  2. Testar CUDA (NVIDIA GPU)
  3. Testar MPS (Apple Metal)
  4. Testar XPU (Intel GPU)
  5. Benchmark - Comparar todos os devices
  6. Verificar Device Automático
  7. Sair
```

---

## 🔧 Configurar Device Específico

### Opção 1: Auto-detecção (Recomendado)

```python
from src.config import config

# Detecta automaticamente o melhor device
print(f"Device: {config.device}")  # cuda, mps, xpu ou cpu
```

### Opção 2: Forçar Device

```python
from src.config import config

# Forçar CPU
config.device = "cpu"

# Forçar CUDA (se disponível)
config.device = "cuda"

# Forçar MPS (Apple)
config.device = "mps"

# Forçar XPU (Intel)
config.device = "xpu"
```

### Opção 3: Variável de Ambiente

```bash
# Linux/Mac
export PYTORCH_DEVICE=cuda
export PYTORCH_DEVICE=mps
export PYTORCH_DEVICE=xpu

# Windows PowerShell
$env:PYTORCH_DEVICE="cuda"
```

---

## 📊 Performance Comparativa

### Treino de MacroNet (1 epoch, 1440 candles)

```
NVIDIA RTX 4090:    0.5 - 1.0 segundo
NVIDIA RTX 4080:    1.0 - 2.0 segundos
Intel Arc A770:     1.5 - 3.0 segundos
Apple M3 Max:       2.0 - 4.0 segundos
Apple M2:           3.0 - 6.0 segundos
Intel Iris Xe:      5.0 - 10.0 segundos
CPU (8-core):       10.0 - 30.0 segundos
CPU (4-core):       30.0 - 60.0 segundos
```

### Backtest (30 dias completos)

```
NVIDIA RTX 4090:    5 - 10 segundos
NVIDIA RTX 4080:    10 - 20 segundos
Intel Arc A770:     20 - 40 segundos
Apple M3 Max:       30 - 60 segundos
CPU (8-core):       2 - 5 minutos
CPU (4-core):       5 - 10 minutos
```

---

## 🔧 Instalação por Tipo de GPU

### NVIDIA GPU (CUDA)

**1. Verificar GPU:**
```bash
# Windows
nvidia-smi

# Linux/Mac
nvcc --version
```

**2. Instalar CUDA Toolkit:**
- Baixar: https://developer.nvidia.com/cuda-downloads
- Instalar seguindo instruções oficiais

**3. Instalar PyTorch com CUDA:**
```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**4. Verificar instalação:**
```python
import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # Nome da GPU
```

---

### Intel GPU (Arc/Iris Xe)

**1. Instalação no Windows:**
```bash
# Intel Arc GPUs (Windows)
pip install intel-extension-for-pytorch

# Depois instalar PyTorch normalmente
pip install -r requirements.txt
```

**2. Instalação no Linux:**
```bash
# Intel Extension for PyTorch
pip install intel-extension-for-pytorch

# Verificar
python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"
```

**3. Usar no código:**
```python
import torch
import intel_extension_for_pytorch as ipex

device = torch.device("xpu")

# Modelo no XPU
model = model.to(device)
model = ipex.optimize(model)
```

### Nota rápida - Instalação no Windows PowerShell

```powershell
# No PowerShell (recomendado):
pip install intel-extension-for-pytorch

# Depois, feche o terminal/IDE e reabra para que o Python carregue ipex corretamente
```

---

### Apple Silicon (M1/M2/M3/M4)

**1. Verificação automática:**
```bash
# PyTorch já tem suporte MPS nativo
python -c "import torch; print(torch.backends.mps.is_available())"
```

**2. Funciona automaticamente:**
```python
import torch

device = torch.device("mps")  # Ou deixar auto-detectar
model = model.to(device)
```

**3. Se não funcionar:**
```bash
# Atualizar PyTorch
pip install --upgrade torch

# Ou reinstalar especificamente para Mac
pip install --upgrade torch torchvision torchaudio
```

---

## 🧪 Testes de Verificação

### Teste Rápido

```bash
python quick_tests.py
```

### Teste com Device Específico

```python
from src.config import config
from src.pipeline import TradingPipeline

# Forçar device
config.device = "cuda"  # ou "mps", "xpu", "cpu"

# Testar
pipeline = TradingPipeline()
signal = pipeline.predict_signal("BTCUSDT")
print(f"Signal: {signal:.4f} (em {config.device})")
```

---

## ⚡ Otimizações por Device

### CUDA (NVIDIA)

```python
import torch

# Auto-tuning
torch.backends.cudnn.benchmark = True

# Usar float16 para melhor performance
model = model.half()
```

### MPS (Apple)

```python
import torch

# MPS é otimizado automaticamente
# Usar mixed precision
from torch.cuda.amp import GradScaler

scaler = GradScaler()
```

### XPU (Intel)

```python
import intel_extension_for_pytorch as ipex
import torch

# Otimizar modelo
model = ipex.optimize(model)

# Usar Automatic Mixed Precision
model.train()
```

---

## 🐛 Troubleshooting

### Problema: "CUDA not available" mas tenho GPU

**Solução:**
```bash
# 1. Verificar driver NVIDIA
nvidia-smi

# 2. Reinstalar CUDA Toolkit
# Baixe de: https://developer.nvidia.com/cuda-downloads

# 3. Reinstalar PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Problema: "MPS not available" no Mac

**Solução:**
```bash
# 1. Atualizar PyTorch
pip install --upgrade torch

# 2. Usar CPU por enquanto
# Vai funcionar, mas mais lento
```

### Problema: "XPU not found" no Intel

**Solução:**
```bash
# 1. Instalar Intel Extension
pip install intel-extension-for-pytorch

# 2. Verificar instalação
python -c "import intel_extension_for_pytorch; print('OK')"

# 3. Se não funcionar, usar CPU
config.device = "cpu"
```

### Problema: Out of Memory (OOM)

**Solução:**
```python
# Reduzir batch size
config.macronet.batch_size = 8  # Ao invés de 32

# Usar menor embedding_dim
config.macronet.embedding_dim = 64  # Ao invés de 128

# Usar CPU ao invés de GPU
config.device = "cpu"
```

---

## 📈 Monitoramento de Performance

### Durante Treino

```python
import torch
from src.pipeline import TradingPipeline

pipeline = TradingPipeline()

# Monitor GPU (NVIDIA)
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Monitor CPU
import psutil
memory = psutil.virtual_memory()
print(f"RAM: {memory.percent}%")
```

### Logs de Performance

```python
from src.logger import get_logger

logger = get_logger(__name__)

logger.info("training_start", device=config.device)
# ... treinamento ...
logger.info("training_end", device=config.device, time_seconds=elapsed)
```

---

## 🎯 Recomendações

| Situação | Recomendado |
|----------|------------|
| Desenvolvimento local | CPU ou MPS (Mac) |
| Produção pequena | Intel GPU ou CPU |
| Produção média | NVIDIA RTX 4080 |
| Produção grande | NVIDIA RTX 4090 ou A100 |
| Laptop Mac | MPS (automático) |
| Laptop Intel | CPU ou Intel Arc (se tiver) |
| Servidor Linux | NVIDIA CUDA |

---

**Última atualização:** Novembro 2025
