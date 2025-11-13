"""
Script para detectar e diagnosticar dispositivos de computação disponíveis
Executável: python check_device.py
"""
import sys
sys.path.insert(0, '.')

def check_pytorch():
    """Verificar PyTorch e dispositivos disponíveis"""
    print("\n" + "="*70)
    print("  PYTORCH - VERIFICAÇÃO DE DISPOSITIVOS")
    print("="*70 + "\n")
    
    try:
        import torch
        print(f"✓ PyTorch versão: {torch.__version__}")
        
        # CPU
        print(f"\n📌 CPU:")
        print(f"   • Disponível: Sim")
        import multiprocessing
        print(f"   • Cores: {multiprocessing.cpu_count()}")
        
        # CUDA (NVIDIA GPU)
        print(f"\n🔷 NVIDIA CUDA:")
        if torch.cuda.is_available():
            print(f"   ✓ Disponível: SIM")
            print(f"   • Versão CUDA: {torch.version.cuda}")
            print(f"   • cuDNN versão: {torch.backends.cudnn.version()}")
            print(f"   • GPUs detectadas: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"     - {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"   ✗ Disponível: NÃO")
        
        # MPS (Apple Metal - M1/M2/M3)
        print(f"\n🍎 APPLE METAL (M1/M2/M3/M4):")
        if hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                print(f"   ✓ Disponível: SIM")
                print(f"   • Metal Performance Shaders habilitado")
            else:
                print(f"   ✗ Disponível: NÃO")
                if torch.backends.mps.is_built():
                    print(f"   • MPS construído, mas não disponível neste sistema")
        else:
            print(f"   ✗ Disponível: NÃO (PyTorch sem suporte MPS)")
        
        # XPU (Intel GPU - Arc, Iris Xe)
            print(f"\n💠 INTEL GPU (Arc, Iris Xe):")
            xpu_supported = False
            try:
                # Check intel extension for pytorch
                import importlib
                if importlib.util.find_spec('intel_extension_for_pytorch') is not None:
                    # ipex installed - report presence and recommend usage
                    print(f"   ✓ ipex (intel-extension-for-pytorch) detectado")
                    xpu_supported = True
                else:
                    # fallback to torch.backends.xpu if available
                    if hasattr(torch.backends, 'xpu') and getattr(torch.backends.xpu, 'is_available', lambda: False)():
                        print(f"   ✓ Disponível: SIM (torch.backends.xpu)")
                        xpu_supported = True
                    else:
                        print(f"   ✗ Disponível: NÃO")
            except Exception:
                print(f"   ✗ Erro ao checar XPU")

            if not xpu_supported:
                print(f"   • Para usar GPU Intel, instale: pip install intel-extension-for-pytorch")
                print(f"   • Depois, reinicie o interpretador/terminal (feche e reabra a sessão Python)")
        
    except ImportError:
        print("✗ PyTorch não instalado!")
        print("  Instale: pip install torch")


def check_cpu_capabilities():
    """Verificar capacidades da CPU"""
    print("\n" + "="*70)
    print("  CPU - CAPACIDADES E INSTRUÇÕES")
    print("="*70 + "\n")
    
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        
        print(f"Marca: {info.get('brand_raw', 'Desconhecida')}")
        print(f"Modelo: {info.get('hz_advertised_friendly', 'Desconhecido')}")
        
        # Flags de instrução
        print(f"\nInstruções suportadas:")
        flags = info.get('flags', [])
        
        # Agrupar flags importantes
        simd_flags = [f for f in flags if any(x in f for x in ['sse', 'avx', 'neon', 'sve'])]
        if simd_flags:
            print(f"   • SIMD: {', '.join(simd_flags)}")
        
        gpu_flags = [f for f in flags if any(x in f for x in ['gpu', 'igpu'])]
        if gpu_flags:
            print(f"   • GPU integrada: {', '.join(gpu_flags)}")
        
        ai_flags = [f for f in flags if any(x in f for x in ['vnni', 'amx', 'bf16'])]
        if ai_flags:
            print(f"   • AI aceleração: {', '.join(ai_flags)}")
        
    except ImportError:
        print("ℹ️  cpuinfo não instalado")
        print("   Instale: pip install py-cpuinfo")


def check_current_config():
    """Verificar configuração atual do projeto"""
    print("\n" + "="*70)
    print("  PROJETO - CONFIGURAÇÃO ATUAL")
    print("="*70 + "\n")
    
    from src.config import config
    
    print(f"📌 Device detectado: {config.device.upper()}")
    
    # Dar recomendações
    print(f"\n💡 Recomendações:")
    
    if config.device == "cuda":
        print(f"   ✓ GPU NVIDIA detectada - Performance MÁXIMA")
        print(f"   • Modelos rodando em paralelo na GPU")
        print(f"   • Treino ~10-20x mais rápido que CPU")
    
    elif config.device == "mps":
        print(f"   ✓ Apple Metal detectado - Performance ALTA")
        print(f"   • Otimizado para M1/M2/M3/M4")
        print(f"   • Treino ~5-10x mais rápido que CPU")
        print(f"   • Melhor que CPU integrada Intel")
    
    elif config.device == "xpu":
        print(f"   ✓ Intel GPU detectada - Performance ÓTIMA")
        print(f"   • GPU Arc ou Iris Xe habilitada")
        print(f"   • Treino ~8-15x mais rápido que CPU")
    
    else:  # CPU
        print(f"   ⚠️  CPU detectada - Performance BÁSICA")
        print(f"   • Usando apenas CPU")
        print(f"   • Treino mais lento, mas funciona")
        print(f"   • Recomendações:")
        print(f"     - Se tem NVIDIA: instale CUDA Toolkit")
        print(f"     - Se tem Intel Arc: pip install intel-extension-for-pytorch")
        print(f"     - Se tem M1/M2/M3: MPS já ativado via PyTorch")


def compare_performance():
    """Comparar performance em diferentes dispositivos"""
    print("\n" + "="*70)
    print("  COMPARAÇÃO DE PERFORMANCE (Tempo estimado por epoch)")
    print("="*70 + "\n")
    
    data = {
        "CPU": "~10-30 segundos",
        "CPU Intel (VNNI)": "~5-10 segundos",
        "Apple M1/M2/M3": "~2-5 segundos",
        "Intel GPU (Arc)": "~1-3 segundos",
        "NVIDIA RTX 4080": "~0.5-1 segundo"
    }
    
    print("Treino de MacroNet (1440 candles × 13 features):\n")
    for device, time_est in data.items():
        print(f"  • {device:.<30} {time_est}")
    
    print("\n📊 Backtest (30 dias de trading):\n")
    
    data_backtest = {
        "CPU": "~2-5 minutos",
        "CPU Intel (VNNI)": "~1-2 minutos",
        "Apple M1/M2/M3": "~30-60 segundos",
        "Intel GPU (Arc)": "~10-30 segundos",
        "NVIDIA RTX 4080": "~5-10 segundos"
    }
    
    for device, time_est in data_backtest.items():
        print(f"  • {device:.<30} {time_est}")


def main():
    """Executar diagnóstico completo"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🔧 DIAGNÓSTICO DE HARDWARE - CPPNCRIPTO".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    check_pytorch()
    check_cpu_capabilities()
    check_current_config()
    compare_performance()
    
    print("\n" + "="*70)
    print("  INSTRUÇÕES DE INSTALAÇÃO")
    print("="*70 + "\n")
    
    print("Para usar GPU NVIDIA (CUDA):")
    print("  1. Instale CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    print("  2. Instale cuDNN: https://developer.nvidia.com/cudnn")
    print("  3. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\nPara usar GPU Intel (Arc/Iris Xe):")
    print("  1. Instale Intel Extension for PyTorch")
    print("  2. pip install intel-extension-for-pytorch")
    print("  3. Reinicie o kernel/terminal Python (feche e reabra o interpretador) ")
    
    print("\nPara usar Apple Metal (M1/M2/M3):")
    print("  1. PyTorch já tem suporte nativo via MPS")
    print("  2. Deve funcionar automaticamente no Apple Silicon")
    
    print("\n✅ Diagnóstico completo!\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
