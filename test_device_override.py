"""
Script para testar e forçar uso de diferentes devices (CPU/GPU/MPS)
Executável: python test_device_override.py
"""
import sys
sys.path.insert(0, '.')

import os
import numpy as np

def test_device(device_name):
    """Testar um dispositivo específico"""
    print(f"\n{'='*70}")
    print(f"  Testando: {device_name.upper()}")
    print(f"{'='*70}\n")
    
    try:
        import torch
        
        # Forçar usar o device especificado
        if device_name == "cpu":
            device = torch.device("cpu")
            print(f"✓ CPU selecionado")
        
        elif device_name == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"✓ CUDA (NVIDIA GPU) selecionado")
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"✗ CUDA não disponível neste sistema")
                return False
        
        elif device_name == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                print(f"✓ MPS (Apple Metal) selecionado")
            else:
                print(f"✗ MPS não disponível neste sistema")
                return False
        
        elif device_name == "xpu":
            if hasattr(torch.backends, "xpu") and torch.backends.xpu.is_available():
                device = torch.device("xpu")
                print(f"✓ XPU (Intel GPU) selecionado")
            else:
                print(f"✗ XPU não disponível. Instale: pip install intel-extension-for-pytorch")
                return False
        
        else:
            print(f"✗ Device desconhecido: {device_name}")
            return False
        
        # Teste 1: Criar tensor
        print(f"\n1️⃣  Criando tensor...")
        x = torch.randn(1000, 1000, device=device)
        print(f"   ✓ Tensor criado em {device}")
        
        # Teste 2: Operação matemática
        print(f"\n2️⃣  Operação matemática (matriz × matriz)...")
        import time
        start = time.time()
        y = torch.mm(x, x.t())
        elapsed = time.time() - start
        print(f"   ✓ Concluído em {elapsed*1000:.2f}ms")
        
        # Teste 3: Rede neural simples
        print(f"\n3️⃣  Teste de rede neural...")
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 64, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1, device=device)
        )
        
        input_data = torch.randn(100, 100, device=device)
        start = time.time()
        output = model(input_data)
        elapsed = time.time() - start
        print(f"   ✓ Forward pass concluído em {elapsed*1000:.2f}ms")
        
        # Teste 4: Backprop
        print(f"\n4️⃣  Teste de backpropagation...")
        loss_fn = torch.nn.MSELoss()
        target = torch.randn(100, 1, device=device)
        loss = loss_fn(output, target)
        
        start = time.time()
        loss.backward()
        elapsed = time.time() - start
        print(f"   ✓ Backward pass concluído em {elapsed*1000:.2f}ms")
        
        print(f"\n✅ {device_name.upper()} funcionando perfeitamente!")
        return True
        
    except Exception as e:
        print(f"✗ Erro ao testar {device_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def force_device_in_config(device_name):
    """Forçar uso de um device específico na configuração"""
    print(f"\n{'='*70}")
    print(f"  Forçando Device: {device_name.upper()}")
    print(f"{'='*70}\n")
    
    from src.config import config
    
    # Forçar o device
    config.device = device_name
    print(f"✓ Device configurado para: {config.device}")
    print(f"  Todos os modelos usarão: {config.device}")
    
    return config


def benchmark_devices():
    """Comparar performance entre devices disponíveis"""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK - Comparar Devices")
    print(f"{'='*70}\n")
    
    import torch
    import time
    
    devices_to_test = []
    
    # Verificar quais estão disponíveis
    if True:
        devices_to_test.append("cpu")
    
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices_to_test.append("mps")
    
    if hasattr(torch.backends, "xpu") and torch.backends.xpu.is_available():
        devices_to_test.append("xpu")
    
    print(f"Testando: {', '.join(devices_to_test)}\n")
    
    results = {}
    
    for device_name in devices_to_test:
        try:
            device = torch.device(device_name)
            
            # Forward pass
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            start = time.time()
            for _ in range(100):
                z = torch.mm(x, y)
            forward_time = (time.time() - start) / 100
            
            # Backward pass
            x.requires_grad = True
            y.requires_grad = True
            
            start = time.time()
            for _ in range(50):
                z = torch.mm(x, y)
                z.sum().backward()
                x.grad.zero_()
                y.grad.zero_()
            backward_time = (time.time() - start) / 50
            
            results[device_name] = {
                'forward': forward_time * 1000,  # ms
                'backward': backward_time * 1000
            }
            
        except Exception as e:
            print(f"  ✗ Erro testando {device_name}: {e}")
    
    # Exibir resultados
    print("📊 Resultados (tempo em ms):\n")
    print(f"{'Device':<15} {'Forward Pass':<15} {'Backward Pass':<15}")
    print("-" * 45)
    
    for device, times in results.items():
        print(f"{device:<15} {times['forward']:<15.4f} {times['backward']:<15.4f}")
    
    # Finder relativo
    if 'cpu' in results:
        cpu_forward = results['cpu']['forward']
        print(f"\nSpeedup relativo à CPU:")
        
        for device, times in results.items():
            if device != 'cpu':
                speedup = cpu_forward / times['forward']
                print(f"  {device}: {speedup:.1f}x mais rápido")


def main():
    """Menu principal"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🔧 TEST DEVICE - Testar e Forçar GPU/CPU".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print("\n📋 Menu:")
    print("  1. Testar CPU")
    print("  2. Testar CUDA (NVIDIA GPU)")
    print("  3. Testar MPS (Apple Metal)")
    print("  4. Testar XPU (Intel GPU)")
    print("  5. Benchmark - Comparar todos os devices")
    print("  6. Verificar Device Automático")
    print("  7. Sair")
    
    while True:
        choice = input("\n👉 Escolha (1-7): ").strip()
        
        if choice == '1':
            test_device("cpu")
        elif choice == '2':
            test_device("cuda")
        elif choice == '3':
            test_device("mps")
        elif choice == '4':
            test_device("xpu")
        elif choice == '5':
            benchmark_devices()
        elif choice == '6':
            from src.config import config
            print(f"\nDevice detectado automaticamente: {config.device.upper()}")
        elif choice == '7':
            print("\n👋 Até logo!")
            break
        else:
            print("❌ Opção inválida!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrompido")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
