import torch
import sys

def check_cuda():
    print(f"Python versie: {sys.version}")
    print(f"PyTorch versie: {torch.__version__}")
    print(f"CUDA beschikbaar: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA versie: {torch.version.cuda}")
        print(f"Aantal CUDA apparaten: {torch.cuda.device_count()}")
        print(f"Huidig CUDA apparaat: {torch.cuda.current_device()}")
        print(f"CUDA apparaat naam: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA met een eenvoudige operatie
        x = torch.rand(5, 3)
        print("\nTest tensor op CPU:")
        print(x)
        print("\nTest tensor op GPU:")
        x = x.cuda()
        print(x)
    else:
        print("\nCUDA is niet beschikbaar. Controleer je installatie.")

if __name__ == "__main__":
    check_cuda() 