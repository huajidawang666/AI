import torch

def check_CUDA_available():
    availability = torch.cuda.is_available()
    print(f'Checking CUDA availability: {availability}')
    if availability:
        print(f'Checking CUDA version:{torch.version.cuda}')
        print(f"Current using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    
    return torch.device('cuda' if availability else 'cpu')