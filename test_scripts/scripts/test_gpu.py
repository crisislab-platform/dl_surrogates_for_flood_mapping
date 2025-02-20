import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"PyTorch is using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. PyTorch is using CPU.")
