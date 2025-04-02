import torch
print(torch.cuda.is_available())  # Nếu trả về False => PyTorch chưa hỗ trợ GPU
print(torch.cuda.device_count()) 