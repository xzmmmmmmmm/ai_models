import torch
print(f"WSL 是否识别 GPU: {torch.cuda.is_available()}")
print(f"显卡型号: {torch.cuda.get_device_name(0)}")
print(f"算力版本: {torch.cuda.get_device_capability(0)}") 
# RTX 5060 应该显示 (12, 0) 左右