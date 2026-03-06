

import torch
print(f"WSL 是否识别 GPU: {torch.cuda.is_available()}")
print(f"显卡型号: {torch.cuda.get_device_name(0)}")
print(f"计算能力: {torch.cuda.get_device_capability(0)}") 
# 期待输出: (12, 0)
exit()