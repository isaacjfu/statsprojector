import torch
import numpy as np

x1 = [[1,2],[3,4]]
x1_data = torch.tensor(x1)
x2 = [[5,6],[7,8]]
x2_data = torch.tensor(x2)

y1 = x1_data @ x2_data

print(y1)
