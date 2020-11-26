"""

batch_size * seq_len * embedding_size
batch_size * seq_len * num_item
batch_size * seq_len
"""

# import torch
# import torch.nn as nn
# group = torch.chunk(torch.arange(10), 10)
# print(group)

# import numpy as np
# a=np.arange(5)

import torch
arrange_tensor=torch.randint(1,10,size=(5,2))
x=arrange_tensor.numpy()
print(x)
for idx,y in enumerate(x):
    x[idx]=y[::-1]
print(x)



# x=nn.Linear(3,2)
# y=nn.Embedding(3,2)
# print(x.weight.size())
# print(y.weight.size())
# matrix_01 = torch.zeros((1,1)).repeat_interleave(2,2)
# print(matrix_01.flatten())

# print(torch.arange(5))