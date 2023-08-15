import torch
from torch.autograd import Variable

a = [[1, 2, 3], [4, 5, 6]]
b = [[0, 1, 2], [5, 6, 7]]

a1 = Variable((torch.Tensor(a)))
a2 = Variable((torch.Tensor(a)))

print(torch.mean(a1, dim=0))