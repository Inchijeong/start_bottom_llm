import torch

tensor2d = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])
print(tensor2d)
print(tensor2d.shape)

t2 =tensor2d.reshape(3, 2)

print(t2)
print(t2.shape)

t3 = tensor2d.view(3, 2)
print(t3)
print(t3.shape)

t3 = tensor2d.T
print(t3)

t4 = tensor2d.matmul(tensor2d.T)
print(t4)

t5 = tensor2d @ tensor2d.T
print(t5)


