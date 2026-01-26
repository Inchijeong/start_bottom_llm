import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]  # 두 번째 입력 토큰이 쿼리입니다

res = 0.

for idx, element in enumerate(inputs[0]):
    product = inputs[0][idx] * query[idx]
    print('product: ', product)
    res += product

print(res)
print(torch.dot(inputs[0], query))