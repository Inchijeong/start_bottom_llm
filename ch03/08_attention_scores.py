import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)

print('inputs:', inputs)

query = inputs[1]  # 두 번째 입력 토큰이 쿼리입니다

# attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)

# for문 대체 방법: 행렬 곱셈
attn_scores = inputs @ inputs.T # [6,3]@[3,6]=[6,6]

print('attn_scores: ', attn_scores)

attn_weights = torch.softmax(attn_scores, dim=1)

print("어텐션 가중치:", attn_weights)
print("합:", attn_weights.sum())

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("두 번째 행의 합:", row_2_sum)

print("모든 행의 합:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs # [6,6]@[6,3] = [6,3]
print('all_context_vecs:', all_context_vecs)