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

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # 점곱 (1차원 벡터이므로 전치가 필요 없습니다)

print('attn_scores_2: ', attn_scores_2)
print('attn_scores_2.sum(): ', attn_scores_2.sum())

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("어텐션 가중치:", attn_weights_2_tmp)
print("합:", attn_weights_2_tmp.sum())

# attn_scores_2 / attn_scores_2.sum()는 attn_scores_2 텐서의
# 각 원소를 전체 합(attn_scores_2.sum())으로 나누는 연산입니다.
# 즉, 각 점수 값을 전체 합으로 나눠서 모든 값의 합이 1이 되도록 정규화합니다.
# 이렇게 하면 각 값이 전체에서 차지하는 비율(확률처럼 동작)을 나타내는 어텐션 가중치가 됩니다.
#
# attn_scores_2 = [2, 3, 5]
# attn_scores_2.sum() = 10
# attn_weights_2_tmp = [2/10, 3/10, 5/10] = [0.2, 0.3, 0.5]
