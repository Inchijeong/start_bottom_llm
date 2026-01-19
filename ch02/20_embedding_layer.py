import torch

from create_dataloder_v1 import create_dataloader_v1
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("토큰 ID:\n", inputs)
print("\n입력 크기:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("\n토큰 임베딩 크기:\n", token_embeddings.shape)

# 임베딩 벡터의 값을 확인합니다.
print("\n토큰 임베딩 벡터의 값:\n",token_embeddings)

context_length = max_length
# context_length = 4
# output_dim: 256
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# 임베딩 층의 가중치를 확인합니다.
print("\n위치 임베딩 층의 가중치:\n", pos_embedding_layer.weight)

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

# 위치 임베딩 값을 확인합니다.
print("\n위치 임베딩 값:\n", pos_embeddings)

input_embeddings = token_embeddings + pos_embeddings
print("\n입력 임베딩의 크기:\n", input_embeddings.shape)

# 입력 임베딩 값을 확인합니다.
print("\n입력 임베딩:\n",input_embeddings)