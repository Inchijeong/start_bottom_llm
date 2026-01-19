import torch

# vocab_size=10, output_dim=2, context_length=3, batch_size=2
token_embedding_layer = torch.nn.Embedding(10, 2)
pos_embedding_layer = torch.nn.Embedding(3, 2)

# 임의의 토큰 ID 입력 (batch_size=2, max_length=3)
inputs = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])

token_embeddings = token_embedding_layer(inputs)
# token_embeddings shape: (2, 3, 2)

pos_embeddings = pos_embedding_layer(torch.arange(3))
# pos_embeddings shape: (3, 2)

# 브로드캐스팅되어 더해짐
input_embeddings = token_embeddings + pos_embeddings

print("token_embeddings:\n", token_embeddings)
print("pos_embeddings:\n", pos_embeddings)
print("input_embeddings:\n", input_embeddings)
