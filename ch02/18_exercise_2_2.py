from create_dataloder_v1 import create_dataloader_v1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

def print_two_batches(dataloader):
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    second_batch = next(data_iter)
    print(second_batch)

# 사용 예시
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=2, stride=2, shuffle=False)
print_two_batches(dataloader)

dataloader2 = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=2, shuffle=False)
# print_two_batches(dataloader2)

dataloader3 = create_dataloader_v1(raw_text, batch_size=2, max_length=3, stride=1, shuffle=False)
# print_two_batches(dataloader3)