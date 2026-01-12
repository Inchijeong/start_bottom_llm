from create_dataloder_v1 import create_dataloader_v1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# stride: 슬라이딩 윈도가 다음 배치로 이동할때 변경 크기
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

dataloader2 = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter2 = iter(dataloader2)
inputs, targets = next(data_iter2)
print("입력:\n", inputs)
print("\n타깃:\n", targets)