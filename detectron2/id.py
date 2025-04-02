import torch

pth_path = "2023-09-20_1407_cucumber_300000.pth"

# 파일 로드
checkpoint = torch.load(pth_path, map_location='cpu')

# 타입 확인
print("Checkpoint type:", type(checkpoint))

# 키 출력
if isinstance(checkpoint, dict):
    print("Checkpoint keys:")
    for k in checkpoint.keys():
        print(" -", k)

# state_dict 내용 확인
state_dict = checkpoint.get('model', checkpoint)
print("\n[State Dict Sample Keys]")
for i, key in enumerate(state_dict.keys()):
    print(f"{i+1}: {key}")
    if i >= 19:
        break
