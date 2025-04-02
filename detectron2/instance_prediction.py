import torch

# 🔹 저장된 파일 경로
pth_file = "inference_result/instances_predictions.pth"

# 🔹 파일 로드
data = torch.load(pth_file, map_location="cpu")

# 🔹 데이터 타입 확인
print(f"파일 로드 완료! 데이터 타입: {type(data)}")

# 🔹 딕셔너리 키 확인
if isinstance(data, dict):
    print(f"🔑 포함된 키들: {data.keys()}")

# 🔹 일부 내용 출력
if isinstance(data, list):
    print(f"📌 리스트 길이: {len(data)}")
    print(f"첫 번째 요소 예시: {data[0]}")
elif isinstance(data, dict):
    for key in data.keys():
        print(f"📂 {key}: {type(data[key])}")
        if isinstance(data[key], list) and len(data[key]) > 0:
            print(f"  🔹 첫 번째 값 예시: {data[key][0]}")
