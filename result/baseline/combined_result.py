import pandas as pd
import glob
import os

# CSV 파일들이 저장된 폴더 경로
folder_path = "./baseline"  # 여기에 폴더 경로 입력
tasks = [
    "FinDER",
    "FinQABench",
    "FinanceBench",
    "TATQA",
    "FinQA",
    "ConvFinQA",
    "MultiHiertt",
]

# 각 task 폴더 내의 모든 CSV 파일 경로 가져오기
csv_files = []
for task in tasks:
    print(os.path.join("./", task, "*.csv"))
    print(glob.glob(os.path.join("./", task, "*.csv")))
    csv_files.extend(glob.glob(os.path.join(task, "*.csv")))

# 각 CSV 파일을 읽어 데이터프레임으로 변환하여 리스트에 저장
print(csv_files)
dataframes = [pd.read_csv(file) for file in csv_files]

# 모든 데이터프레임을 하나로 합치기
combined_df = pd.concat(dataframes, ignore_index=True)

# 결과를 새로운 CSV 파일로 저장
combined_df.to_csv("combined_output.csv", index=False)
