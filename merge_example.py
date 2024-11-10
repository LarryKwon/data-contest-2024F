import os
import csv
from glob import glob

# 결과 파일 경로 설정
output_dir = "output"
merged_file_path = os.path.join(output_dir, "merged_results.csv")

# 첫 번째 파일에서만 헤더를 쓰기 위한 플래그
write_header = True

with open(merged_file_path, mode="w", newline="") as merged_file:
    writer = csv.writer(merged_file)

    # output 폴더 내 모든 query_corpus_matches.csv 파일 찾기
    csv_files = glob(os.path.join(output_dir, "*", "*", "results.csv"))

    for file_path in csv_files:
        with open(file_path, mode="r", newline="") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)  # 첫 번째 줄을 헤더로 처리

            # 첫 번째 파일에만 헤더를 쓰고, 나머지는 데이터만 추가
            if write_header:
                writer.writerow(header)
                write_header = False

            for row in reader:
                writer.writerow(row)

print(f"All query_corpus_matches.csv files have been merged into {merged_file_path}")
