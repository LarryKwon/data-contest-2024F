import pandas as pd
from financerag.tasks import FinDER

# FinDER 작업 초기화
finder_task = FinDER()

# 답변 레이블의 30%가 포함된 TSV 파일 로드
df = pd.read_csv(".files/FinDER_qrels.tsv", sep="\t")

# TSV 데이터를 평가를 위한 사전 형식으로 변환
qrels_dict = (
    df.groupby("query_id")
    .apply(lambda x: dict(zip(x["corpus_id"], x["score"])))
    .to_dict()
)

# 검색 또는 재정렬 결과가 `results` 변수에 저장된 경우
# Recall, Precision, MAP, nDCG와 같은 다양한 지표로 모델 평가
finder_task.evaluate(qrels_dict, results, [1, 5, 10])

# 평가 결과를 출력합니다 (즉, `Recall`, `Precision`, `MAP`, `nDCG`)
