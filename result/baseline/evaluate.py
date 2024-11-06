import pandas as pd
import sys
import os
import glob

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sentence_transformers import SentenceTransformer

# from financerag.retrieval import SentenceTransformerEncoder, DenseRetrieval
from financerag.tasks import BaseTask, FinDER

# from financerag.rerank import CrossEncoderReranker
# from financerag.common import CrossEncoder


finder_task = FinDER()

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
    print(os.path.join("./result/baseline/", task, "*.csv"))
    file = os.path.join("./result/baseline/", task, "results.csv")
    dataframes = pd.read_csv(file)
    results_dict = dict(zip(dataframes["query_id"], dataframes["corpus_id"]))
    # TSV 데이터를 평가를 위한 사전 형식으로 변환
    df = pd.read_csv(f".files/{task}_qrels.tsv", sep="\t")
    qrels_dict = (
        df.groupby("query_id")
        .apply(lambda x: dict(zip(x["corpus_id"], x["score"])))
        .to_dict()
    )
    print(results_dict)
    finder_task.evaluate(qrels_dict, results_dict, [1, 5, 10])
