import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sentence_transformers import SentenceTransformer
from financerag.retrieval import SentenceTransformerEncoder, DenseRetrieval
from financerag.tasks import FinDER
from financerag.rerank import CrossEncoderReranker
from financerag.common import CrossEncoder
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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

# We need to put prefix for e5 models.
# For more details, see Arxiv paper https://arxiv.org/abs/2212.03533
encoder_model = SentenceTransformerEncoder(
    model_name_or_path="intfloat/e5-large-v2",
    query_prompt="query: ",
    doc_prompt="passage: ",
    device=device,
)

retriever = DenseRetrieval(model=encoder_model)

# Retrieve relevant documents
results = finder_task.retrieve(retriever=retriever)
crossEncoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Rerank the results
reranker = CrossEncoderReranker(CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2"))
reranked_results = finder_task.rerank(reranker, results, top_k=100, batch_size=32)

# 검색 또는 재정렬 결과가 `results` 변수에 저장된 경우
# Recall, Precision, MAP, nDCG와 같은 다양한 지표로 모델 평가
# 평가 결과를 출력합니다 (즉, `Recall`, `Precision`, `MAP`, `nDCG`)
finder_task.evaluate(qrels_dict, results, [1, 5, 10])
finder_task.save_results(output_dir="results")
