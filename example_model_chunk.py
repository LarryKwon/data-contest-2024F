from sentence_transformers import SentenceTransformer, CrossEncoder
from financerag.retrieval import SentenceTransformerEncoder, DenseRetrieval
from financerag.tasks import (
    FinDER,
    FinQABench,
    FinanceBench,
    TATQA,
    FinQA,
    ConvFinQA,
    MultiHiertt,
)
from financerag.rerank import CrossEncoderReranker
import torch
import os
from nltk.tokenize import sent_tokenize
import pandas as pd


def split_text_into_chunks(text, chunk_size, overlap_size):
    sentences = sent_tokenize(text, language="english")

    chunks = []
    start = 0
    while start < len(sentences):
        end = start + chunk_size
        chunk_sentences = sentences[start:end]
        chunk_text = " ".join(chunk_sentences)
        chunks.append(chunk_text)
        start += chunk_size - overlap_size  # 오버래핑 처리
    return chunks


# 결과 파일을 불러와서 corpus_id를 변환하는 함수
def convert_corpus_id_to_original(results_path, doc_id_mapping):
    # CSV 파일 읽기
    df = pd.read_csv(results_path)
    # print(df)
    # corpus_id를 원래 ID로 매핑
    df["corpus_id"] = df["corpus_id"].astype(str).map(doc_id_mapping)

    # print(df)
    # 변환된 결과를 다시 CSV 파일로 저장
    df.to_csv(results_path, index=False)

    print(f"'{results_path}' 파일의 corpus_id가 원래 ID로 변환되었습니다.")


def process_results(results_path):
    # CSV 파일 읽기
    df = pd.read_csv(results_path)

    # 중복 제거 및 상위 10개만 남기기 위한 처리
    processed_df = df.groupby(
        "query_id", group_keys=False
    ).apply(  # query_id별로 그룹화
        lambda x: x.drop_duplicates(subset="corpus_id").head(10)
    )  # corpus_id 중복 제거 후 상위 10개만 남김

    # 처리된 결과를 다시 CSV 파일로 저장
    processed_df.to_csv(results_path, index=False)
    print(f"'{results_path}' 파일이 중복 제거 및 상위 10개 항목으로 필터링되었습니다.")


# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 중인 디바이스: {device}")

# 모델 경로
model_name = "intfloat/e5-large-v2"
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# 각 task를 위한 클래스와 결과 저장 경로 정의
tasks = {
    "FinDER": FinDER(),
    "FinQABench": FinQABench(),
    "FinanceBench": FinanceBench(),
    "TATQA": TATQA(),
    "FinQA": FinQA(),
    "ConvFinQA": ConvFinQA(),
    "MultiHiertt": MultiHiertt(),
}

# 모든 Task에 대해 반복 수행
for task_name, task_instance in tasks.items():
    print(f"\n=== {task_name} Task 처리 시작 ===")

    # 청크 나누기
    # 기본값 설정
    chunk_size = 3
    overlap_size = 1

    corpus = list(task_instance.corpus.items())
    new_corpus = []
    for doc_id, doc_content in corpus:
        text = doc_content["text"]
        title = doc_content.get("title", "")

        # 텍스트를 청크로 분할
        chunks = split_text_into_chunks(text, chunk_size, overlap_size)

        # 청크별로 새로운 코퍼스 생성
        for idx, chunk_text in enumerate(chunks):
            new_doc_id = f"{doc_id}_chunk{idx}"
            new_corpus.append(
                (doc_id, {"title": title, "text": chunk_text})  # 부모 아이디 유지
            )

    # 원래 문서 ID를 매핑할 딕셔너리 생성
    doc_id_mapping = {}

    # 새 딕셔너리에 저장할 데이터를 담을 딕셔너리
    new_corpus_dict = {}

    index = 1
    # new_corpus 리스트를 순회하며 새로운 딕셔너리에 값 추가
    for original_doc_id, doc_content in new_corpus:
        # 새로운 ID 생성 (예: "1", "2", ...)
        new_doc_id = str(index)

        # 새로 생성한 ID를 키로 하여 new_corpus_dict에 저장
        new_corpus_dict[new_doc_id] = doc_content

        # 원래의 문서 ID를 매핑 딕셔너리에 저장
        doc_id_mapping[new_doc_id] = original_doc_id

        # 인덱스를 1 증가
        index += 1

    task_instance.corpus = new_corpus_dict
    print(f"{task_name} 청크나누기 완료.")
    # SentenceTransformerEncoder 초기화

    encoder_model = SentenceTransformerEncoder(
        model_name_or_path=model_name,
        query_prompt="query: ",
        doc_prompt="passage: ",
        device=device,
    )

    # DenseRetrieval 객체 생성
    retriever = DenseRetrieval(model=encoder_model)
    print(
        f"{task_name}에 대해 SentenceTransformerEncoder와 DenseRetrieval이 성공적으로 초기화되었습니다."
    )

    # 문서 검색 수행
    results = task_instance.retrieve(retriever=retriever)
    print(f"{task_name}에 대해 문서 검색 완료.")

    # Reranker 초기화 및 재정렬 수행
    reranker = CrossEncoderReranker(CrossEncoder(reranker_model_name, device=device))
    reranked_results = task_instance.rerank(reranker, results, top_k=100, batch_size=32)
    print(f"{task_name}에 대해 재정렬 완료.")

    # 결과 저장
    output_dir = os.path.join("output", task_name)
    os.makedirs(output_dir, exist_ok=True)
    task_instance.save_results(top_k=70, output_dir=output_dir)
    print(f"{task_name} 결과가 '{output_dir}'에 저장되었습니다.")

    path_name = next(os.walk(output_dir))[1][0]
    convert_corpus_id_to_original(
        os.path.join(output_dir, path_name, "results.csv"), doc_id_mapping
    )

    process_results(os.path.join(output_dir, path_name, "results.csv"))
