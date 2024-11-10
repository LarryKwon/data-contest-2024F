### (1) Environment Setup
To begin, install the necessary dependencies:

```bash
# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: financerag_env\Scripts\activate

# Install Poetry (package manager for python, like npm)
pip install poetry

# set config for automatically install python 3.12
poetry config virtualenvs.prefer-active-python true


# install all packages in pyproject.toml. the installed packages can be found in .venv/bin
poetry install

# adding package
poetry add [package]

# removing package
poetry remove [package]

## executing python. it executes the command inside the .venv
poetry run python -V

```

---

### (2) Folder/Class Overview

- **`retrieval/`**:
  - `DenseRetrieval`: Retrieves documents based on dense embeddings.
  - `SentenceTransformerEncoder`: Encodes queries and documents into dense vector representations.

- **`rerank/`**:
  - `CrossEncoderReranker`: Reranks retrieval results using a cross-encoder.

- **`tasks/`**:
  - `BaseTask`: A parent class of each dataset for document retrieval and ranking. Use other dataset tasks that inherit this class.

- **`generate/`**: Handles answer generation processes.

---

### (3) Example Code

1. **Initialize Dataset Task**:
   ```python
   # FinDER for example.
   # You can use other tasks such as `FinQA`, `TATQA`, etc.
   from financerag.tasks import FinDER
   finder_task = FinDER()
   ```

2. **Setup Models**:
   ```python
   from sentence_transformers import SentenceTransformer
   from financerag.retrieval import SentenceTransformerEncoder, DenseRetrieval

   model = SentenceTransformer('intfloat/e5-large-v2')
   # We need to put prefix for e5 models.
   # For more details, see Arxiv paper https://arxiv.org/abs/2212.03533
   encoder_model = SentenceTransformerEncoder(
       q_model=model,
       doc_model=model,
       query_prompt='query: ',
       doc_prompt='passage: '
   )
   retriever = DenseRetrieval(model=encoder_model)
   ```

3. **Retrieve and Rerank**:
   ```python
   # Retrieve relevant documents
   results = finder_task.retrieve(retriever=retriever)

   # Rerank the results
   from financerag.rerank import CrossEncoderReranker
   reranker = CrossEncoderReranker(CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2'))
   reranked_results = finder_task.rerank(reranker, results, top_k=100, batch_size=32)
   ```

4. **Save the Results**:
   After reranking, you can save the results:
   ```python
   finder_task.save_results(output_dir='path_to_save_directory')
   ```

This provides a complete workflow for initializing tasks, performing document retrieval, reranking, and saving the final results.

# Chunk 나누기 설명


### NLTK 설치

NLTK가 설치되어 있지 않은 경우, 아래의 파이썬 코드를 실행하여 설치합니다.

```python
import nltk
nltk.download()
```
- `example_model_chunk.py` 파일을 실행하면 `output` 폴더에 7가지 자료형에 따른 결과가 생성됩니다.
- `merge_example.py` 파일을 실행하면 하나의 CSV 파일로 결과가 병합됩니다.


## `example_model_chunk.py` 청크 나누기 과정 요약

- `task_instance.corpus`에는 검색해야 할 문서들이 `{아이디1: 내용1, 아이디2: 내용2, ...}` 형태의 딕셔너리 자료형으로 저장되어 있습니다.
- 내용을 특정 청크 크기로 나누기 위해 다음 과정을 거칩니다:

  1. 딕셔너리를 리스트로 변환하여 `[(아이디1, 내용1), (아이디2, 내용2)]` 형태로 만듭니다.
  2. 각 내용을 청크 단위로 나누어 `[(아이디1, 내용1-1), (아이디1, 내용1-2), (아이디2, 내용2-1), ...]`와 같은 리스트 형태로 만듭니다.
  3. 이 리스트를 다시 딕셔너리로 변환할 때, 딕셔너리는 하나의 키만 가질 수 있으므로, 각 키를 1부터 시작하여 1씩 증가하는 형태로 변환합니다. `[(1, 내용1-1), (2, 내용1-2), (3, 내용2-1), ...]`
  4. 원래 키로 복원하기 위해 매핑 정보를 `doc_id_mapping`에 저장하며, 이는 `{1: qweqweqwe, 2: qweqweqwe, 3: aaaaaaa, ...}`와 같은 구조로 저장됩니다.


- 이후에는 같은 과정으로 진행이되고 결과 csv파일에는 이런식의 결과가 저장될거임
```
query_id,corpus_id
q6164f62e,3
q6164f62e,2
q6164f62e,2
...
```

- 해야 할 두 가지 작업:
   1. `corpus_id`를 원래 아이디로 변환하기.
   2. 중복되는 `corpus_id`는 가장 상위의 `corpus_id`만 남기고 나머지는 제거하기.

- **2번 작업 문제점**: 중복을 제거하면 기존 상위 10개에서 줄어듭니다. 이를 해결하기 위해 상위 50개의 자료를 우선 기록하고, 마지막에 그중 상위 10개를 다시 추출하는 과정이 포함되어 있습니다.




