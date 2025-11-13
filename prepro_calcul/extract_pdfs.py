import fitz
import hashlib
import requests
import json
import os
import numpy as np
import faiss
import csv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -----------------------
# 설정
# -----------------------
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gpt-oss"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
EMBED_DIM = 384

# -----------------------
# 유틸: 파일 해시
# -----------------------
def file_hash_bytes(data: bytes) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

# -----------------------
# PDF -> 텍스트
# -----------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for p in pdf:
        pages.append(p.get_text())
    return "\n\n".join(pages).strip()

# -----------------------
# 텍스트를 청크로 분할
# -----------------------
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == L:
            break
        start = end - overlap
    return chunks

# -----------------------
# 임베딩 준비
# -----------------------
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

# -----------------------
# Ollama 호출
# -----------------------
def call_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout=120):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_API, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "response" in data:
            return data["response"]
        return None
    except Exception as e:
        print("Ollama error:", e)
        return None

# -----------------------
# RAG 질의 처리
# -----------------------
def rag_extract_drugs(pdf_path, embedder):
    # PDF 읽기
    with open(pdf_path, "rb") as f:
        raw = f.read()

    title = os.path.basename(pdf_path)
    text = extract_text_from_pdf_bytes(raw)
    if not text:
        print(f"[WARN] {title} : 텍스트 추출 실패")
        return title, []

    # 청크 분할
    chunks = chunk_text(text)

    # 임베딩 생성
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(embeddings)

    # 인덱스 생성
    retriever = faiss.IndexFlatIP(embeddings.shape[1])
    retriever.add(embeddings)

    # 질문: 신약명 추출
    q = "논문에서 언급된 신약명들(후보)을 알려줘. 결과는 세미콜론(;)으로 구분된 목록 형태로 출력해."
    q_emb = embedder.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = retriever.search(q_emb, TOP_K)
    idxs = I[0].tolist()
    context = "\n\n---\n\n".join([chunks[i] for i in idxs if i < len(chunks)])

    prompt = (
        f"다음 문서 발췌문을 참고하여 질문에 답하세요.\n\n"
        f"==== 문서 발췌 시작 ====\n{context}\n==== 문서 발췌 끝 ====\n\n"
        f"질문: {q}\n"
        f"응답:"
    )

    answer = call_ollama(prompt)
    if not answer:
        return title, []

    # 세미콜론으로 split
    drugs = [d.strip() for d in answer.split(";") if d.strip()]
    return title, drugs

def main(pdf_dir="pdfs", output_csv="results.csv"):
    embedder = load_embedder()
    rows = []

    # 이미 처리한 CSV 읽기 (중복 방지)
    processed_titles = set()
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_titles.add(row["title"])

    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, fname)
            title = os.path.basename(pdf_path)

            # 이미 처리된 논문은 스킵
            if title in processed_titles:
                print(f"스킵: {title} (이미 CSV에 존재)")
                continue

            print(f"처리 중: {title}")
            title, drugs = rag_extract_drugs(pdf_path, embedder)
            rows.append({"title": title, "drugs": ";".join(drugs)})

            # 매 파일마다 결과를 바로 append 저장 (안전)
            write_header = not os.path.exists(output_csv)
            with open(output_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["title", "drugs"])
                if write_header:
                    writer.writeheader()
                writer.writerow({"title": title, "drugs": ";".join(drugs)})

    print(f"완료 ✅ 결과 CSV 업데이트됨: {output_csv}")