# app_rag.py
import streamlit as st
import fitz                # pip install pymupdf
import requests
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss               # pip install faiss-cpu (or faiss-gpu)
from tqdm import tqdm
import os
import json
from langchain.schema import SystemMessage, HumanMessage

# -----------------------
# 설정
# -----------------------
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2-vision"        # 로컬에서 돌아가는 모델: 멀티모달 씀 
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800               # 청크 길이 (문자)
CHUNK_OVERLAP = 150
TOP_K = 5                      # 검색할 청크 개수
EMBED_DIM = 384                # all-MiniLM-L6-v2 임베딩 차원 (모델 기준)

# -----------------------
# 유틸: 파일 해시
# -----------------------
def file_hash_bytes(data: bytes) -> str:
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
# 임베딩 준비 (캐시)
# -----------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

# -----------------------
# FAISS 인덱스 생성
# -----------------------
def build_faiss_index(embeddings: np.ndarray):
    # L2 내적 유사도용 인덱스 (Normalized cosine)
    index = faiss.IndexFlatIP(EMBED_DIM)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# -----------------------
# Ollama 호출 (안전 처리)
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
        # Ollama 응답 포맷 안전 처리
        if isinstance(data, dict):
            if "response" in data:
                return data["response"], None
            else:
                # 전체 JSON을 에러로 보여주기
                return None, data
        else:
            return None, {"error": "unexpected response type", "raw": data}
    except Exception as e:
        return None, {"error": str(e)}

# -----------------------
# RAG 질의 처리: 질문 리스트 순차 실행
# -----------------------

# 시스템 메시지 정의
system_message = SystemMessage(content="you are a reasearcher for new drug development related to animal experiments. You have to explain generic names in papers.")


def rag_answer_questions(questions, retriever, chunks, embedder):
    results = {}
    for q in questions:
        # 임베딩: 질문 임베딩
        q_emb = embedder.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = retriever.search(q_emb, TOP_K)  # I shape (1, k)
        idxs = I[0].tolist()
        context = "\n\n---\n\n".join([chunks[i] for i in idxs if i < len(chunks)])
        # 프롬프트 템플릿: 명확한 포맷 요청
        prompt = (
                f"You are a biomedical text analysis assistant.\n\n"
                f"==== Document Excerpt Start ====\n{context}\n==== Document Excerpt End ====\n\n"
                f"Task:\n"
                f"- Extract only **generic (non-brand)** drug names mentioned in the 'Results' section of the provided research paper.\n"
                f"- If no drug names are found in the Results section, then extract them from the 'Methods' section instead.\n"
                f"- Exclude any drug names appearing in the 'References' section.\n"
                f"- Exclude gene names, protein names, biomarkers, molecular targets, and signaling pathway components "
                f"(e.g., EGFR, MEK, CDK4, PTEN, ERBB2, FGFR3, etc.).\n"
                f"- Exclude experimental or investigational code names (e.g., TAS0728, T-DM1, AZD9496, MK-2206, LY294002).\n"
                f"- Include only drugs that have an established or officially recognized generic name (INN).\n"
                f"- Extract only therapeutic agents or compounds actually tested, administered, or mentioned as drugs.\n"
                f"- List only essential and unique generic drug names.\n\n"
                f"Output format:\n"
                f"List all unique generic drug names separated by semicolons (;), without quotation marks.\n"
                f"Do not include any additional text, explanation, or numbering.\n\n"
                f"Example output:\n"
                f"nivolumab; pembrolizumab; sunitinib; fulvestrant\n\n"
                f"Question: {q}\n\n"
                f"Answer:"
            )

        answer, err = call_ollama(prompt)
        if err:
            results[q] = {"answer": None, "error": err, "context_idxs": idxs}
        else:
            results[q] = {"answer": answer, "error": None, "context_idxs": idxs}
    return results

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="PDF → RAG → LLM (신약 정보 자동화)", layout="wide")
st.title("🔬 PDF → RAG → LLM : 신약명 자동 추출")

st.markdown("업로드 된 논문(PDF)에서 **신약명** 항목을 자동으로 추출합니다. Ollama와 로컬 임베딩+FAISS를 사용한 RAG 방식입니다.")

uploaded_file = st.file_uploader("논문(PDF) 업로드", type=["pdf"])
run_button = st.button("분석 시작")

if uploaded_file and run_button:
    raw = uploaded_file.read()
    fid = file_hash_bytes(raw)
    st.info(f"파일 해시: {fid[:12]}... (캐싱용)")
    with st.spinner("PDF에서 텍스트 추출 중..."):
        text = extract_text_from_pdf_bytes(raw)

    if not text:
        st.error("PDF에서 텍스트를 추출할 수 없습니다. 스캔한 이미지형 PDF인 경우 OCR 필요.")
    else:
        st.success(f"총 문자 수: {len(text)}")
        # 청크 분할
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        st.write(f"생성된 청크 수: {len(chunks)} (청크 길이={CHUNK_SIZE}, 겹침={CHUNK_OVERLAP})")

        # 임베딩 & FAISS (캐시 파일 폴더 사용)
        cache_dir = os.path.join(".cache_rag")
        os.makedirs(cache_dir, exist_ok=True)
        index_path = os.path.join(cache_dir, f"{fid}.index")
        chunks_path = os.path.join(cache_dir, f"{fid}_chunks.json")

        embedder = load_embedder()

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            try:
                st.info("이 파일에 대한 FAISS 인덱스 캐시를 로드합니다.")
                idx = faiss.read_index(index_path)
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                retriever = idx
            except Exception as e:
                st.warning("캐시 로드 실패, 인덱스 재생성합니다.")
                os.remove(index_path) if os.path.exists(index_path) else None
                # fallthrough to create
                retriever = None
        else:
            retriever = None

        if retriever is None:
            with st.spinner("임베딩 생성 및 FAISS 인덱스 구축 중... (약간 시간 소요)"):
                # 배치 임베딩
                batch_size = 64
                embeddings = []
                for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
                    batch = chunks[i:i+batch_size]
                    embs = embedder.encode(batch, convert_to_numpy=True)
                    embeddings.append(embs)
                embeddings = np.vstack(embeddings).astype("float32")
                # Build index
                if embeddings.shape[1] != EMBED_DIM:
                    st.warning(f"임베딩 차원 {embeddings.shape[1]} (예상 {EMBED_DIM})입니다. EMBED_DIM 값을 맞춰주세요.")
                faiss.normalize_L2(embeddings)
                retriever = faiss.IndexFlatIP(embeddings.shape[1])
                retriever.add(embeddings)
                # save index and chunks
                faiss.write_index(retriever, index_path)
                with open(chunks_path, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, ensure_ascii=False)
                st.success("FAISS 인덱스 생성 완료 및 캐시 저장됨.")

        # 자동질문 리스트
        questions = [
            "신약명 추출"
        ]

        st.info("검색된 문서 청크를 기반으로 Ollama(gpt-oss)에 질의합니다...")
        results = rag_answer_questions(questions, retriever, chunks, embedder)

        # 결과 표시
        for q in questions:
            st.markdown("---")
            st.subheader(q)
            entry = results[q]
            if entry["error"]:
                st.error(f"질의 실패: {entry['error']}")
                st.write("관련 청크 인덱스:", entry.get("context_idxs"))
            else:
                st.write(entry["answer"])
                # 간략하게 관련 청크도 보여주기
                # st.markdown("**참조된 문장(청크):**")
                # for idx in entry.get("context_idxs", []):
                    # if idx < len(chunks):
                        # st.write(f"- (idx {idx}) " + chunks[idx][:400].replace("\n", " ") + ("..." if len(chunks[idx]) > 400 else ""))
st.markdown("---")
st.caption("Tip: 청크 크기, overlap, top_k 값은 문서 특성에 따라 조절하세요.")
