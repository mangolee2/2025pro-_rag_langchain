import streamlit as st
import requests
import pandas as pd

# PDF → 텍스트, 임베딩, RAG 관련 함수는 기존 코드 그대로 두시면 됩니다.
# 여기서는 RAG로 찾은 상위 근거 문장만 사용한다고 가정

st.title("💊 신약명 · 효능 · 독성 추출기 (GPT-OSS + RAG)")

uploaded_file = st.file_uploader("PDF 논문 업로드", type=["pdf"])

if uploaded_file is not None:
    # 1. PDF → 텍스트 변환 + 청크 나누기 (기존 코드 함수 호출)
    text_chunks = pdf_to_chunks(uploaded_file)  # 기존 함수 사용
    vector_store = build_vectorstore(text_chunks)  # FAISS 등
    
    question = st.text_input("질문 입력", value="이 논문에서 신약명, 효능(Responsive 여부), 독성(Toxic 여부)을 알려줘.")
    
    if st.button("분석 시작"):
        # 2. 질문 기반 RAG 검색
        top_k = 3
        docs = vector_store.similarity_search(question, k=top_k)
        evidence_sentences = [doc.page_content.strip() for doc in docs]
        
        # 3. 프롬프트 작성 (근거 문장은 따로 전달하지만 최종 답변에는 포함 X)
        prompt = f"""
        아래는 논문에서 찾은 근거 문장들입니다.
        ---
        {' '.join(evidence_sentences)}
        ---
        위 문장들을 바탕으로, 다음 3가지를 표로 작성하세요.
        - 신약명
        - 효능 (Responsive / Non-Responsive)
        - 독성 (Toxic / Non-Toxic)
        
        형식:
        신약명 | 효능 | 독성
        """
        
        # 4. LLM 호출 (Ollama)
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gpt-oss", "prompt": prompt}
        )
        
        answer_text = resp.json().get("response", "").strip()
        
        # 5. 표 형태로 가공
        try:
            rows = [r.strip().split("|") for r in answer_text.split("\n") if "|" in r]
            df = pd.DataFrame(rows, columns=["신약명", "효능", "독성"])
            df = df.apply(lambda x: x.str.strip())
        except:
            df = pd.DataFrame([{"신약명": "", "효능": "", "독성": ""}])
        
        # 6. 근거 문장 표시
        st.subheader("근거 문장")
        for sent in evidence_sentences:
            st.write(f"- {sent}")
        
        # 7. 표 출력
        st.subheader("추출 결과")
        st.table(df)
