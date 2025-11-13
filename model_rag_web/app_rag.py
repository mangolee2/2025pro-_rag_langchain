import streamlit as st
import fitz  # PyMuPDF
import requests
import json

st.set_page_config(page_title="PDF + GPT-OSS", layout="wide")
st.title("📄 PDF 기반 질의응답")

# PDF 업로드
uploaded_file = st.file_uploader("PDF 업로드", type="pdf")
question = st.text_input("질문을 입력하세요")

if uploaded_file and question:
    # 1. PDF → 텍스트 변환
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in pdf])

    # 2. Ollama API 호출
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gpt-oss",   # GPT-OSS 로컬 모델
                "prompt": f"{text}\n\nQuestion: {question}",
                "stream": False
            },
            timeout=120
        )
        data = resp.json()

        if "response" in data:
            answer = data["response"]
            st.subheader("답변")
            st.write(answer)
        else:
            st.error(f"API 오류: {data}")

    except requests.exceptions.RequestException as e:
        st.error(f"Ollama 서버 연결 실패: {e}")
