import streamlit as st
import fitz  # PyMuPDF
import requests

# ===== 환경 설정 =====
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gpt-oss"  # Ollama에서 설치된 모델 이름 (ex: gpt-oss)

# ===== Streamlit UI =====
st.set_page_config(page_title="PDF GPT-OSS Q&A")
st.title("PDF 기반 GPT-OSS 질의응답")
st.markdown("**로컬 GPT-OSS 모델을 활용한 PDF 질의응답 데모**")

uploaded_file = st.file_uploader("PDF 파일 업로드", type="pdf")
question = st.text_input("질문을 입력하세요:")

# ===== PDF 텍스트 추출 =====
def extract_text_from_pdf(file):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text.strip()

# ===== Ollama 요청 =====
def query_ollama(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"❌ Ollama 요청 오류: {e}"

# ===== 동작 =====
if uploaded_file and question:
    with st.spinner("PDF 처리 중..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    with st.spinner("GPT-OSS가 답변 중..."):
        prompt = (
            f"다음 PDF 내용에 근거하여 질문에 답변하세요.\n\n"
            f"PDF 내용:\n{pdf_text}\n\n"
            f"질문1: 신약명 {question}\n"
            f"질문2: 신약의 효과(responsiv)와 수치 {question}\n, 수치로 간단하게 답해주세요."
            f"질문3: 신약의 독성(toxic)여부 {question}\n, 아니오, 예로 답해주세요."
            f"답변: table format으로 작성하세요."
        )
        answer = query_ollama(prompt)

    st.subheader("답변")
    st.write(answer)
