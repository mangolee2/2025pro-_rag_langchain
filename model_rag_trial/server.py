# server.py
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import pandas as pd
from tempfile import TemporaryDirectory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

app = FastAPI(title="Mouse Drug RAG WebApp")

# --- LLM & Prompt 준비 ---
prompt = PromptTemplate.from_template("""
당신은 마우스 신약 논문을 분석하는 과학 보조원입니다.  
아래 Context를 기반으로 "신약명, 효과, 독성, 출처"를 요약 테이블로 작성하세요.

출력 형식:
Drug | Effect | Toxicity | Source
---- | ------ | -------- | ------
{context}
""")

llm = ChatOllama(model="gpt-oss", base_url="http://localhost:11434")
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# --- HTML 업로드 폼 ---
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Mouse Drug PDF Upload</title>
        </head>
        <body>
            <h1>PDF 업로드 후 CSV 다운로드</h1>
            <form action="/upload_pdf" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept=".pdf" multiple>
                <input type="submit" value="업로드 및 처리">
            </form>
        </body>
    </html>
    """

# --- PDF 업로드 & 처리 ---
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "PDF 파일만 업로드 가능합니다."}

    with TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, file.filename)
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        # PDF 로드
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)

        # FAISS 벡터화 (RAG)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # LLM로 테이블 추출
        all_rows = []
        for doc in split_docs:
            chunk_text = doc.page_content
            formatted_prompt = prompt.format(context=chunk_text)
            try:
                table_text = llm.invoke(formatted_prompt)
                if hasattr(table_text, "page_content"):
                    table_text = table_text.page_content
            except Exception as e:
                return {"error": str(e)}

            lines = table_text.split("\n")
            for line in lines:
                if "|" in line and not line.startswith("Drug") and not line.startswith("---"):
                    cells = [c.strip() for c in line.split("|")]
                    if len(cells) == 4:
                        all_rows.append(cells)

        # CSV 생성
        df = pd.DataFrame(all_rows, columns=["Drug", "Effect", "Toxicity", "Source"])
        output_path = os.path.join(tmpdir, "mouse_drug_summary.csv")
        df.to_csv(output_path, index=False)

        return FileResponse(output_path, media_type="text/csv", filename="mouse_drug_summary.csv")
