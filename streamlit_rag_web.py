import os
import streamlit as st

from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

__import__('pysqlite3')
import sys
sys.modules['sqlite3']=sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma

# 🔑 환경 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# ✅ Chroma DB 초기화
db3 = Chroma(
    persist_directory=r"/mount/src/team-chatbot/chroma_db",
    embedding_function=embedding
)

# ✅ 한-영 교수 이름 매핑
professor_name_map = {
    "노맹석": "Maengseok Noh",
    "문형빈": "HyungBin Moon",
    "하지환": "Jihwan Ha",
    "지준화": "Junhwa Chi",
}

# ✅ 번역 함수
def translate_with_gpt(text, source_lang="ko", target_lang="en") -> str:
    prompt = f"Translate this from {source_lang} to {target_lang}:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# ✅ 질문 유형 분류 함수
def classify_question_type(question_ko: str) -> str:
    prompt = f"""
다음 질문의 유형을 아래 중 하나로 분류해 주세요:
- 논문_목록
- 논문_요약
- 연구_흐름

질문: {question_ko}
질문 유형:"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ✅ 교수명 추출 함수
def extract_professor_name(question: str) -> str | None:
    match = re.search(r"([가-힣]{2,4})\s*교수", question)
    return match.group(1) if match else None

# def get_first_page_summary(doc: Document) -> str:
#     title = doc.metadata.get("title", "제목 정보 없음")
#     content = doc.page_content.strip().split("\n")[0]
#     return f"📌 제목: {title}\n📄 요약: {content}"

# ✅ 프롬프트 템플릿 정의
prompt_templates = {
    "논문_목록": PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are provided with a collection of academic papers written by a professor. 
Based on the following user request, list the key papers along with:

1. The title of each paper (📌 Please keep the title in English)  
2. The publication year (if available)  
3. A few core keywords representing the main topic (in Korean)  
4. The author(s) of each paper (in Korean)

User question:
{question}

Paper content:
{context}

📌 Please write your response in Korean using a respectful and organized tone, **but keep the paper titles in English**.

논문 목록 요약 (in Korean):"""
    ),
    "논문_요약": PromptTemplate(
        input_variables=["context"],
        template="""
You are a research summarization assistant. Based on the following academic paper, provide a clear and concise summary including the following elements:

1. Research subject (what or who is being studied)  
2. Research method (how it was studied)  
3. Research findings (what was discovered)  
4. Suggestions or implications (recommendations or conclusions)

Paper content:
{context}

📌 Please write your summary in Korean, using a polite and professional tone.

논문 요약 (in Korean):"""
    ),
    "연구_흐름": PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an academic assistant. Given a collection of research papers written by a single professor, analyze how the research topics or areas of interest have evolved over time. 
Identify key shifts, trends, or patterns chronologically based on the publication content.

Context:
{context}

Question:
{question}

Now, summarize the chronological progression of the professor’s research focus. 
📌 Please write your answer in Korean using a clear and respectful tone.

연구 흐름 요약 (한국어로):"""
    )
}

# ✅ Streamlit UI 시작
st.set_page_config(page_title="논문 분석 챗봇", page_icon="📄")
st.header("📄 교수님 논문 분석 챗봇")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "빅데이터 융합 전공 교수님들의 논문 목록, 논문 내용, 연구 동향을 알려드립니다 \n논문 제목을 넣을 시 큰 따옴표로 감싸주세요"}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt_message := st.chat_input("질문을 입력해주세요 :)"):
    st.chat_message("user").write(prompt_message)
    st.session_state["messages"].append({"role": "user", "content": prompt_message})

    with st.chat_message("assistant"):
        with st.spinner("질문 분석 중..."):
            try:
                question_type = classify_question_type(prompt_message)
                target_author_ko = extract_professor_name(prompt_message)
                target_author_en = professor_name_map.get(target_author_ko) if target_author_ko else None

                if question_type in ["논문_목록", "연구_흐름"] and not target_author_en:
                    st.error("질문에서 유효한 교수 이름을 찾을 수 없습니다.")
                else:
                    collection = db3._collection.get(include=["metadatas", "documents"])
                    docs = []

                    if question_type == "논문_요약":
                        # 질문에서 논문 제목 추출
                        title_match = re.search(r'\"(.+?)\"', prompt_message)  # 큰따옴표 안의 제목 추출
                        if title_match:
                            target_title = title_match.group(1).lower()
                            docs = [
                                Document(page_content=page, metadata=meta)
                                for page, meta in zip(collection["documents"], collection["metadatas"])
                                if meta.get("title", "").lower() == target_title.lower()
                            ]
                        else:
                            st.error("논문 제목을 \"큰따옴표\"로 감싸 입력해 주세요.")
                    elif question_type == "논문_목록":
                        docs = [
                            Document(page_content=page, metadata=meta)
                            for page, meta in zip(collection["documents"], collection["metadatas"])
                            if meta.get("professor") == target_author_en and meta.get("page") in [0, 1]
                        ]
                    elif question_type == "연구_흐름":
                        docs = [
                            Document(page_content=page, metadata=meta)
                            for page, meta in zip(collection["documents"], collection["metadatas"])
                            if meta.get("professor") == target_author_en and meta.get("page") in [0, 1]
                        ]

                    if question_type in ["논문_목록", "연구_흐름"]:
                        context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)
                    else:
                        context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)

                    prompt = prompt_templates[question_type]
                    chain = prompt | ChatOpenAI(model="gpt-4o")
                    inputs = {"context": context_text}
                    if "question" in prompt.input_variables:
                        inputs["question"] = prompt_message

                    result = chain.invoke(inputs)

                    st.session_state["messages"].append({"role": "assistant", "content": result.content})
                    st.markdown(f"### 🔍 분석 결과: `{question_type}`")
                    st.write(result.content)

            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
