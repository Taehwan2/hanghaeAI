import chromadb
import streamlit as st

from transformers import pipeline
import torch
from huggingface_hub import login
from langchain import hub
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter




@st.cache_resource
def get_docs(pdf_stream):
    with open('tmp.pdf', 'wb') as f:
        f.write(pdf_stream.getvalue())
        loader = PyPDFLoader("tmp.pdf")

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        return splits


def get_retriever(docs):
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    splits = get_docs(docs)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(api_key=OPEN_API)
    )
    return vectorstore.as_retriever()

    rag_prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

st.title("RAG Bot")


@st.cache_resource
def load_exaone():
    hf_token =
    login(hf_token)
    model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

    # 모델 & 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,  # 필수[1][3]
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True  # 추가 필수[3]
    )

    # 파이프라인 구성
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        torch_dtype=torch.bfloat16
    )
    return HuggingFacePipeline(pipeline=pipe)

model = load_exaone()

if docs := st.file_uploader("Upload your PDF here and click", type="pdf"):
    retriever = get_retriever(docs)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("What is your question about the PDF?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        retrieved_docs = retriever.invoke(prompt)
        user_prompt = rag_prompt.invoke({
            "context": format_docs(retrieved_docs),
            "question": prompt
        })
        result = model.invoke(user_prompt)
        response = getattr(result, "content", result)
        st.markdown(response)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
