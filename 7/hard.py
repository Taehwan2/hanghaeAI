import chromadb

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

OPEN_API = 


def get_docs(pdf_stream):
        loader = PyPDFLoader(pdf_stream)

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


model = AutoModelForCausalLM.from_pretrained(
    "./my_finetuned_model",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "./my_finetuned_model",
    trust_remote_code=True
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = "이 논문 결과에 대해서 요약해봐 50자 이내로." + " ### Answer:"

retriever = get_retriever('basketball.pdf')
retrieved_docs = retriever.invoke(prompt)
user_prompt = rag_prompt.invoke({
        "context": format_docs(retrieved_docs),
        "question": prompt
})
if not isinstance(user_prompt, str):
        user_prompt = str(user_prompt)
inputs = tokenizer(user_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)





