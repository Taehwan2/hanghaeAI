import chromadb
import streamlit as st
import wandb
import json
from transformers import pipeline
import torch
from datasets import Dataset
from huggingface_hub import login
from langchain import hub
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

#완디비 시작
wandb.init(project='Hanghae99', name='gpt-finetuning2')


#사용안함
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts



#모델불러오기
hf_token = ''
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

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)



#파일열고 저장하기 
with open('./basket.json', 'rb') as f:
    d = json.load(f)




formatted_data = [
    {"instruction": instr, "output": out} 
    for instr, out in zip(d["instruction"], d["output"])
]

n_insts = len(formatted_data)
train_data = formatted_data[:int(n_insts * 0.8)]
eval_data = formatted_data[int(n_insts * 0.8):]

train_dataset = Dataset.from_list(formatted_data)
eval_dataset =  Dataset.from_list(formatted_data2)

def formatting_prompts_func2(example):
    return f"### Question: {example['instruction']}\n ### Answer: {example['output']}"

# 검증 메트릭 계산 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    loss = torch.nn.CrossEntropyLoss()(torch.tensor(logits), torch.tensor(labels))
    return {"eval_loss": loss.item()}
    
#실제 학습하기
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="/tmp/clm-instruction-tuning"),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func2,
    data_collator=collator,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model("./my_finetuned_model")  # 모델 저장
tokenizer.save_pretrained("./my_finetuned_model")  # 토크나이저 저장
