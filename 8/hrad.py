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
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import wandb

wandb.login(key="")
wandb.init(project='Hanghae99', name='gpt-finetuning2')
with open('./basket.json', 'rb') as f:
    d = json.load(f)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts


# 데이터 분할
# n_insts = len(formatted_data)
# train_data = formatted_data[:int(n_insts * 0.8)]
# eval_data = formatted_data[int(n_insts * 0.8):]
# train_dataset = Dataset.from_list(train_data)
# eval_dataset = Dataset.from_list(eval_data)


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

lora_r: int = 8
lora_dropout: float = 0.1
lora_alpha: int = 32
target_modules = set()

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        names = name.split('.')
        target_modules.add(names[0] if len(names) == 1 else names[-1])

if "lm_head" in target_modules:  # needed for 16-bit
    target_modules.remove("lm_head")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules
)

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

formatted_data = [
    {"instruction": instr, "output": out}
    for instr, out in zip(d["instruction"], d["output"])
]
train_dataset = Dataset.from_list(formatted_data)


def formatting_prompts_func2(example):
    return f"### Question: {example['instruction']}\n ### Answer: {example['output']}"


# 검증 메트릭 계산 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    loss = torch.nn.CrossEntropyLoss()(torch.tensor(logits), torch.tensor(labels))
    return {"eval_loss": loss.item()}


trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="/tmp/clm-instruction-tuning", report_to="wandb"),
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func2,
    data_collator=collator,
    compute_metrics=compute_metrics,
    peft_config=peft_config  # LoRA 설정 추가
)
trainer.train()
print('Max Alloc:', round(torch.cuda.max_memory_allocated(0) / 1024 ** 3, 1), 'GB')
trainer.save_model("./my_finetuned_model2")  # 모델 저장
tokenizer.save_pretrained("./my_finetuned_model2")  # 토크나이저 저장