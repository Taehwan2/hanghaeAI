from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import wandb
wandb.login(key="")
wandb.init(
    project="opt350m-lora-sft",      # 원하는 프로젝트명
    name="run-3"                  # 실험 이름
)
dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
lora_r: int = 256
lora_dropout: float = 0.1
lora_alpha: int = 32
target_modules = set()

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        names = name.split('.')
        target_modules.add(names[0] if len(names) == 1 else names[-1])

if "lm_head" in target_modules:  # needed for 16-bit
    target_modules.remove("lm_head")

target_modules = list(target_modules)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules
)
model = get_peft_model(model, peft_config)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


trainer = SFTTrainer(
    model,
    train_dataset=dataset,
        args=SFTConfig(
        output_dir="/tmp/clm-instruction-tuning",
        report_to="wandb",
        logging_steps=10,
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)
trainer.train()
print('Max Alloc:', round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')
wandb.finish()  # wandb 세션 종료