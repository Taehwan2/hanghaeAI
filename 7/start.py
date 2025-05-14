from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
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

prompt = "손의 각속도 변화를 요약해줘." + " ### Answer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs=inputs["input_ids"],
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)