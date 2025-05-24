from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights, disk_offload
import torch
import os

os.environ["HF_HOME"] = "./hf_cache"

app = FastAPI()

MODEL_NAME = "ALLaM-AI/ALLaM-7B-Instruct-preview"
offload_dir = "./offload"
os.makedirs(offload_dir, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

print("Loading model with disk offload...")
# Load config only first
from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL_NAME)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

device_map = infer_auto_device_map(model, max_memory={0: "10GiB"}, no_split_module_classes=["LlamaDecoderLayer"])

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device_map,
    offload_folder=offload_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

print("Model loaded.")

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200

@app.get("/")
def root():
    return {"message": "🚀 ALLaM-7B API is running"}

@app.post("/generate")
def generate_text(request: PromptRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=request.max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
