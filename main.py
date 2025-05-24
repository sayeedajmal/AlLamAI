from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch
import os

os.environ["HF_HOME"] = "/mnt/disk1/hf_cache"

app = FastAPI()

MODEL_NAME = "ALLaM-AI/ALLaM-7B-Instruct-preview"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

print("Initializing model with offloading...")
with init_empty_weights():
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_config(config)

offload_dir = "/mnt/disk1/offload"
os.makedirs(offload_dir, exist_ok=True)

model = load_checkpoint_and_dispatch(
    model,
    MODEL_NAME,
    device_map="auto",
    offload_folder=offload_dir,
    dtype=torch.float16,
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
