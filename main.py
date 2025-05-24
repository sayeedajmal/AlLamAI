from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import offload_model, OffloadConfig
import torch
import os

# Set HF cache dir (optional, change to your path)
os.environ["HF_HOME"] = "/mnt/disk1/hf_cache"

app = FastAPI()

MODEL_NAME = "ALLaM-AI/ALLaM-7B-Instruct-preview"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

print("Loading model with disk offloading...")
# Load model on CPU with float16 for memory savings
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map={"": "cpu"},  # Load entirely on CPU first
)

# Setup disk offload config, point to a folder with enough free space
offload_dir = "/mnt/disk1/offload"
os.makedirs(offload_dir, exist_ok=True)

offload_config = OffloadConfig(
    offload_folder=offload_dir,
    pin_memory=True,
    max_memory={  # you can tune this based on your RAM; example:
        "cpu": "12GiB",  # your RAM size approx
        "disk": "100GiB",  # free disk space for offloading
    }
)

offload_model(model, offload_config)

print("Model loaded and offloaded to disk.")

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
