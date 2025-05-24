from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Optional: Set HF cache dir if needed
# os.environ["HF_HOME"] = "/mnt/disk1/hf_cache"

app = FastAPI()

MODEL_NAME = "ALLaM-AI/ALLaM-7B-Instruct-preview"
offload_dir = "./offload"  # Local disk offload directory
os.makedirs(offload_dir, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

print("Loading model on CPU with disk offload...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,         # Use float32 for CPU
    device_map={"": "cpu"},            # Force CPU
    offload_folder=offload_dir,        # Disk offload folder
    offload_state_dict=True,
    low_cpu_mem_usage=True,
)

print("Model loaded.")

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200

@app.get("/")
def root():
    return {"message": "🚀 ALLaM-7B API is running on CPU"}

@app.post("/generate")
def generate_text(request: PromptRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")
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
