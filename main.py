from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch
import os
import shutil

MODEL_ID = "ALLaM-AI/ALLaM-7B-Instruct-preview"
LOCAL_MODEL_DIR = "./local_model"
OFFLOAD_DIR = "./offload"

os.environ["HF_HOME"] = "hf_cache"

app = FastAPI()

# === Download model if not exists ===
if not os.path.isdir(LOCAL_MODEL_DIR):
    print("🔽 Downloading model to local folder...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    model.save_pretrained(LOCAL_MODEL_DIR)
    print("✅ Download complete.")
else:
    print("✅ Using cached local model.")

# === Load tokenizer ===
print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, use_fast=False)

# === Load config & model with disk offloading ===
print("⚙️ Loading config...")
config = AutoConfig.from_pretrained(LOCAL_MODEL_DIR)

print("🧠 Initializing empty model...")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

print("💾 Offloading model to disk...")
os.makedirs(OFFLOAD_DIR, exist_ok=True)
model = load_checkpoint_and_dispatch(
    model,
    LOCAL_MODEL_DIR,
    device_map="auto",
    offload_folder=OFFLOAD_DIR,
    dtype=torch.float16
)

print("✅ Model loaded.")

# === API schema ===
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
