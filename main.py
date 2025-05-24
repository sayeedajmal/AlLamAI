from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Model setup
MODEL_NAME = "ALLaM-AI/ALLaM-7B-Instruct-preview"

# Automatically download model/tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Request schema
class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200

@app.get("/")
def root():
    return {"message": "🚀 ALLaM-7B API is running"}

@app.post("/generate")
def generate_text(request: PromptRequest):
    # Tokenize and generate
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

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
