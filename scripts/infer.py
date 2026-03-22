import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

MODEL_DIR = "/content/cs-exam-coach-qlora/outputs/qwen-cs-coach-lora"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    messages = [
        {
            "role": "system",
            "content": "你是一位資工考試助理。回答必須使用繁體中文，並固定輸出四個段落：1.一句話定義 2.核心觀念 3.常見誤區 4.簡單例子。內容要清楚、考試導向、避免太口語。"
        },
        {
            "role": "user",
            "content": "什麼是 page fault？"
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)

if __name__ == "__main__":
    main()
