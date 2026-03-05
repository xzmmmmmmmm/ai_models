import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/xzm/ai_models/qwen_model"

print("⏳ 正在唤醒 RTX 5060 里的 Qwen...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

# 初始系统设置
history = [{"role": "system", "content": "你是一个幽默、博学的 AI 助手。"}]

print("\n✨ 聊天室已开启！输入 'quit' 退出，输入 'clear' 清空记忆。")

while True:
    user_input = input("\n👤 你: ")
    
    if user_input.lower() == 'quit':
        break
    if user_input.lower() == 'clear':
        history = [{"role": "system", "content": "你是一个幽默、博学的 AI 助手。"}]
        print("🧹 记忆已清空！")
        continue

    # 添加用户话语到历史
    history.append({"role": "user", "content": user_input})

    # 准备输入
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成回复
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]

    print(f"\n🤖 Qwen: {response}")
    
    # 把 AI 的回复也存入历史，实现连续对话
    history.append({"role": "assistant", "content": response})