from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/wuyifan/.cache/modelscope/hub/models/shakechen/Llama-2-7b-hf")
s1 = "### Question: Hello\n### Answer:"
s2 = "\n### Answer:"
print(tokenizer.encode(s1))
print("s1 decoded:", [tokenizer.decode([id]) for id in tokenizer.encode(s1)])
print("s2 decoded:", [tokenizer.decode([id]) for id in tokenizer.encode(s2)])
