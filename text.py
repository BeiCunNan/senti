# # 读取原始数据
# with open('data/MR_CV.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# # 打乱数据
# random.shuffle(data)
#
# # 将打乱后的数据输出到文件中
# with open('data/MR_CV.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)
from numpy import mean

l = [88.2, 92.5, 98.0, 97.4, 92.0, 93.0, 55.6, 76.2]
print(mean(l))

from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings

model_name = "microsoft/xtremedistil-l12-h384-uncased"  # Find popular HuggingFace models here: https://huggingface.co/models
input_text = "The cat sat on the mat"
model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
outputs = model(inputs)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
model_view(attention, tokens)  # Display model view