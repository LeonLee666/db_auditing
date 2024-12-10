from transformers import BertTokenizer, BertModel

# 指定模型名称
model_name = 'bert-base-uncased'

# 下载并保存模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 保存到本地
tokenizer.save_pretrained('./bert-base-uncased')
model.save_pretrained('./bert-base-uncased')
