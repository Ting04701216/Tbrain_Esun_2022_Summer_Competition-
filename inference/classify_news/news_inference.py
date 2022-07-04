import re
import torch
from transformers import BertTokenizerFast, AutoModelForSequenceClassification
from src.classify_news import infer, preprocess


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model = 'ckiplab/bert-base-chinese'

tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
model_classify = AutoModelForSequenceClassification.from_pretrained(
    "ckiplab/albert-tiny-chinese",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)
model_classify.load_state_dict(torch.load('../../classify_news/model/finetuned_raw_albert_maxlen32_epoch_2.model', map_location=device))
model_classify.to(device)

def prediction_post_correction(pred):
    """remove non chinese characters"""
    prediction = re.sub('[^\u4E00-\u9FFF]', '', pred)
#     prediction = prediction[:200]
    return prediction

# Example:

# Input: list[str]
sentence_list = ['進而 形成 新 的 傳統',
  '進而 形成 心得 傳統',
  '金額 行程 新 的 傳統',
  '進而 形成 心 的 傳統',
  '金額 形成 新 的 傳統',
  '進而 形成 心地 傳統',
  '金額 行程 心得 傳統',
  '金額 行程 心 的 傳統',
  '金額 形成 心得 傳統',
  '金額 行程 心地 傳統']

# Output: 0 or 1(is_news)
answer = prediction_post_correction(sentence_list[0])
print(infer(preprocess([answer], tokenizer), model_classify, device)[0])