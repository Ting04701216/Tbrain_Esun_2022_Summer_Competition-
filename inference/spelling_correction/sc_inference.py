import os

import re
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, BertForMaskedLM

from src import pred_correction as sc_infer
from src.dimsim import prepare_dimsim
from src.module import PhoneticMLM


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model = 'ckiplab/bert-base-chinese'
alpha = 8
detector_threshold = 0.5

model_path = '../../phonetic_mlm/saved_models/bert_detection/same_length_best_f1.pth'
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)

def prediction_post_correction(pred):
    """remove non chinese characters"""
    prediction = re.sub('[^\u4E00-\u9FFF]', '', pred)
#     prediction = prediction[:200]
    return prediction

bopomofo_dict, bopomofo_to_dist = prepare_dimsim(tokenizer.vocab)
detector = BertForTokenClassification.from_pretrained(pretrained_model, return_dict=True, num_labels=2)
detector.load_state_dict(torch.load(model_path, map_location=device))
detector.to(device)
mlm = BertForMaskedLM.from_pretrained(pretrained_model)
mlm.to(device)
unknown_bopomofo_id = bopomofo_dict['UNK']
phonetic_mlm_model = PhoneticMLM(detector, mlm, tokenizer, bopomofo_to_dist, unknown_bopomofo_id, alpha=alpha, detector_threshold=detector_threshold)

def inference(sentence_list):
    texts = [prediction_post_correction(x) for x in sentence_list]
    texts = [x for x in texts if x != '']
    det_output = sc_infer.predict_spelling_correction(texts, phonetic_mlm_model, bopomofo_dict, tokenizer, device)
    return det_output

# texts = [prediction_post_correction(x) for x in sentence_list]
# texts = [x for x in texts if x != '']
# det_output = sc_infer.predict_spelling_correction(texts, phonetic_mlm_model, bopomofo_dict, tokenizer, device)
print(inference(sentence_list))
# det_output = list(dict.fromkeys(det_output))
# det_out = sc_infer.predict_typo(det_output, detector, bopomofo_dict, tokenizer, device, threshold=detector_threshold)
# det_count = [len(x) for x in det_out]
# mlm_list = det_output[np.argmin(det_count)]
# print(mlm_list)