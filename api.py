import datetime
import hashlib
import time

from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

####
import os
import re
import json
import pandas as pd
from src import classify_news_infer as classify
from fairseq.models.transformer import TransformerModel

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForTokenClassification, BertForMaskedLM, AutoModel, BertTokenizerFast, AutoModelForSequenceClassification

from src import translate_utils
from src.sc import pred_correction as sc_infer
from src.sc.dimsim import prepare_dimsim
from src.sc.module import PhoneticMLM
####

app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'a0989303198@gmail.com' #
SALT = 'TWICE'                          #
#########################################

####### CONFIG ##########################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
pretrained_model = 'ckiplab/bert-base-chinese'
alpha = 8
detector_threshold = 0.5
#########################################

####### MODEL ##########################
## classify news
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
model_classify = AutoModelForSequenceClassification.from_pretrained(
    "ckiplab/albert-tiny-chinese",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)
model_classify.load_state_dict(torch.load('model/finetuned_raw_albert_maxlen32_epoch_2.model', map_location=device))
model_classify.to(device)

## fairseq translate
checkpoint_name = 'checkpoint_best.pt'
path_esun = 'fairseq/pinyin2chinese/nmt/models/all/checkpoints'
pinyin2zh_esun = TransformerModel.from_pretrained(model_name_or_path=path_esun, checkpoint_file=checkpoint_name, data_name_or_path='../../../data/all/data-bin')
path_news = 'fairseq/pinyin2chinese/nmt/models/news/checkpoints'
pinyin2zh_news = TransformerModel.from_pretrained(model_name_or_path=path_news, checkpoint_file=checkpoint_name, data_name_or_path='../../../data/news/data-bin')

## phonetic_mlm
bopomofo_dict, bopomofo_to_dist = prepare_dimsim(tokenizer.vocab)
detector = BertForTokenClassification.from_pretrained(pretrained_model, return_dict=True, num_labels=2)
detector.load_state_dict(torch.load('model/same_length_best_f1.pth', map_location=device))
detector.to(device)
mlm = BertForMaskedLM.from_pretrained(pretrained_model)
mlm.to(device)
unknown_bopomofo_id = bopomofo_dict['UNK']
phonetic_mlm_model = PhoneticMLM(detector, mlm, tokenizer, bopomofo_to_dist, unknown_bopomofo_id, alpha=alpha, detector_threshold=detector_threshold)
#########################################

def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def predict(sentence_list, phoneme_sequence_list):
    """ Predict your model result.

    @param:
        sentence_list (list): an list of sentence sorted by probability.
        phoneme_sequence_list (list): an list of phoneme sequence sorted by probability.
    @returns:
        prediction (str): a sentence.
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    t1 = time.time()
    
    default_answer = prediction_post_correction(sentence_list[0])
    
    if len(default_answer) > 64:
        return default_answer
    
    
    ## classify news
    is_news = classify.infer(classify.preprocess([default_answer], tokenizer), model_classify, device)[0]
    
    
    ## phone2zh translate
    model_pinyin2zh = pinyin2zh_news if is_news == 1 else pinyin2zh_esun
    phone2zh_translate = translate_utils.infer(sentence_list, phoneme_sequence_list, is_news, model_pinyin2zh)
    
    
    ## phonetic_mlm spelling_correction
    texts = [prediction_post_correction(x) for x in sentence_list]
    texts = [x for x in texts if x != '']
    det_output = sc_infer.predict_spelling_correction(texts, phonetic_mlm_model, bopomofo_dict, tokenizer, device)
    if (is_news == 0) & (len(phone2zh_translate) <= 3):
        det_output = [phone2zh_translate] + det_output
    else:
        det_output.append(phone2zh_translate)
    det_output = list(dict.fromkeys(det_output))
    if len(det_output) == 1:
        return det_output[0]
    det_out = sc_infer.predict_typo(det_output, detector, bopomofo_dict, tokenizer, device, threshold=detector_threshold)
    det_count = [len(x) for x in det_out]
    mlm_list = det_output[np.argmin(det_count)]   
    
    
    ## spelling correction again
    final_output = sc_infer.predict_spelling_correction([mlm_list], phonetic_mlm_model, bopomofo_dict, tokenizer, device)[0]
    
    return final_output



def prediction_post_correction(pred):
    prediction = re.sub('[^\u4E00-\u9FFF]', '', pred)
    prediction = prediction[:200]
    return prediction


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)

    # 自行取用，可紀錄玉山呼叫的 timestamp
    esun_timestamp = data['esun_timestamp']

    # 取 sentence list 中文
    sentence_list = data['sentence_list']
    # 取 phoneme sequence list (X-SAMPA)
    phoneme_sequence_list = data['phoneme_sequence_list']

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)
            
    try:
        answer = predict(sentence_list, phoneme_sequence_list)
    except:
        answer = prediction_post_correction(sentence_list[0])
        
    server_timestamp = time.time()
        
    return jsonify({'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': answer,
                    'server_timestamp': server_timestamp})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8050, debug=False)
