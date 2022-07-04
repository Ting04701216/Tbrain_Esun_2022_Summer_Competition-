import re
import numpy as np
from src.Phone2BoPoMo import phone_to_bopomo
from pyzhuyin import zhuyin_to_pinyin
import pandas as pd

def add_tone(ph, bpm, tn):
    if re.search('[1-5]', ph):
        return bpm + tn
    return bpm

bopomo = pd.read_csv('./data/Phone2BoPoMo_Mapping.csv')
bopomo['BPM'] = bopomo.apply(lambda x: add_tone(x.Phone, x.BoPoMo, x.Tone), axis=1)
bopomo_dict = dict(zip(bopomo.Phone, bopomo.BPM))

def zhuyin_to_pinyin_safe(text):
    try:
        return zhuyin_to_pinyin(text)
    except:
        return text

def bopomo_to_pinyin(sent):
    texts = sent.split()
    return ' '.join([zhuyin_to_pinyin_safe(text) for text in texts])


def infer(sentence_list, phoneme_sequence_list, is_news, model_pinyin2zh):
    
    pattern_clean = '[^\u4E00-\u9FFF]'
    
    l = len(sentence_list)
    
    if l > 3:
        bopomo = phone_to_bopomo(bopomo_dict, phoneme_sequence_list[0])
        pinyin = bopomo_to_pinyin(bopomo)
        pinyin2zh_translate = model_pinyin2zh.translate(pinyin)
        
        answer = pinyin2zh_translate
    else:
        answer = sentence_list[0]
        
    return re.sub(pattern_clean, '', answer)
