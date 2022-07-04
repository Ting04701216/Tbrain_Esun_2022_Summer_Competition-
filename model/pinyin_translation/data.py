import json
import os
from ckiptagger import data_utils, construct_dictionary, WS


ws = WS("~/ckip/data", disable_cuda=False)


## 生成供建模使用的raw data
def prepare_raw_data(data, path, is_train=True):
    name = 'train' if is_train else 'valid'
    
    sentence_list = [x['ground_truth_sentence'] for x in data]
    word_sentence_list = ws(sentence_list)
    
    wsl = [' '.join(x) for x in word_sentence_list]
    n = [len(x['pinyin_sequence_list']) for x in data]
    out = [[x]*y for x, y in zip(wsl, n)]
    flat_list = [item for sublist in out for item in sublist]
    
    zh_name = name + '.raw.zh'
    with open(os.path.join(path, zh_name), 'w') as output_file:
        for x in flat_list:
            output_file.write(x + '\n')

    psl = [x['pinyin_sequence_list'] for x in data]
    flat_list2 = [item for sublist in psl for item in sublist]
    
    en_name = name + '.raw.en'
    with open(os.path.join(path, en_name), 'w') as output_file:
        for x in flat_list2:
            output_file.write(x + '\n')
            
    return None


## 原始JSON
with open('../../../data/train_phoneme_0617.json', 'r') as f:
    train = json.load(f)
with open('../../../data/valid_phoneme_0617.json', 'r') as f:
    valid = json.load(f)

prepare_raw_data(train, 'data/all', is_train=True)
prepare_raw_data(valid, 'data/all', is_train=False)
