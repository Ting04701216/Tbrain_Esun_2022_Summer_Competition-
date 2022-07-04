from Phone2BoPoMo import phone_to_bopomo
from pyzhuyin import zhuyin_to_pinyin
import copy
import re
from typing import List
import pandas as pd

bopomo_path = 'Phone2BoPoMo_Mapping.csv'

# bopomo mapping

def add_tone(ph, bpm, tn):
    if re.search('[1-5]', ph):
        return bpm + tn
    return bpm

bopomo = pd.read_csv(bopomo_path)
bopomo['BPM'] = bopomo.apply(lambda x: add_tone(x.Phone, x.BoPoMo, x.Tone), axis=1)
bopomo_dict = dict(zip(bopomo.Phone, bopomo.BPM))


# bopomo to pinyin
def zhuyin_to_pinyin_safe(text):
    try:
        return zhuyin_to_pinyin(text)
    except:
        return text

def bopomo_to_pinyin(sent):
    texts = sent.split()
    return ' '.join([zhuyin_to_pinyin_safe(text) for text in texts])


class DataCleaning:
    def __init__(self, sample):
        self.sample = copy.deepcopy(sample)
    
    def process_ground_truth(self):
        """basic cleaning
        
        Examples:
            # '○' -> '零'
            預計今天恐湧現一五○萬人潮 -> 零
            預估稅損約四○億元 -> 零 (新聞源應該是阿拉伯數字直翻)
            
            # '〇' -> '零'
            
            # '[^\u4e00-\u9fff]' -> ''
            學生的回答不一樣了⋯⋯ -> 學生的回答不一樣了

        """
        
        sent = self.sample['ground_truth_sentence']
        
        # if re.search('○', sent):
        if '○' in  sent:
            sent = re.sub('○', '零', sent)
            
        if '〇' in sent:
            sent = re.sub('〇', '零', sent)
        
        # if re.search('[^\w\s]', sent):
        sent = re.sub('[^\u4e00-\u9fff]+', '', sent)
        
        # sent = re.sub('\xa0', '', sent)
        
        self.sample['ground_truth_sentence'] = sent
        
        # return self.sample
    
    
    def process_sent_list(self, filter_num):
        """basic cleaning

        Examples:
            # 'e-mail' -> ''
            
            # '[^\u4e00-\u9fff]' -> ''
            學生的回答不一樣了⋯⋯ -> 學生的回答不一樣了
            
            # <htr> -> *
            TBD
            
        """
        
        sents = self.sample['sentence_list']
        to_append = []
        
        if filter_num and filter_num < len(sents):
            sents = sents[:filter_num]
        
        for ind, sent in enumerate(sents):
            
            
            if '.' in sent:
                # if re.search('[^\w\s]', sent):
                sents[ind] = re.sub('\\.', '', sent)
                
            if 'e-mail' in sent:
                sents[ind] = re.sub(' e-mail', '', sent)
                
            if '<htr>' in sent:
                sents[ind] = re.sub('<htr>', '*', sent)
            
            # add a clean version
            if re.search('[^\u4e00-\u9fff ]', sent):
                to_append.append(re.sub('[^\u4e00-\u9fff ]+', '', sent).strip())
            
        sents = sents + list(set(to_append))
        self.sample['sentence_list'] = sents
        
    def clean_data(self, filter_num=None):
        """main

        Examples:
            >>> DC = DataCleaning(train[369])
            >>> DC.clean_data()
        """
        
        self.process_ground_truth()
        self.process_sent_list(filter_num)
        return self.sample

def prepare_phoneme_input(sample):
    DC = DataCleaning(sample)
    sample = DC.clean_data(filter_num=None)
    sample['bopomo_sequence_list'] = [phone_to_bopomo(bopomo_dict, sent) for sent in sample['phoneme_sequence_list']]
    sample['pinyin_sequence_list'] = [bopomo_to_pinyin(sent) for sent in sample['bopomo_sequence_list']]
    return sample


## Example:

# Input: 含 ground_truth_sentence, sentence_list 以及 phoneme_sequence_list 之原始資料
import json
with open('../../data/train_all.json') as jfile:
    train_all = json.load(jfile)
test = train_all[369]


# Output: 
# prepare_phoneme_input: dictionary

prepare_phoneme_input(test)
# {'ground_truth_sentence': '欸喂你好我想要問分期', 'sentence_list': ['欸 你好 我 想要 問 分期', '痾 你好 我 想要 問 分期', '痾 您好 我 想要 問 分期', '痾 學姊 你好 我 想要 問 分期', '欸 您好 我 想要 問 分期', '呃 你好 我 想要 問 分期', '呃 沒 你好 我 想要 問 分期', '呃 學姊 您好 我 想要 問 分期', '呃 學姊 你好 我 想要 問 分期', '<htr> 你好 我 想要 問 分期'], 
#  'phoneme_sequence_list': ['eI4 n i:3 x aU3 w O:3 s6 j A: N3 j aU4 w ax n4 f ax n1 ts6_h i:2', 'ax1 n i:3 x aU3 w O:3 s6 j A: N3 j aU4 w ax n4 f ax n1 ts6_h i:2', 'ax1 n j ax n2 x aU3 w O:3 s6 j A: N3 j aU4 w ax n4 f ax n1 ts6_h i:2', 'ax1 s6 H E2 ts6 j E3 n i:3 x aU3 w O:3 s6 j A: N3 j aU4 w ax n4 f ax n1 ts6_h i:2', 'eI4 n j ax n2 x aU3 w O:3 s6 j A: N3 j aU4 w ax n4 f ax n1 ts6_h i:2', 'ax4 n i:3 x aU3 w O:3 s6 j A: N3 j aU4 w ax n4 f ax n1 ts6_h i:2', 'ax4 m eI2 n i:3 x aU3 w O:3 s6 j A: N3 j aU4 w ax n4 f ax n1 ts6_h i:2', 'ax4 s6 H E2 ts6 j E3 n j ax n2 x aU3 w O:3 s6 j A: N3 j aU4 w ax n4 f ax n1 ts6_h i:2', 'ax4 s6 H E2 ts6 j E3 n i:3 x aU3 w O:3 s6 j A: N3 j aU4 w ax n4 f ax n1 ts6_h i:2', 'spn n i:3 x aU3 w O:3 s6 j A: N3 j aU4 w ax n4 f ax n1 ts6_h i:2'], 
#  'id': 369, 
#  'bopomo_sequence_list': ['ㄟˋ ㄋㄧˇ ㄏㄠˇ ㄨㄛˇ ㄒㄧㄤˇ ㄧㄠˋ ㄨㄣˋ ㄈㄣ ㄑㄧˊ', 'ㄜ ㄋㄧˇ ㄏㄠˇ ㄨㄛˇ ㄒㄧㄤˇ ㄧㄠˋ ㄨㄣˋ ㄈㄣ ㄑㄧˊ', 'ㄜ ㄋㄧㄣˊ ㄏㄠˇ ㄨㄛˇ ㄒㄧㄤˇ ㄧㄠˋ ㄨㄣˋ ㄈㄣ ㄑㄧˊ', 'ㄜ ㄒㄩㄝˊ ㄐㄧㄝˇ ㄋㄧˇ ㄏㄠˇ ㄨㄛˇ ㄒㄧㄤˇ ㄧㄠˋ ㄨㄣˋ ㄈㄣ ㄑㄧˊ', 'ㄟˋ ㄋㄧㄣˊ ㄏㄠˇ ㄨㄛˇ ㄒㄧㄤˇ ㄧㄠˋ ㄨㄣˋ ㄈㄣ ㄑㄧˊ', 'ㄜˋ ㄋㄧˇ ㄏㄠˇ ㄨㄛˇ ㄒㄧㄤˇ ㄧㄠˋ ㄨㄣˋ ㄈㄣ ㄑㄧˊ', 'ㄜˋ ㄇㄟˊ ㄋㄧˇ ㄏㄠˇ ㄨㄛˇ ㄒㄧㄤˇ ㄧㄠˋ ㄨㄣˋ ㄈㄣ ㄑㄧˊ', 'ㄜˋ ㄒㄩㄝˊ ㄐㄧㄝˇ ㄋㄧㄣˊ ㄏㄠˇ ㄨㄛˇ ㄒㄧㄤˇ ㄧㄠˋ ㄨㄣˋ ㄈㄣ ㄑㄧˊ', 'ㄜˋ ㄒㄩㄝˊ ㄐㄧㄝˇ ㄋㄧˇ ㄏㄠˇ ㄨㄛˇ ㄒㄧㄤˇ ㄧㄠˋ ㄨㄣˋ ㄈㄣ ㄑㄧˊ', 'spn ㄋㄧˇ ㄏㄠˇ ㄨㄛˇ ㄒㄧㄤˇ ㄧㄠˋ ㄨㄣˋ ㄈㄣ ㄑㄧˊ'],
#  'pinyin_sequence_list': ['ei4 ni3 hao3 wo3 xiang3 yao4 wen4 fen1 qi2', 'e1 ni3 hao3 wo3 xiang3 yao4 wen4 fen1 qi2', 'e1 nin2 hao3 wo3 xiang3 yao4 wen4 fen1 qi2', 'e1 xue2 jie3 ni3 hao3 wo3 xiang3 yao4 wen4 fen1 qi2', 'ei4 nin2 hao3 wo3 xiang3 yao4 wen4 fen1 qi2', 'e4 ni3 hao3 wo3 xiang3 yao4 wen4 fen1 qi2', 'e4 mei2 ni3 hao3 wo3 xiang3 yao4 wen4 fen1 qi2', 'e4 xue2 jie3 nin2 hao3 wo3 xiang3 yao4 wen4 fen1 qi2', 'e4 xue2 jie3 ni3 hao3 wo3 xiang3 yao4 wen4 fen1 qi2', 'spn ni3 hao3 wo3 xiang3 yao4 wen4 fen1 qi2']}
