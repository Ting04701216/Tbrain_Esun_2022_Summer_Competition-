import copy
import re
from src.align import *
from transformers import BertTokenizerFast
from typing import List
import pandas as pd
import json

tokenizer = BertTokenizerFast.from_pretrained('ckiplab/bert-base-chinese')

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
            # '○' -> '零'
            預計今天恐湧現一五○萬人潮 -> 零
            預估稅損約四○億元 -> 零 (新聞源應該是阿拉伯數字直翻)
            
            # 'e-mail' -> ''
            
            # '[^\u4e00-\u9fff]' -> ''
            學生的回答不一樣了⋯⋯ -> 學生的回答不一樣了
            
            # <htr> -> [UNK]?
            TBD
            
            # <void> -> [PAD]? ##
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
            
        # sents = gen_align(sents + list(set(to_append)), tokenizer)
        sents = sents + list(set(to_append))
        self.sample['sentence_list'] = sents
        
        # return sample_
    
    # def process_align(self):
    #     all_hypo_line_target = [None] * len(train_aug)
    
    def clean_data(self, filter_num=None):
        """main

        Examples:
            >>> DC = DataCleaning(train[369])
            >>> DC.clean_data()
        """
        
        self.process_ground_truth()
        self.process_sent_list(filter_num)
        return self.sample


def get_substitute(correct, wrong):
    subs = []
    for i, (cor, wrn) in enumerate(zip(correct, wrong)):
        # wrong = 
        # correct = 
        # start = 
        if cor != wrn:
            sub_word = [wrn, cor, i]
            subs.append(sub_word)
    
    return subs

def void2pad(tokens: List[str]):
    return [x.replace('<void>', '[PAD]') if x == '<void>' else x for x in tokens]
def void2underscore(tokens: List[str]):
    return [x.replace('<void>', '_') if x == '<void>' else x for x in tokens]

def bake_train_input(gt_align, sent_align):
    
    result_full = []
    true_text = ''.join(gt_align)
    
    for sent in sent_align:
        result = {}
        result['text'] = re.sub('#', '', ''.join(sent))
        # result['text'] = [''.join(sent) for sent in sent_align]
        result['substitute_errors'] = get_substitute(correct=gt_align, wrong=sent)
        # result['substitute_errors'] = [get_substitute(correct=gt_align, wrong=sent) for sent in sent_align]
        
        result['true_text'] = true_text
        result_full.append(result)
    
    return result_full

#############################################################
# for only same length pairs
#############################################################

def prepair_train_gt_sent(sample):
    DC = DataCleaning(sample)
    result = DC.clean_data(filter_num=None)
    all_sent = [result['ground_truth_sentence']] + result['sentence_list']
    aligns = gen_align(all_sent, tokenizer)
    # aligns_to_pad = [void2pad(tokens) for tokens in aligns]
    aligns_to_pad = [void2underscore(tokens) for tokens in aligns]
    
    if len({len(toks) for toks in aligns_to_pad}) != 1:
        raise ValueError("not same length")
    
    gt_align, *sent_align = aligns_to_pad
    
    return result['ground_truth_sentence'], result['sentence_list'], gt_align, sent_align

def count_wo_us(ll):
    remain = [l for l in ll if l != '_']
    return remain

def get_same_length_sent(sample):
    gt, sents, gt_align, sent_align = prepair_train_gt_sent(sample)
    # gt_us_inds = [i for i, x in enumerate(gt_align) if x == "_"]
    gt_align_true_len = len([x for x in gt_align if x != "_"])
    sent_align_true_len = [len(count_wo_us(sent)) for sent in sent_align]
    sent_align_remain_ind = [i for i, x in enumerate(sent_align_true_len) if x == gt_align_true_len]
    sent_align_remain = [sent for i, sent in enumerate(sent_align) if i in sent_align_remain_ind]
    sents_remain = [sent for i, sent in enumerate(sents) if i in sent_align_remain_ind]

    return gt, sents_remain, gt_align, sent_align_remain

def remove_underscore(gt, sents, gt_align_, sent_align_):
    gt_us_inds = [i for i, x in enumerate(gt_align_) if x == "_"]
    gt_align_remain = [x for x in gt_align_ if x != "_"]
    sent_remain = []
    sent_align_remain = []
    for sent, sent_al in zip(sents, sent_align_):
        if gt_us_inds == [i for i, x in enumerate(sent_al) if x == "_"]:
            sent_align_remain.append([x for x in sent_al if x != "_"])
            sent_remain.append(''.join(sent.split()))
    
    if sent_align_remain:
        
        # only keep unique sents / sent_aligns
        length_of_sent = len(sent_remain)
        if length_of_sent > 1:
            unq_ind = [sent_remain.index(x) for x in set(sent_remain)]
            if len(unq_ind) < length_of_sent:
                sent_remain = [sr for i, sr in enumerate(sent_remain) if i in unq_ind]
                sent_align_remain = [sr for i, sr in enumerate(sent_align_remain) if i in unq_ind]
        
        assert len(sent_remain) == len(sent_align_remain)
        return gt, sent_remain, gt_align_remain, sent_align_remain
    
def handle_unk(gt, sent_, gt_align_, sent_align_):
    gt_handled = [g if gl == '[UNK]' else gl for g, gl in zip(gt, gt_align_)]
    sent_handled = []
    for s, sl in zip(sent_, sent_align_):
        sent_handled.append([ss if ssl == '[UNK]' else ssl for ss, ssl in zip(s, sl)])
    
    return gt_handled, sent_handled


def prepare_text_input_same_length(sample):
        tmp = remove_underscore(*get_same_length_sent(sample))
        if tmp:
            return bake_train_input(*handle_unk(*tmp))
        
## Example:

# Input: 含 ground_truth_sentence, sentence_list之資料
with open('../../../data/train_all.json') as jfile:
    train_all = json.load(jfile)
test = train_all[369]


# Output: 
# prepare_text_input_same_length: list[Dict]

print(prepare_text_input_same_length(test))
# [{'text': '呃沒你好我想要問分期', 'substitute_errors': [['呃', '欸', 0], ['沒', '喂', 1]], 'true_text': '欸喂你好我想要問分期'}]
