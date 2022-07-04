import os
import re
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, AutoModelForSequenceClassification, BertForTokenClassification, BertForMaskedLM

from classify_news.src.classify_news import infer, preprocess

from fairseq.models.transformer import TransformerModel
from pinyin_translation.src import translate_utils

from spelling_correction.src import pred_correction as sc_infer
from spelling_correction.src.dimsim import prepare_dimsim
from spelling_correction.src.module import PhoneticMLM


####### CONFIG ##########################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
pretrained_model = 'ckiplab/bert-base-chinese'
alpha = 8
detector_threshold = 0.5
#########################################

####### LOAD MODELS ##########################
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

## pinyin translation
checkpoint_name = 'checkpoint_best.pt'
path_esun = 'fairseq/pinyin2chinese/nmt/models/all/checkpoints'
pinyin2zh_esun = TransformerModel.from_pretrained(model_name_or_path=path_esun, checkpoint_file=checkpoint_name, data_name_or_path='../../../data/all/data-bin')
path_news = 'fairseq/pinyin2chinese/nmt/models/news/checkpoints'
pinyin2zh_news = TransformerModel.from_pretrained(model_name_or_path=path_news, checkpoint_file=checkpoint_name, data_name_or_path='../../../data/news/data-bin')

## spelling correction
bopomofo_dict, bopomofo_to_dist = prepare_dimsim(tokenizer.vocab)
detector = BertForTokenClassification.from_pretrained(pretrained_model, return_dict=True, num_labels=2)
detector.load_state_dict(torch.load('model/same_length_best_f1.pth', map_location=device))
detector.to(device)
mlm = BertForMaskedLM.from_pretrained(pretrained_model)
mlm.to(device)
unknown_bopomofo_id = bopomofo_dict['UNK']
phonetic_mlm_model = PhoneticMLM(detector, mlm, tokenizer, bopomofo_to_dist, unknown_bopomofo_id, alpha=alpha, detector_threshold=detector_threshold)
#########################################


def predict(sentence_list, phoneme_sequence_list):
    """ Predict model result.

    @param:
        sentence_list (list): an list of sentence sorted by probability.
        phoneme_sequence_list (list): an list of phoneme sequence sorted by probability.
    @returns:
        prediction (str): a sentence.
    """
    
    ## 取 sentence_list 第一句判斷屬於新聞或口語對話
    is_news = classify.infer(classify.preprocess([re.sub('[^\u4E00-\u9FFF]', '', sentence_list[0])], tokenizer), model_classify, device)[0]
    
    ## pinyin翻譯
    model_pinyin2zh = pinyin2zh_news if is_news == 1 else pinyin2zh_esun
    phone2zh_translate = translate_utils.infer(sentence_list, phoneme_sequence_list, is_news, model_pinyin2zh)
    
    ## 糾錯模型
    texts = [re.sub('[^\u4E00-\u9FFF]', '', x) for x in sentence_list]
    det_output = sc_infer.predict_spelling_correction(texts, phonetic_mlm_model, bopomofo_dict, tokenizer, device)
    det_output.append(phone2zh_translate)  # 把翻譯結果放進來一起當作候選句子
    det_output = list(dict.fromkeys(det_output))  # 去重複
    det_out = sc_infer.predict_typo(det_output, detector, bopomofo_dict, tokenizer, device, threshold=detector_threshold)  # 再套一次糾錯
    det_count = [len(x) for x in det_out]
    answer = det_output[np.argmin(det_count)]  # 取預測錯誤字數最少者做為結果輸出
    
    return answer


## Example:

# Input: sentence_list and phoneme_sequence_list
data = {'sentence_list': ['做 個 聰明 用 路人', '作 個 聰明 用 路人'],
        'phoneme_sequence_list': ['ts w O:4 k ax5 ts_h w ax N1 m j ax N2 H ax N4 l u:4 zz ax n2', 'ts w O:4 k ax5 ts_h w ax N1 m j ax N2 H ax N4 l u:4 zz ax n2']}

# Output: 
answer = predict(data['sentence_list'], data['phoneme_sequence_list'])
print(answer)  # '做個聰明用路人'
