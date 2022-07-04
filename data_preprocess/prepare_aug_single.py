import dimsim
import random
from pypinyin import pinyin, Style


def dimsim_get_candidates(string):
    candidates = []
    
    try:
        if len(string) == 1:
            candidates = [x for x in all_char if dimsim.get_distance(string, x) < 0.5]
        elif len(string) == 2:
            candidates = dimsim.get_candidates(string, mode="traditinoal", theta=1)
        else:
            candidates = []
    except:
        candidates = []
    
    out = [string] + candidates
    
    return out

def create_pinyin(text):
    text = ''.join(text.split())
    if text:
        pinyin_res = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
        return ' '.join([x for xl in pinyin_res for x in xl])
    return None

def get_sentence_candidates(seg):
    gt = ''.join(seg)
    dimsim_list = [dimsim_get_candidates(x) for x in seg]
    n = len(dimsim_list)
    n_sentence = 10
    n_choose = n // 4 + 1
    
    sentence_set = set()
    for i in range(n_sentence):
        id_select = random.sample(range(n), n_choose)
        sentence_set.add(' '.join([random.choice(v) if i in id_select else v[0] for i, v in enumerate(dimsim_list)]))
    
    pinyin_list = list(filter(None, [create_pinyin(text) for text in sentence_set]))
    
    out = {'ground_truth_sentence': gt, 'sentence_list': list(sentence_set), 'pinyin_sequence_list': pinyin_list}
    return out


## Example:

# Input: 空白斷句的 ground_truth_sentence
data = '從 一 個 分產 鬧劇 看 一般人 面對 繼承 常 搞錯 的 四 觀念'

# Output: dictionary 如下
get_sentence_candidates(data.split())
# {'ground_truth_sentence': '從一個分產鬧劇看一般人面對繼承常搞錯的四觀念',
#  'sentence_list': ['從 一 個 分產 牢記 看 一般人 面對 繼承 常 搞錯 的 四 觀念',
#   '從 一 個 分產 鳥擊 看 一般人 面對 繼承 常 搞錯 的 四 關連',
#   '從 一 個 奮戰 牢記 看 一般人 面對 繼承 常 搞錯 的 四 貫連',
#   '從 一 個 分產 鬧劇 看 一般人 面對 繼承 常 搞錯 的 四 貫連',
#   '從 一 個 分產 鬧劇 看 一般人 面對 集成 常 搞錯 的 四 觀念',
#   '從 一 個 分產 鬧劇 看 一般人 面對 繼承 常 搞錯 的 四 觀念'],
#  'pinyin_sequence_list': ['cong2 yi1 ge4 fen1 chan3 lao2 ji4 kan4 yi4 ban1 ren2 mian4 dui4 ji4 cheng2 chang2 gao3 cuo4 de5 si4 guan1 nian4',
#   'cong2 yi1 ge4 fen1 chan3 niao3 ji1 kan4 yi4 ban1 ren2 mian4 dui4 ji4 cheng2 chang2 gao3 cuo4 de5 si4 guan1 lian2',
#   'cong2 yi1 ge4 fen4 zhan4 lao2 ji4 kan4 yi4 ban1 ren2 mian4 dui4 ji4 cheng2 chang2 gao3 cuo4 de5 si4 guan4 lian2',
#   'cong2 yi1 ge4 fen1 chan3 nao4 ju4 kan4 yi4 ban1 ren2 mian4 dui4 ji4 cheng2 chang2 gao3 cuo4 de5 si4 guan4 lian2',
#   'cong2 yi1 ge4 fen1 chan3 nao4 ju4 kan4 yi4 ban1 ren2 mian4 dui4 ji2 cheng2 chang2 gao3 cuo4 de5 si4 guan1 nian4',
#   'cong2 yi1 ge4 fen1 chan3 nao4 ju4 kan4 yi4 ban1 ren2 mian4 dui4 ji4 cheng2 chang2 gao3 cuo4 de5 si4 guan1 nian4']}