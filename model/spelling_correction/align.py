import re, sys

py_dict = {}
def get_dict(dict_path):
    for line in open(dict_path, 'r', encoding='utf-8'):
        char, pinyin = line.strip().split('\t')
        py_dict[char] = pinyin
get_dict('pinyin_dict.txt')


def get_pinyin(text):
    if text == '<void>':
        return ''
    text = re.sub('([\u4e00-\u9fff])', r' \1 ', text)
    text = re.sub('\s+', ' ', text)
    text_list = text.strip().split(' ')
    for i in range(len(text_list)):
        word = text_list[i]
        if word == '▁':
            text_list[i] = '_'
        else:
            if len(word) == 1 and word >= '\u4e00' and word <= '\u9fff':
                if word in py_dict:
                    text_list[i] = py_dict[word]
                else:
                    text_list[i] = 'unk'
    return ''.join(text_list)

def edit_matrix(query, reference):
    matrix = [[0] * (len(query) + 1) for _ in range(len(reference) + 1)]
    j_count = 0
    for j in range(1, len(query) + 1):
        if query[j-1] == '<void>':
            j_count += 1
        matrix[0][j] = j - j_count
    for i in range(len(reference) + 1):
        matrix[i][0] = i
    for i in range(1, len(reference) + 1):
        for j in range(1, len(query) + 1):
            if query[j-1] == '<void>':
                matrix[i][j] = matrix[i][j-1]
            elif query[j-1] == reference[i-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(matrix[i-1][j-1]+1, matrix[i-1][j]+1, matrix[i][j-1]+1)
    return matrix

def get_raw_align_script(matrix, query, reference):
    j = len(query)
    i = len(reference)
    edit_script = []
    score = [sys.maxsize for i in range(3)]
    while i>0 or j>0:
        #del
        if i>0 and matrix[i-1][j] == matrix[i][j] - 1:
            score[0] = len(get_pinyin(reference[i-1]))
        #ins
        if j>0 and matrix[i][j-1] == matrix[i][j] - 1:
            score[1] = len(get_pinyin(query[j-1]))
        #sub
        if i>0 and j>0 and matrix[i-1][j-1] == matrix[i][j]-1:
            score[2] = edit_matrix(get_pinyin(query[j-1]), get_pinyin(reference[i-1]))[-1][-1]
        idx = score.index(min(score))

        #identical
        if i>0 and j>0 and matrix[i-1][j-1] == matrix[i][j] and score[idx] == sys.maxsize:
            edit_script.append(('EQUAL', query[j-1], reference[i-1]))
            i -= 1
            j -= 1
        #del
        elif idx ==0:
            edit_script.append(('DEL', '<void>', reference[i-1]))
            i -= 1
        #ins
        elif idx == 1:
            edit_script.append(('INS', query[j-1], '<void>'))
            j -= 1
        #sub
        elif idx == 2:
            edit_script.append(('SUB', query[j-1], reference[i-1]))
            i -= 1
            j -= 1
        else:
            raise Exception('unexpected edit matrix')
        score = [sys.maxsize for i in range(3)]
    edit_script.reverse()
    return edit_script

def align(query, reference, align_recog_list):
    matrix = edit_matrix(query, reference)
    raw_align_script = get_raw_align_script(matrix, query, reference)
    # print(f'{raw_align_script = }')
    align_query = []
    align_reference = []
    for i in range(len(raw_align_script)):
        e = raw_align_script[i]
        align_query.append(e[1])
        align_reference.append(e[2])
        if e[2] == '<void>' and e[0] == 'INS':
            for j in range(len(align_recog_list)):
                if len(align_recog_list[j]) != 0:
                    align_recog_list[j].insert(i, '<void>')
    return align_query, align_reference, align_recog_list

# ed2
def gen_align(recog_list, tokenizer):
    recog_tokenize = [tokenizer.tokenize(recog) for recog in recog_list]
    recog_len = [len(recog) for recog in recog_tokenize]
    max_idx = recog_len.index(max(recog_len))
    align_recog_list = [[] for _ in recog_list]
    align_recog_list[max_idx] = recog_tokenize[max_idx]
    for i in range(len(recog_list)):
        if i != max_idx:
            align_query, align_reference, align_recog_list = align(recog_tokenize[i], align_recog_list[max_idx], align_recog_list)
            align_recog_list[i] = align_query
    return align_recog_list