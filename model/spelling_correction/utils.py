import re
import logging
import sys

CHINESE_RE = r'[\u4e00-\u9fa5]'


class RunningAverage:
    def __init__(self):
        self.values = []

    def add(self, val):
        self.values.append(val)

    def add_all(self, vals):
        self.values += vals

    def get(self):
        return sum(self.values) / len(self.values)

    def flush(self):
        self.values = []


def wordize_and_map(text):
    words = []
    index_map_from_text_to_word = []
    index_map_from_word_to_text = []
    while len(text) > 0:
        match_space = re.match(r'^ +', text)
        if match_space:
            space_str = match_space.group(0)
            index_map_from_text_to_word += [None] * len(space_str)
            text = text[len(space_str):]
            continue

        match_en = re.match(r'^[a-zA-Z0-9]+', text)
        if match_en:
            en_word = match_en.group(0)

            word_start_pos = len(index_map_from_text_to_word)
            word_end_pos = word_start_pos + len(en_word)
            index_map_from_word_to_text.append((word_start_pos, word_end_pos))

            index_map_from_text_to_word += [len(words)] * len(en_word)

            words.append(en_word)
            text = text[len(en_word):]
        else:
            word_start_pos = len(index_map_from_text_to_word)
            word_end_pos = word_start_pos + 1
            index_map_from_word_to_text.append((word_start_pos, word_end_pos))

            index_map_from_text_to_word += [len(words)]

            words.append(text[0])
            text = text[1:]
    return words, index_map_from_text_to_word, index_map_from_word_to_text


def wordize_and_map_(text):
    words = []
    index_map_from_text_to_word = []
    index_map_from_word_to_text = []
    while len(text) > 0:
        match_space = re.match(r'^(\[PAD\]| )+', text)
        if match_space:
            space_str = match_space.group(0)
            index_map_from_text_to_word += [None] * len(space_str)
            text = text[len(space_str):]
            continue
        
        match_unk = re.match(r'\[UNK\]', text)
        if match_unk:
            unk_word = match_unk.group(0) # '[UNK]'
            
            word_start_pos = len(index_map_from_text_to_word)
            word_end_pos = word_start_pos + len(unk_word)
            index_map_from_word_to_text.append((word_start_pos, word_end_pos))
            
            index_map_from_text_to_word += [len(words)] * len(unk_word)
            
            words.append(unk_word)
            text = text[len(unk_word):]
            continue
        
        match_en = re.match(r'^[a-zA-Z0-9]+', text)
        if match_en:
            en_word = match_en.group(0)

            word_start_pos = len(index_map_from_text_to_word)
            word_end_pos = word_start_pos + len(en_word)
            index_map_from_word_to_text.append((word_start_pos, word_end_pos))

            index_map_from_text_to_word += [len(words)] * len(en_word)

            words.append(en_word)
            text = text[len(en_word):]
        else:
            word_start_pos = len(index_map_from_text_to_word)
            word_end_pos = word_start_pos + 1
            index_map_from_word_to_text.append((word_start_pos, word_end_pos))

            index_map_from_text_to_word += [len(words)]

            words.append(text[0])
            text = text[1:]
    return words, index_map_from_text_to_word, index_map_from_word_to_text


def tokenize_and_map(tokenizer, text):
    words, text2word, word2text = wordize_and_map(text)

    tokens = []
    index_map_from_token_to_text = []
    for word, (word_start, word_end) in zip(words, word2text):
        word_tokens = tokenizer.tokenize(word)

        if len(word_tokens) == 0 or word_tokens == ['[UNK]']:
            index_map_from_token_to_text.append((word_start, word_end))
            tokens.append('[UNK]')
        else:
            current_word_start = word_start
            for word_token in word_tokens:
                word_token_len = len(re.sub(r'^##', '', word_token))
                index_map_from_token_to_text.append(
                    (current_word_start, current_word_start + word_token_len))
                current_word_start = current_word_start + word_token_len
                tokens.append(word_token)

    index_map_from_text_to_token = text2word
    for i, (token_start, token_end) in enumerate(index_map_from_token_to_text):
        for token_pos in range(token_start, token_end):
            index_map_from_text_to_token[token_pos] = i

    return tokens, index_map_from_text_to_token, index_map_from_token_to_text

def tokenize_and_map_(tokenizer, text):
    words, text2word, word2text = wordize_and_map_(text)

    tokens = []
    index_map_from_token_to_text = []
    for word, (word_start, word_end) in zip(words, word2text):
        word_tokens = tokenizer.tokenize(word)

        if len(word_tokens) == 0 or word_tokens == ['[UNK]']:
            index_map_from_token_to_text.append((word_start, word_end))
            tokens.append('[UNK]')
        else:
            current_word_start = word_start
            for word_token in word_tokens:
                word_token_len = len(re.sub(r'^##', '', word_token))
                index_map_from_token_to_text.append(
                    (current_word_start, current_word_start + word_token_len))
                current_word_start = current_word_start + word_token_len
                tokens.append(word_token)

    index_map_from_text_to_token = text2word
    for i, (token_start, token_end) in enumerate(index_map_from_token_to_text):
        for token_pos in range(token_start, token_end):
            index_map_from_text_to_token[token_pos] = i

    return tokens, index_map_from_text_to_token, index_map_from_token_to_text


def load_config(config_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('__init__', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def get_logger(file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    output_file_handler = logging.FileHandler(file_path, encoding='utf-8')
    stdout_handler = logging.StreamHandler(sys.stdout)

    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    return logger
