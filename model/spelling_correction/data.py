import json
from tqdm.notebook import tqdm
from prepare_input_single import prepare_text_input_same_length

with open('../../../data/train_aug.json') as jfile:
    train_aug = json.load(jfile)

train_align = []

for num, sample in tqdm(enumerate(train_data_aug)):
    tmp = prepare_text_input_same_length(sample)
    if tmp:
        for i, samp_dict in enumerate(tmp):
            samp_dict['id'] = 'id_' + str(num) + '_ord_' + str(i)
        train_align.extend(tmp)
        
for_train = []
for_valid = []
for i, data in enumerate(train_data_aug):
    if 59000 <= i < 60000:
        for_valid.append(data)
    else:
        for_train.append(data)
        
with open('../../../data/train_same_length_0617.json', 'w', encoding='utf8') as jfile:
    json.dump(for_train, jfile)
with open('../../../data/valid_same_length_0617.json', 'w', encoding='utf8') as jfile:
    json.dump(for_valid, jfile)
