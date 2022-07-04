model_source = 'ckiplab/bert-base-chinese'
max_len = 202
num_workers = 2
batch_size = 16

# for training
manual_seed = 1313
exp_name = 'ckipbert_detection_same_length'
train_json = '../../../data/train_same_length_0617.json'
valid_json = '../../../data/valid_same_length_0617.json'
lr = 2e-5
val_interval = 800
num_iter = 160000
