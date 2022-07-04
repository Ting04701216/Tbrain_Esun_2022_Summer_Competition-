from src.dataset import TypoDataset
from src.detector_utils import obtain_valid_detection_preds
import torch
from torch.utils.data import DataLoader

def predict_spelling_correction(texts, model, bopomofo_dict, tokenizer, device):
    dataset = TypoDataset(tokenizer, texts, bopomofo_dict=bopomofo_dict, bopomofos=None, max_length=202, for_train=False, for_detect=False)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        collate_fn=dataset.create_mini_batch,
        num_workers=1
    )
    
    model.eval()
    
    result = []
    
    for data in dataloader:
        input_ids, token_type_ids, attention_mask, bopomofo_ids = [d.to(device) for d in data[:4]]
        
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                bopomofo_ids=bopomofo_ids
            )
            
            output_ids = logits.argmax(dim=-1)
        output_ids = output_ids.cpu().detach().tolist()
        res = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        res = [''.join(x.split()) for x in res]
        result.extend(res)
    return result

def process(input, output):
    return input if output == '' else output

def predict_typo(texts, model, bopomofo_dict, tokenizer, device, threshold=0.5):
    
    dataset = TypoDataset(tokenizer, texts, bopomofo_dict=bopomofo_dict, bopomofos=None, max_length=202, for_train=False, for_detect=False)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        collate_fn=dataset.create_mini_batch,
        num_workers=1
    )
    
    model.eval()

    detected_char_positions_collect = []

    for data in dataloader:
        input_ids, token_type_ids, attention_mask = [d.to(device) for d in data[:3]]
        infos = data[-1]
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )

            preds = obtain_valid_detection_preds(outputs.logits, input_ids, tokenizer, threshold=threshold)

        for pred_seq, info in zip(preds, infos):
            token2text = info['token2text']
            detected_char_positions = []
            for i, is_detected in enumerate(pred_seq):
                if is_detected == 1:
                    token_position = i - 1  # remove [CLS]
                    start, end = token2text[token_position]
                    detected_char_positions.append(start)
            detected_char_positions_collect.append(detected_char_positions)

    return detected_char_positions_collect
