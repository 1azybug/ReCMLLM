from transformers import AutoTokenizer
import torch
import json
import os

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)

def load_json(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj

cfg = load_json("prepare_state.json")
state_path = cfg["data_save_path"]+f"/state.json"
if os.path.exists(state_path):
    cfg = load_json(state_path)


tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'])


# data = torch.load(cfg["data_save_path"]+"/1020.pt")
data = torch.load(cfg["data_save_path"]+"/10200.pt")

cnt = 1
for example in data:
    # print(example['input_ids'].tolist())
    # print(example['labels'].tolist())

    input_ids_str = str()
    # input_str = str()
    for idx in example['input_ids'].tolist():
        token = tokenizer._convert_id_to_token(idx)
        s = f"""{idx}({token})"""
        input_ids_str += "{:>30}".format(s)
        # input_str += "{:>15}".format(tokenizer._convert_id_to_token(idx))

    labels_str = str()
    # lab_str = str()
    valid_label = []
    for idx in example['labels'].tolist():
        if idx == -100:
            token = " "
        else:
            token = tokenizer._convert_id_to_token(idx)
            valid_label.append(idx)
        s = f"""{idx}({token})"""
        labels_str += "{:>30}".format(s)
        # print(idx)
        # if idx == -100:
        #     lab_str += "{:>15}".format("-100")
        # else:
        #     lab_str += "{:>15}".format(tokenizer._convert_id_to_token(idx))

    
    


    with open("../ReCMLLM_outputs/info.jsonl", 'a') as f:
        # f.write(input_ids_str+"\n")
        # f.write(labels_str+"\n")
        # f.write(json.dumps(input_str.replace("\u2581",""))+"\n")
        # f.write(json.dumps(lab_str.replace("\u2581",""))+"\n")
        f.write(tokenizer.decode(example['input_ids'].tolist(), skip_special_tokens=False)+"\n")
        f.write(tokenizer.decode(valid_label, skip_special_tokens=False)+"\n")
        # f.write(json.dumps(tokenizer.decode(example['labels'].tolist().remove(-100), skip_special_tokens=True))+"\n")
        

    # print(tokenizer.decode(example['input_ids'].tolist(), skip_special_tokens=False))
    print(example['input_ids'].shape)
    print(example['labels'].shape)
    print(example["input_ids"][505:515])
    print(example["labels"][505:515])

    if cnt>100:
        break
    cnt+=1

# python check_data.py