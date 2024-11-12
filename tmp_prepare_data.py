import os
from datasets import load_dataset
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

# dataset = load_dataset("parquet", data_files={'train': [f'../ReCMLLM_outputs/HuggingFaceFW/fineweb/sample/10BT/{i:03d}_00000.parquet' for i in range(2)]})
# print(dataset)


# num = 5
# for ind,example in enumerate(dataset['train']):
#     print(example['text'])
#     # 问题大概也能占和文本一样多的token
#     estimated_total_length = example['token_count']*2 + example['token_count']
#     print(estimated_total_length)
#     if num == ind:
#         break

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)

def load_json(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj

def logINFO(info):
    with open('prepare_data.txt', 'a') as f:
        f.write(info)

def get_fixed_length(token_count):
    return min(((token_count*3)//510+1)*510,131580)

def init_data(cfg):
    data={length:[] for length in range(510,131582,510)}
    # for length in data.keys():
    #     save_path = cfg["save_path"]+f"/{length}.json"
    #     if os.path.exists(save_path):
    #         data[length]=load_json(save_path)
    #         print(f"loading {length}.json...")
    return data

def save_data(data, cfg):
    for length in data.keys():
        # 最后决定还是进行分片，下次可以把这个改成多进程。
        save_path = cfg["save_path"]+f"/{length}_{cfg['finished_file_index']}.json"
        save_json(data[length], save_path)
        data[length]=[]
    save_json(cfg, cfg["save_path"]+f"/state.json")
            
def get_sorted_files_full_path(directory):
    # 列出目录中的所有文件，并获取它们的完整路径
    files_with_path = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    # 对文件列表进行排序
    sorted_files_with_path = sorted(files_with_path)
    return sorted_files_with_path

def prepare_text():
    cfg = load_json("prepare_state.json")
    state_path = cfg["save_path"]+f"/state.json"
    if os.path.exists(state_path):
        cfg = load_json(state_path)
    data = init_data(cfg)
    # print(cfg)
    # print(data)
    
    files = get_sorted_files_full_path(cfg["data_path"])
    unfinished_files = []
    if "finished_file" in cfg:
        have_seen_finished_file = False
        for f in files:
            
            if have_seen_finished_file:
                unfinished_files.append(f)

            if f==cfg["finished_file"]:
                have_seen_finished_file = True
    else:
        unfinished_files = files.copy()

    file_index = 0
    if "finished_file_index" in cfg:
        file_index = cfg["finished_file_index"]+1
    for data_path in tqdm(unfinished_files):
        print(f"processing {data_path}...")
        dataset = load_dataset("parquet", data_files={'train':data_path})
        for example in tqdm(dataset['train']):
            data[get_fixed_length(example['token_count'])].append(example['text'])


        cfg["finished_file"] = data_path
        cfg["finished_file_index"] = file_index
        save_data(data,cfg)
        file_index += 1

###################################################################################################################

def get_multi_page_text(text_ids, tokenizer):
    cursor = 0
    cnt = 0
    whole_text = []
    multi_pages = []
    while cursor<len(text_ids):
        cnt+=1
        head_ids = tokenizer(f"<Page {cnt}>", add_special_tokens=False)["input_ids"]
        tail_ids = tokenizer(f"<\Page {cnt}>", add_special_tokens=False)["input_ids"]
        leave_len = 102 - len(head_ids) - len(tail_ids) # one page have 102 tokens
        whole_text += (head_ids + text_ids[cursor:cursor+leave_len] + tail_ids, text_ids[cursor:cursor+leave_len])
        multi_pages.append(text_ids[cursor:cursor+leave_len])
        cursor+=leave_len
    return whole_text, multi_pages


def get_question_and_answer(multi_pages, tokenizer):

    def get_snippet_with_page_id(multi_pages, tokenizer):
        # 随机获取一段长度为8~32的snippet，同时获取它所在的页

        snippet_len = random.randint(8,32)
        begin_page_id = random.randint(0, len(multi_pages)-1)

        # if len(multi_pages[begin_page_id])<snippet_len: # last page maybe no enough tokens
        #     if begin_page_id == 0:
        #         return (multi_pages[begin_page_id], [begin_page_id])

        #     num_of_page = len(multi_pages[begin_page_id])
        #     leave_num = snippet_len-num_of_page
        #     return (multi_pages[begin_page_id-1][-leave_num:]+multi_pages[begin_page_id],[begin_page_id-1,begin_page_id])
        
        begin_index = random.randint(0, len(multi_pages[begin_page_id])-1)
        if begin_page_id==len(multi_pages)-1 and begin_index+snippet_len>len(multi_pages[begin_page_id]):
            if begin_page_id == 0:  # only one page
                begin_index = random.randint(0,max(0, len(multi_pages[begin_page_id])-snippet_len) )
                return (multi_pages[begin_page_id][begin_index:begin_index+snippet_len], [begin_page_id])

            if snippet_len<=len(multi_pages[begin_page_id]): # enough tokens in last page.
                begin_index = random.randint(0, len(multi_pages[begin_page_id])-snippet_len)
                return (multi_pages[begin_page_id][begin_index:begin_index+snippet_len], [begin_page_id])

            # not enough , but over one page.
            leave_num = snippet_len-len(multi_pages[begin_page_id])
            return (multi_pages[begin_page_id-1][-leave_num:]+multi_pages[begin_page_id],[begin_page_id-1,begin_page_id])


        snippet = multi_pages[begin_page_id][begin_index:begin_index+snippet_len]
        if len(snippet)<snippet_len:
            snippet += multi_pages[begin_page_id+1][:len(snippet)-snippet_len]
            return (snippet, [begin_index, begin_index+1])

        return (snippet, [begin_index])





    def get_q1(multi_pages, tokenizer):
        question = '''On which page does "<snippet>" appear?'''





def to_tensor(text, tokenizer, length):
    text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(text_ids)>(length*4)//3:
        text_ids = text_ids[:(length*4)//3]
        logINFO(f"truncate {text} to {length}")

    whole_text, multi_pages = get_multi_page_text(text_ids,tokenizer)


    # QAs = [get_question_and_answer(multi_pages, tokenizer) for ]


    "User input will be repeated twice for better context understanding and generation.\n### First occurrence\n"


def prepare_data():
    cfg = load_json("prepare_state.json")
    state_path = cfg["data_save_path"]+f"/state.json"
    if os.path.exists(state_path):
        cfg = load_json(state_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'])
    
    files = get_sorted_files_full_path(cfg["save_path"])
    sorted_files_with_path = sorted(files, key=lambda file_path: (
        int(file_path.split('/')[-1].split('_')[0]),  # 提取长度并转换为整数
        int(file_path.split('/')[-1].split('_')[1].split('.')[0])  # 提取序号并转换为整数
    ))

    print(sorted_files_with_path)
    unfinished_texts = []
    if "finished_text" in cfg:
        have_seen_finished_text = False
        for f in sorted_files_with_path:
            
            if have_seen_finished_text:
                unfinished_texts.append(f)

            if f==cfg["finished_text"]:
                have_seen_finished_text = True
    else:
        unfinished_texts = sorted_files_with_path.copy()


    length = 510
    if "finished_length" in cfg:
        length = cfg["finished_length"]+510

    data = []
    num = (1000*1000*1000)//length  # total 1B token

    for data_path in tqdm(unfinished_texts):
        print(f"processing {data_path}...")
        texts = load_json(data_path)

        for text in tqdm(texts):

            data.append(to_tensor(text, tokenizer, length))

            # save
            if len(data)==num:

                torch.save(data, cfg["data_save_path"]+f"/{length}.pt")
                cfg["finished_length"] = length
                cfg["finished_text"] = data_path # if interrupt, ignore the remain text in the file, and use next file.
                save_json(cfg, cfg["data_save_path"]+f"/state.json")
                print(f"finish {cfg['finished_length']}.pt")

                data = []
                length += 510
                num = (1000*1000*1000)//length

                if length > 131580:
                    return


if __name__ == "__main__":
    prepare_text()
    print("finish BucketSort for Text")
    
