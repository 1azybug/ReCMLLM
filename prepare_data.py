import os
from datasets import load_dataset
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import random
import torch

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
    with open('../ReCMLLM_outputs/prepare_data.txt', 'a') as f:
        f.write(info+"\n")

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
        whole_text += head_ids + text_ids[cursor:cursor+leave_len] + tail_ids
        multi_pages.append(text_ids[cursor:cursor+leave_len])
        cursor+=leave_len
    return whole_text, multi_pages


def get_stat(multi_pages):
    token_pids = {}
    tokens = []
    for i,page in enumerate(multi_pages):
        # pid is the index of each page.
        pid = i+1
        for token in page:
            if token not in token_pids:
                token_pids[token] = []
                tokens.append(token)
            token_pids[token].append(pid)

    frequency_dict = {}
    freqs = []
    max_freq = 0
    min_freq = 132000
    for token, pids in token_pids.items():
        count = len(pids)
        if count not in frequency_dict:
            frequency_dict[count] = []
            freqs.append(count)
            max_freq = max(max_freq, count)
            min_freq = min(min_freq, count)
        frequency_dict[count].append(token)

    return (token_pids, tokens, frequency_dict, freqs, max_freq, min_freq)


def get_question_and_answer(multi_pages, tokenizer, other_multi_pages, stat, usage=None):

    token_pids, tokens, frequency_dict, freqs, max_freq, min_freq = stat
            

    def get_snippet_with_page_id(multi_pages, tokenizer, min_len=8, max_len=32):
        # 随机获取一段长度为min_len~max_len的snippet，同时获取它所在的页

        snippet_len = random.randint(min_len,max_len)
        begin_page_id = random.randint(0, len(multi_pages)-1) # ensure the prob of page id is the same.
        
        begin_index = random.randint(0, len(multi_pages[begin_page_id])-1)
        if begin_page_id==len(multi_pages)-1 and begin_index+snippet_len>len(multi_pages[begin_page_id]): #last page and not enough token 
            if begin_page_id == 0:  # only one page, so random index in valid length
                begin_index = random.randint(0,max(0, len(multi_pages[begin_page_id])-snippet_len) )
                return (multi_pages[begin_page_id][begin_index:begin_index+snippet_len], [begin_page_id])

            if snippet_len<=len(multi_pages[begin_page_id]): # enough tokens in last page.
                return (multi_pages[begin_page_id][-snippet_len:], [begin_page_id])

            # not enough , but over one page.
            leave_num = snippet_len-len(multi_pages[begin_page_id])
            return (multi_pages[begin_page_id-1][-leave_num:]+multi_pages[begin_page_id],[begin_page_id-1,begin_page_id])


        snippet = multi_pages[begin_page_id][begin_index:begin_index+snippet_len]
        if len(snippet)<snippet_len: # not enough token, so is not the last page
            snippet += multi_pages[begin_page_id+1][:snippet_len-len(snippet)]
            if len(snippet)==snippet_len: # enough token in next page
                return (snippet, [begin_page_id, begin_page_id+1])
            else: # next page not enough token
                leave_num = snippet_len-len(multi_pages[begin_page_id+1])
                return (multi_pages[begin_page_id][-leave_num:]+multi_pages[begin_page_id+1],[begin_page_id,begin_page_id+1])                

        return (snippet, [begin_page_id])


    def get_q0(multi_pages, tokenizer, other_multi_pages):
        # Q: Dose "<snippet>" appear in this text? 
        # A: <Yes/No>.
        if random.random()<0.5:
            snippet, page_ids = get_snippet_with_page_id(multi_pages, tokenizer)
            snippet = tokenizer.decode(snippet, skip_special_tokens=True)
            question = f'''Dose "{snippet}" appear in this text?'''
            answer = "Yes."
            return (0, question, answer)
        else:
            snippet, page_ids = get_snippet_with_page_id(other_multi_pages, tokenizer)
            snippet = tokenizer.decode(snippet, skip_special_tokens=True)
            question = f'''Dose "{snippet}" appear in this text?'''
            answer = "No."
            return (0, question, answer)



    def get_q1(multi_pages, tokenizer, other_multi_pages):
        # Q: On which page (segment) does "<snippet>" appear?
        # A: Page <number>.       
        snippet, page_ids = get_snippet_with_page_id(multi_pages, tokenizer)
        snippet = tokenizer.decode(snippet, skip_special_tokens=True)
        question = f'''On which page does "{snippet}" appear?'''
        answer = f"Page {','.join(map(str, [pid+1 for pid in page_ids]))}."
        return (1, question, answer)

    def get_q2(multi_pages, tokenizer, other_multi_pages):
        # Q:Does "<snippet A>" appear before "<snippet B>"?
        # A:<Yes/No>.
        # Q:Does "<snippet A>" appear after "<snippet B>"?
        # A:<Yes/No>.
        if len(multi_pages)<=2:  # if too short, will be page_idsA[0]==page_idsB[0] forever. 至少>=3,两个的话,有可能最后一页太短,总是取到第一页
            return None

        snippetA,page_idsA = get_snippet_with_page_id(multi_pages, tokenizer)
        snippetB,page_idsB = get_snippet_with_page_id(multi_pages, tokenizer)
        while page_idsA[0]==page_idsB[0]:
            snippetB,page_idsB = get_snippet_with_page_id(multi_pages, tokenizer)

        # avoid A containing B or vice versa
        min_len = min(len(snippetA),len(snippetB))
        snippetA = tokenizer.decode(snippetA[:min_len], skip_special_tokens=True)
        snippetB = tokenizer.decode(snippetB[:min_len], skip_special_tokens=True)

        # snippetA and snippetB is random order, but can use different expression.
        if page_idsA[0]<page_idsB[0]:
            if random.random() < 0.5:
                question = f'''Does "{snippetA}" appear before "{snippetB}"?'''
                answer = "Yes."
                return (2, question, answer)
            else:
                question = f'''Does "{snippetA}" appear after "{snippetB}""?'''
                answer = "No."
                return (2, question, answer)
        else:
            if random.random() < 0.5:
                question = f'''Does "{snippetA}" appear before "{snippetB}"?'''
                answer = "No."
                return (2, question, answer)
            else:
                question = f'''Does "{snippetA}" appear after "{snippetB}""?'''
                answer = "Yes."    
                return (2, question, answer)   

    def get_q3(multi_pages, tokenizer, other_multi_pages):
        # Q: What is the snippet between "<snippet A>" and "<snippet B>"?
        # A: <snippet_middle>
        long_snippet, page_ids = get_snippet_with_page_id(multi_pages, tokenizer,min_len=24, max_len=96)
        tot_len = len(long_snippet)
        snippet_A = tokenizer.decode(long_snippet[:tot_len//3], skip_special_tokens=True)
        snippet_middle = tokenizer.decode(long_snippet[tot_len//3:(tot_len//3)*2], skip_special_tokens=True)
        snippet_B = tokenizer.decode(long_snippet[(tot_len//3)*2:], skip_special_tokens=True)
        question = f'''What is the snippet between "{snippet_A}" and "{snippet_B}"?'''
        answer = snippet_middle
        return (3, question, answer)

    # can't not use twice
    def get_q4(multi_pages, tokenizer, other_multi_pages):
        # Q: What is the most frequent token (excluding special tokens) in the text?
        # A: <most_frequent_token>
        question = 'What is the most frequent token (excluding special tokens) in the text?'
        answer = f'''{', '.join(map(tokenizer._convert_id_to_token, frequency_dict[max_freq]))}''' 
        return (4, question, answer)
              

    def get_q5(multi_pages, tokenizer, other_multi_pages):
        # Q: How many times does the token "<token>" appear in the text?
        # A: <number> times.

        # equal prob in answer.
        cnt = random.choice(freqs)
        tok = random.choice(frequency_dict[cnt])
        question = f'''How many times does the token "{tokenizer._convert_id_to_token(tok)}" appear in the text?'''
        answer = f'{cnt}.'
        return (5, question, answer)
        

    def get_q6(multi_pages, tokenizer, other_multi_pages):
        # Q: The token "<token>" first appears on which page?
        # A: Page <number>.
        # Q: The token "<token>" last appears on which page?
        # A: Page <number>.
        tok = random.choice(tokens)
        if random.random()<0.5:
            question = f'''The token "{tokenizer._convert_id_to_token(tok)}" first appears on which page?'''
            answer = f"Page {token_pids[tok][0]}."
            return(6, question, answer)
        else:
            question = f'''The token "{tokenizer._convert_id_to_token(tok)}" last appears on which page?'''
            answer = f"Page {token_pids[tok][-1]}."
            return(6, question, answer)

    funs = [
        get_q0,
        get_q1,
        get_q2,
        get_q3,
        get_q4,
        get_q5,
        get_q6
    ]


    if usage is None:
        usage = [10000000 for x in range(7)]
        usage[4] = 1 # q4 can only be used once.
    

    question_type_id = random.randint(0,6)
    output = None
    while output is None:
        while usage[question_type_id]==0:
            question_type_id = random.randint(0,6)
        # print(f"use q{question_type_id}")
        # print(f"usage[q{question_type_id}] = {usage[question_type_id]}")
        output = funs[question_type_id](multi_pages, tokenizer, other_multi_pages)
        
        if output is None: # None mean that the text not meat the condition to use the question.
            usage[question_type_id]=0  # every text has unique usage. 

    usage[question_type_id]-=1
    question_type_id, question, answer = output

    return question_type_id, question, answer, usage


def to_tensor(text, tokenizer, length, other_text):
    text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    other_text_ids = tokenizer(other_text, add_special_tokens=False)["input_ids"]
    if len(text_ids)>(length*4)//3:
        text_ids = text_ids[:(length*4)//3]
        logINFO(f"truncate {text}")

    whole_text, multi_pages = get_multi_page_text(text_ids,tokenizer)
    other_whole_text, other_multi_pages = get_multi_page_text(other_text_ids,tokenizer)

    # 一开始没想到会用到这么多参数，早知道传入字典了
    stat = get_stat(multi_pages) # stat: statistics
    question_type_id, question, answer, usage = get_question_and_answer(multi_pages, tokenizer, other_multi_pages, stat)

    first_prompt = 'User input will be repeated twice for better understanding and response generation.\n\n### First occurrence'
    first_prompt = tokenizer(first_prompt, add_special_tokens=False)["input_ids"]
    text = tokenizer("\nText: ", add_special_tokens=False)["input_ids"] + whole_text


    questions_concat = []
    question_ids = tokenizer("\nQuestion 1: "+question, add_special_tokens=False)["input_ids"]
    questions_concat += question_ids

    answers_concat = []
    answers_ids = tokenizer("\nAnswer 1: "+answer, add_special_tokens=False)["input_ids"]
    answers_concat += answers_ids

    qa_interleave = []
    qa_interleave_tgt = []
    qa_interleave += question_ids + answers_ids
    qa_interleave_tgt += [-100 for q_id in question_ids] + [a_id for a_id in answers_ids]


    second_prompt = tokenizer("""\n\n### Second occurrence""", add_special_tokens=False)["input_ids"]

    last_prompt = tokenizer("""\n\n### Answer the following questions based on the provided text""", 
    add_special_tokens=False)["input_ids"]

    tot_len = len(first_prompt) + len(second_prompt) + len(last_prompt) + 2*len(text) + 2 # one for <bos>;one for <eos>
    tot_len += 2*len(questions_concat) + len(qa_interleave) 

    cnt = 1
    while tot_len <= length:
        cnt+=1
        question_type_id, question, answer, usage = get_question_and_answer(multi_pages, tokenizer, other_multi_pages, stat, usage)
        question_ids = tokenizer(f"\nQuestion {cnt}: "+question, add_special_tokens=False)["input_ids"]
        answers_ids = tokenizer(f"\nAnswer {cnt}: "+answer, add_special_tokens=False)["input_ids"]
        tot_len +=  3*len(question_ids) + len(answers_ids) 
        if tot_len > length:
            break
        questions_concat += question_ids
        qa_interleave += question_ids + answers_ids
        qa_interleave_tgt += [-100 for q_id in question_ids] + [a_id for a_id in answers_ids]

    #        <bos>                                                                                                                  <eos>
    inp_ids = [1] + first_prompt + text + questions_concat + second_prompt + text + questions_concat + last_prompt + qa_interleave + [2]
    tgt_ids = [-100] + [-100 for e in first_prompt] + [-100 for e in text] + [-100 for e in questions_concat] + \
                    [-100 for e in second_prompt] + [e for e in text] + [e for e in questions_concat] + \
                    [-100 for e in last_prompt] + qa_interleave_tgt + [2]
    
    # padding; The earlier token cannot see the later token
    pad_len = length - len(inp_ids) + 1
    inp_ids += [2 for x in range(pad_len)]
    tgt_ids += [-100 for x in range(pad_len)]

    return {"input_ids":torch.LongTensor(inp_ids[:-1]), "labels":torch.LongTensor(tgt_ids[1:])}
    


def prepare_data():
    cfg = load_json("prepare_state.json")
    state_path = cfg["data_save_path"]+f"/state.json"
    if os.path.exists(state_path):
        cfg = load_json(state_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'])
    # print(dir(tokenizer))
    
    files = get_sorted_files_full_path(cfg["save_path"])
    files.remove(cfg["save_path"]+'/state.json')
    files = [fff for fff in files if "510_" not in fff]  # leave shortest text for other task.

    sorted_files_with_path = sorted(files, key=lambda file_path: (
        int(file_path.split('/')[-1].split('_')[0]),  # 提取长度并转换为整数
        int(file_path.split('/')[-1].split('_')[1].split('.')[0])  # 提取序号并转换为整数
    ))

    # print(sorted_files_with_path)
    # save_json(sorted_files_with_path,"info.json")
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


    length = 1020
    if "finished_length" in cfg:
        length = cfg["finished_length"]+510

    data = []
    # num = 1000000//length # for debug
    num = (1000*1000*1000)//length  # total 1B token

    last_text = None
    for data_path in tqdm(unfinished_texts):
        print(f"processing {data_path}...")
        texts = load_json(data_path)

        # preing = 1
        for text in tqdm(texts):
            # print(f"preprocessing {preing}")
            # preing += 1
            if last_text is None:
                last_text = text
                continue

            # other_text hasn't been seen or trained.
            data.append(to_tensor(last_text, tokenizer, length, other_text=text))
            last_text = text

            # save
            if len(data)==num:
                

                torch.save(data, cfg["data_save_path"]+f"/{length}.pt")
                cfg["finished_length"] = length
                cfg["finished_text"] = data_path # if interrupt, ignore the remain text in the file, and use next file.
                logINFO(f"""{cfg['finished_length']}.pt use ~{data_path}""")
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
    prepare_data()
    print("finish data for lengh.pt")
    
