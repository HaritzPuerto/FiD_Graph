from transformers import T5Tokenizer
import datasets
from datasets import Dataset

import torch
import json
from tqdm import trange, tqdm
import numpy as np

def add_html_tags2tokenizer(tokenizer):
    tokenizer.add_tokens(["<li>"])
    tokenizer.add_tokens(["</li>"])
    tokenizer.add_tokens(["<p>"])
    tokenizer.add_tokens(["</p>"])
    tokenizer.add_tokens(["<h1>"])
    tokenizer.add_tokens(["</h1>"])
    tokenizer.add_tokens(["<h2>"])
    tokenizer.add_tokens(["</h2>"])
    tokenizer.add_tokens(["<h3>"])
    tokenizer.add_tokens(["</h3>"])
    tokenizer.add_tokens(["<h4>"])
    tokenizer.add_tokens(["</h4>"])
    tokenizer.add_tokens(["<tr>"])
    tokenizer.add_tokens(["</tr>"])
    tokenizer.add_tokens(["\\n"])
    
def get_sections(doc):
    list_sections = []
    section = []
    for tag in doc:
        if "<h1>" in tag or "<h2>" in tag or "<h3>" in tag or "<h4>" in tag:
            if len(section) > 0:
                list_sections.append(section)
            section = []
    
        section.append(tag)
    return list_sections

def get_label(x):
    label = ""
    for answer_dict in x['answers']: # more than one answer per question
        label += "Answer: "
        label += answer_dict[0]
        label += ". Conditions: "
        if len(answer_dict[1]) > 0:
            for condition in answer_dict[1]:
                label += condition
                label += " "
        else:
            label += "NA"
    return label

def get_idx_lines(encodings, sent_sep_encoding):
    list_sent_idx = []
    prev_idx = 0
    for i, idx in enumerate((encodings.input_ids[0] == sent_sep_encoding).nonzero().view(-1).tolist()):
        list_sent_idx.append((prev_idx, idx))
        prev_idx = idx
    list_sent_idx.append((prev_idx, encodings.input_ids[0].tolist().index(1)))
    return list_sent_idx

def truncate_section(tokenizer, question, section, max_length, sent_sep_encoding):
    if len(section[1:]) > 0:
        ctx = " \\n ".join(section[1:])
        txt = f"{question} \\n title: {section[0]} context: \\n {ctx}"
    else:
        txt = f"{question} \\n title: {section[0]} context:"

    encodings = tokenizer(txt, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
    list_sent_idx = get_idx_lines(encodings, sent_sep_encoding)
    truncated_text = tokenizer.decode(encodings.input_ids[0], skip_special_tokens=True)
    # remove \\n at the end 
    # (in some edge cases, after the truncation, the last character is a \\n and 
    # this would be considered as a sentence when creatin the graph)
    if truncated_text[-2:] == "\\n":
        truncated_text = truncated_text[:-2]
    return truncated_text, list_sent_idx

def format_data(tokenizer, data, url2doc, sent_sep_encoding, max_length: 512, print_info: False):
    list_len_sections = []
    
    dataset = []
    for x in tqdm(data):
        list_sections = get_sections(url2doc[x['url']]['contents'])
        question = "Question: " + x['question'] + " Scenario: " + x['scenario']
        # truncate sections
        truncated_sec = []
        for s in list_sections:
            truncated_sec_txt, list_sent_idx = truncate_section(tokenizer, question, s, max_length, sent_sep_encoding)
            # truncated_sec_txt = [l.strip() for l in truncated_sec_txt.split("\\n")]
            truncated_sec.append({'title': s[0], 'text': truncated_sec_txt})
        # new data point
        new_x = {
                'id': x['id'],
                'question': question,
                'target': get_label(x),
                'answers': [''],
                'full_ctxs': [{'title': s[0], "text": s[1:]} for s in list_sections],
                'ctxs': truncated_sec,
                'list_sent_idx': list_sent_idx
                }
        dataset.append(new_x)
        
        list_len_sections.append(len(list_sections))
    if print_info:
        print(f"Max. num. sections: {max(list_len_sections)}")
        print(f"Mean num. sections: {np.mean(list_len_sections)}")
        print(f"Std. num. sections: {np.std(list_len_sections)}")
    return dataset
    
if __name__ == "__main__":
    print(f"Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=512)
    add_html_tags2tokenizer(tokenizer)
    sent_sep_encoding = tokenizer.get_added_vocab()['\\n']
    
    documents_path = '/home/puerto/projects/FiD_Graph/FiD/data/documents.json'
    print(f"Loading documents from {documents_path}")
    with open(documents_path) as f:
        documents = json.load(f)
    url2doc = {d['url']: d for i, d in enumerate(documents)}
    
    training_path = '/home/puerto/projects/FiD_Graph/FiD/data/train.json'
    print(f"Loading training data from {training_path}")
    with open(training_path) as f:
        train = json.load(f)
    print(f"Format training data")
    training_dataset = format_data(tokenizer, train, url2doc, sent_sep_encoding, max_length=512, print_info=True)
    with open("../data/fid_format_base_qnode/train.json", 'w') as f:
        json.dump(training_dataset, f)
       
    print(f"Loading validation data from {training_path}") 
    dev_path = '../data/dev.json'
    with open(dev_path) as f:
        dev = json.load(f)
    print(f"Format validation data")
    dev_dataset = format_data(tokenizer, dev, url2doc, sent_sep_encoding, max_length=512, print_info=True)
    with open("../data/fid_format_base_qnode/dev.json", 'w') as f:
        json.dump(dev_dataset, f)
        
    with open("../data/fid_format_base_qnode/DONE.txt", 'w') as f:
        f.write("DONE")