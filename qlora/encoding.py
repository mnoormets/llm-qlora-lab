"""Completion-only labels; never silently truncate a supervised answer."""
import json
from .data import SYSTEM

def prompt_for(tokenizer,row):
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':row['document']}]
    return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

def encode_row(tokenizer,row,max_length=768):
    prompt=prompt_for(tokenizer,row)
    inputs=tokenizer.encode(prompt,add_special_tokens=False)
    completion=tokenizer.encode(json.dumps(row['expected'],ensure_ascii=False)+tokenizer.eos_token,add_special_tokens=False)
    if not completion or len(inputs)+len(completion)>max_length:raise ValueError('Example exceeds token budget; no answer truncation allowed')
    return {'input_ids':inputs+completion,'attention_mask':[1]*(len(inputs)+len(completion)),'labels':[-100]*len(inputs)+completion}

class CompletionCollator:
    def __init__(self,pad_token_id):self.pad_token_id=pad_token_id
    def __call__(self,rows):
        import torch
        width=max(len(row['input_ids']) for row in rows);width=((width+7)//8)*8
        output={key:[] for key in ['input_ids','attention_mask','labels']}
        for row in rows:
            extra=width-len(row['input_ids'])
            output['input_ids'].append(row['input_ids']+[self.pad_token_id]*extra)
            output['attention_mask'].append(row['attention_mask']+[0]*extra)
            output['labels'].append(row['labels']+[-100]*extra)
        return {key:torch.tensor(value,dtype=torch.long) for key,value in output.items()}
