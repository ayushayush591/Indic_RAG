import pandas as pd
import random
import numpy as np
import torch
import os
import argparse
import json
import re
import string
import csv
import sys
from collections import Counter
from itertools import zip_longest
# from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
set_seed()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default=None,
    help="if specified, we will load the model_name from here.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    help="if specified, we will load the dataset from here.",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="if specified, we will load the device from here.",
)
parser.add_argument(
    "--shot",
    type=int,
    default=0,
    help="if specified, we will load the shot from here.",
)
args = parser.parse_args()
model_name=args.model_name
dataset=args.dataset
device=args.device
shot=args.shot
df = pd.read_csv(f'./data/{dataset}')
# df=df.drop(['answer_start','language'], axis=1)
# df=df[3734:]
# df=df[df['language']=="hindi"]
context=df['context'].to_list()
question=df['question'].to_list()
answer_text=df['answer_text'].to_list()


prompts=[]
start="Answer the following question based on the information in the given passage.:"
prompt=""
prompt=start+prompt
run=shot
while(run>0):
    prompt=prompt+f'\n\nPassage:{context[run-1]}\nQuestion:{question[run-1]}\nAnswer:{answer_text[run-1]}'
    run=run-1
for i in range(shot,len(context)):
    pro=prompt+f'\n\nPassage:{context[i]}\nQuestion:{question[i]}\nAnswer:'    
    prompts.append(pro)      

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

answer=[]
prediction=[]
for index,item in enumerate(prompts):
    inputs = tokenizer(item, return_tensors="pt").to(device)
    print(f"{index} prompt is going on")
    try:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens=50)
        prediction.append(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        answer.append(answer_text[index])
    except:
        continue    

def extract_answer_content(passage):
    delim="\nAnswer:"
    index = passage.find(delim)

    if index != -1:
        content = passage[index + len(delim):]
        return content
    else:
        return " "
output=[]
for i in range(len(prediction)):
    output.append(extract_answer_content(prediction[i]))

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction_, ground_truth):
    prediction_tokens = normalize_answer(prediction_).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def jacard(prediction_, ground_truth):
    a = set(prediction_.lower().split()) 
    b = set(ground_truth.lower().split())
    c = a.intersection(b)
    return float(len(c)) / abs(len(a) + len(b) - len(c))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction_, ground_truths):
    score = metric_fn(str(prediction_),str(ground_truths))
    return score

def evaluate(reference, prediction):
    f1 = exact_match = total = jacard_sim = 0
    for i in range(len(reference)):
        ground_truths = reference[i]
        prediction_ = prediction[i]
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction_, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction_, ground_truths)
        jacard_sim += jacard(prediction_, ground_truths)
        total=total+1
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    jacard_sim = 100.0 * jacard_sim / total

    return {"exact_match": exact_match, "f1": f1, "jacard":jacard_sim}

# combined_lists = [[x, y] for x, y in zip(answer, output)]
# with open('combined_lists_.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['ground_truth','answer'])  # Header
#     writer.writerows(combined_lists)

file_path = "results_2.txt"
with open(file_path, "a") as file:
    file.write(f'{model_name} on {dataset} {str(evaluate(answer,output))}')
# print(evaluate(answer,output))