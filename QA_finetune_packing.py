import pandas as pd
from datasets import Dataset
import pandas as pd
import torch
import argparse
from datasets import Dataset, load_dataset
from random import randrange
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments,LlamaTokenizer
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
from peft import AutoPeftModelForCausalLM

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
args = parser.parse_args()
model_name=args.model_name
dataset=args.dataset

df = pd.read_csv(f'./{dataset}')
if (dataset=="squad_hindi.csv"):
    df=df[:3734]
train_dataset = Dataset.from_pandas(df)

output_dir='airavta-tuned-qna'

# #for airavata
# df['text'] = '<|system|>\n Answer the following question based on the information in the given passage.\n <|user|>\n Passage:' + df['context'] + '\n <|user|>\n Question:' + df['question'] + '\n <|assistant|>\n Answer:\n' + df['answer_text'] + f"{tokenizer.eos_token}\n"
# train = Dataset.from_pandas(df)

df['text'] = 'Answer the following question based on the information in the given passage.\n Passage:\n' + df['context'] + '\nQuestion:\n' + df['question'] + '\n\nAnswer:\n' + df['answer_text']
train = Dataset.from_pandas(df)

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer,mlm=False)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             quantization_config=bnb_config,
                                             device_map="auto")

peft_config = LoraConfig(
                          lora_alpha=32,
                          lora_dropout=0.05,
                          r=16,
                          target_modules=["q_proj", "v_proj", "k_proj", "down_proj", "gate_proj", "up_proj"],
                          bias="none",
                          task_type="CAUSAL_LM",
                        )

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1, # adjust based on the data size
    per_device_train_batch_size=2, # use 4 if you have more GPU RAM
    # save_strategy="epoch", #steps
    # evaluation_strategy="epoch",
    learning_rate=5e-4,
    fp16=True,
    # bf16=True,
    seed=42,
)
# # Create the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train,
    # eval_dataset=test,
    dataset_text_field='text',
    peft_config=peft_config,
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=args,
    packing=True,
)
# train
trainer.train()
trainer.save_model()

del model
del trainer
import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache()
gc.collect()

new_model = AutoPeftModelForCausalLM.from_pretrained(
    'airavta-tuned-qna',
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
merged_model = new_model.merge_and_unload()
merged_model.save_pretrained("airavta-qa-tuned-merged", safe_serialization=True)
tokenizer.save_pretrained("airavta-qa-tuned-merged")
# hf_model_repo = "ayushayush591/airavta-tunned"
# merged_model.push_to_hub(hf_model_repo)
# tokenizer.push_to_hub(hf_model_repo)
