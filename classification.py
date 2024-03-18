# CUDA_VISIBLE_DEVICES=0,1 python classification.py
# !pip -q install langchain #tiktoken #duckduckgo-search
# # !pip install -q -U bitsandbytes
# # !pip install -q -U git+https://github.com/huggingface/transformers.git
# # !pip install -q -U git+https://github.com/huggingface/peft.git
# # !pip install -q -U git+https://github.com/huggingface/accelerate.git
# # !pip -q install sentencepiece Xformers einops
# !pip -q install vllm

import os
from vllm import LLM, SamplingParams
model ="meta-llama/Llama-2-70b-chat-hf"

context=[]
question=[]
with open("./translation/original_english/context_llama.txt", "r") as file:
  for item in file:
    context.append(item.strip())
with open("./translation/original_english/question_llama.txt", "r") as file:
  for item in file:
    question.append(item.strip())

prompts=[]
prompt="""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>""" + "Please Answer Just Yes or No based on whether the question can Exactly be answered by Given Context:\n"

for i in range(len(context)):
    prompts.append(f"{prompt} context:\n{context[i]}\n question:\n {question[i]}\nAnswer:[/INST]")

sampling_params = SamplingParams(temperature=0.0, top_p=0.95)
llm = LLM(model=model,device='auto')
outputs = llm.generate(prompts, sampling_params)

with open("output.txt", "w") as file:
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_text = repr(generated_text)
        file.write(f'{generated_text}\n')
        # print(f"Generated text: {generated_text!r}")
