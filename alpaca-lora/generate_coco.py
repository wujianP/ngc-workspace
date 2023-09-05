import os
import sys

import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


@torch.no_grad()
def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        prompt_template: str = "",  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            instruction="Change the meaning of the input sentence with minimal modifications while keeping the overall structure unchanged.",
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=80,
            **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeatition_penalty=2.0,
            num_beams=num_beams,
            **kwargs,
        )

        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
        s = generation_output.sequences[0]
        # tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return tokenizer.decode(s)

    from IPython import embed
    embed()


if __name__ == "__main__":
    fire.Fire(main)


# evaluate(input="A man is smiling while talking on his cell phone.")
def test():
    inputs = [
        "A woman marking a cake with the back of a chef's knife",
        "a young kid with head phones on using a computer",
        "A small child wearing headphones plays on the computer.",
        "Baby stands up in car styled walker with bunch of beads around his neck.",
        "A tennis player runs across the court to hit a ball."
    ]
    for ipt in inputs:
        print(evaluate(instruction=instruction, input=ipt))

#############
from datasets import load_dataset

examples = load_dataset('facebook/winoground', use_auth_token="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")['test']
data = []
for i in range(len(examples)):
    data.append({
        "instruction": "modify the input sentence to change its meaning solely by rearranging the order of the words.",
        "input": examples[i]['caption_1'],
        "output": examples[i]['caption_0']
    })

    data.append({
        "instruction": "modify the input sentence to change its meaning solely by rearranging the order of the words.",
        "input": examples[i]['caption_0'],
        "output": examples[i]['caption_1']
    })
import json
# 将数据列表保存为JSON文件（UTF-8编码）
file_path = "/discobox/wjpeng/dataset/alpaca/winoground_edit.json"
with open(file_path, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)
