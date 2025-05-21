'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-12-18 14:54:57
LastEditTime: 2025-05-18 16:38:28
Description: 

'''

import asyncio
import base64
from functools import partial
import random
import fire

from utils.request_tool import request_LLM
from utils.tools import read_jsonl

random.seed(42)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
ALPHA_MAP = ["A", "B", "C", "D", "E", "F", "G"]
def create_prompt(data, prompt_config):
    choice_str = ""
    for idx, choice in enumerate(data["choices"]):
        choice_str += f"({ALPHA_MAP[idx]}) {choice}\n"
    context_str = ""
    if data["context"] != "":
        context_str = f"""[Context]\n{data["question"]}\n\n"""
    instruction = f"""{context_str}[Question]
{data["question"]}\n\n[Choices]\n{choice_str}\nLet's think step-by-step!"""
    base64_image = encode_image(data["image_path"])

    return [
        {"type": "text", "text": instruction},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
            },
        },
    ]


class DataLoader:
    def __init__(self, load_path: str) -> None:
        input_data = []
        for i, data in enumerate(read_jsonl(load_path)):
            if "index" not in data:
                data["index"] = str(i)
            input_data.append(data)
        input_data.reverse()
        self.data = input_data


def run(total=1, 
        split=0, 
        model_type="gpt",
        model_name="gemini-2.0-flash-thinking-exp-1219",
        api_key="sk-xxx",
        request_proxy="https:/xxx",
        temperature=0.0,):
    
    model_config = {
        "temperature": temperature,
    }
    asyncio.run(request_LLM(
        total=total,
        split=split,
        dataset=DataLoader("data/m3cot/test.jsonl"),
        save_path=f"experiments/m3cot/{model_name}-tp-00.jsonl",
        consumer_size=25,
        create_prompt_fn=partial(create_prompt, prompt_config=None),
        model_type=model_type,
        model_name=model_name,
        api_key=api_key,
        enable_multi_turn = False,
        request_proxy=request_proxy,
        return_origin=True,
        model_config=model_config
        ))

if __name__ == "__main__":
    fire.Fire(run)
