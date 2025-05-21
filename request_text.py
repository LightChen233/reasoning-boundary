'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-12-18 14:54:57
LastEditTime: 2024-05-18 16:38:28
Description: 

'''

import asyncio
from functools import partial
import random
import fire

from utils.request_tool import request_LLM
from utils.tools import read_jsonl

random.seed(42)

def create_prompt(data, prompt_config):
    instruction = """Please reason and answer the following questions, and finally use the format of "### xxx" to give "xxx" as the final answer.

Question: """
    instruction += data["question"]
    return instruction


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
        model_name="gpt-4o-mini",
        api_key="sk-xxx",
        request_proxy="https://xxx", # base_url; None means OpenAI base_url by default
        temperature=0.0):
    
    model_config = {
        "temperature": temperature,
    }
    asyncio.run(request_LLM(
        total=total,
        split=split,
        dataset=DataLoader("data/biggsm++/data.jsonl"),
        save_path=f"deepseek-r1/o1/{model_name}-biggsm++-tp-00.jsonl",
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
