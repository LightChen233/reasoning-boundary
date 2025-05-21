'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2024-03-03 15:00:53
LastEditTime: 2024-05-18 18:04:24
Description: 

'''
import tiktoken
import fire
from utils.request_tool import RequestOutput
from utils.tools import get_combined_boundary
from prettytable import PrettyTable

def loop_judge(condition_list, input_str):
    for cond in condition_list:
        if cond in input_str:
            return True
    return False



PARAM_DICT = {
    "o3-mini-low": {
        "K": 0.14,
        "K2": 3.20,
        "mode": "nl",
        "result_path": "experiments/RBF++/zero-shot/biggsm++/o3-mini-low-tp-00.jsonl"
    },
    "o3-mini-medium": {
        "K": 0.18,
        "K2": 40.2,
        "mode": "nl",
        "result_path": "experiments/RBF++/zero-shot/biggsm++/o3-mini-medium-tp-00.jsonl"
    },
    "o3-mini-high": {
        "K": 0.16,
        "K2": 45.10,
        "mode": "nl",
        "result_path": "experiments/RBF++/zero-shot/biggsm++/o3-mini-high-tp-00.jsonl"
    },
    "gpt-4o-mini": {
        "K": 0.111,
        "K2": 0.17,
        "mode": "nl",
        "result_path": "experiments/RBF++/zero-shot/biggsm++/gpt-4o-mini-tp-00.jsonl"
    },
    "deepseek-r1": {
        "K": 0.52,
        "K2": 39.06,
        "mode": "nl",
        "result_path": "experiments/RBF++/zero-shot/biggsm++/deepseek-r1-tp-00.jsonl"
    },
    "deepseek-v3": {
        "K": 0.12,
        "K2": 1.23,
        "mode": "nl",
        "result_path": "experiments/RBF++/zero-shot/biggsm++/deepseek-v3-tp-00.jsonl"
    },
}

def main(data_split="deepseek-v3",
         K=None,
         K2=None,
         mode=None,
         result_path=None):
    if data_split == "custom":
        if K is None:
            raise ValueError("Missing Args: --K")
        elif K2 is None:
            raise ValueError("Missing Args: --K2")
        elif mode is None:
            raise ValueError("Missing Args: --mode")
        elif result_path is None:
            raise ValueError("Missing Args: --result_path")
    else:
        K = PARAM_DICT[data_split]["K"]
        K2 = PARAM_DICT[data_split]["K2"]
        mode = PARAM_DICT[data_split]["mode"]
        result_path = PARAM_DICT[data_split]["result_path"]
    
    response_list = RequestOutput(result_path)
    table = PrettyTable()
    table.field_names = ["Boundary", "Acc", "Total", "Input Token", "Output Token"]

    token_num = 0
    input_token_num = 0
    enc = tiktoken.encoding_for_model("gpt-4")
    acc = {">90%": {"correct": 0, "total": 0}, "10%~90%": {"correct": 0, "total": 0}, "<10%": {"correct": 0, "total": 0}}
    MAX_CAL = 8e8
    MAX_PLAN = 500
    for idx in range(len(response_list)):
        boundary = get_combined_boundary(response_list.get_origin_input(idx), return_dict=True)
        if boundary["calculate_boundary"] > MAX_CAL or boundary["plan_boundary"] > MAX_PLAN:
            continue
        input_token_num += len(enc.encode(response_list.data[idx]["pred"][0]["content"][0]["text"]))
        token_num += len(enc.encode(response_list.get_last_pred_text(idx)))
        boundary = boundary["combined_boundary"]
        if boundary <= K:
            boundary_key = ">90%"
        elif boundary > K  and boundary <= K2:
            boundary_key = "10%~90%"
        elif boundary > K2:
            boundary_key = "<10%"
        if response_list.judge_correct(idx, mode=mode):
            acc[boundary_key]["correct"] += 1
        acc[boundary_key]["total"] += 1
    total = 0
    correct = 0
    for key in acc:
        if acc[key]['total']>0:
            total += acc[key]['total']
            correct += acc[key]['correct']
            table.add_row([
                key,
                round(acc[key]['correct']/acc[key]['total'] * 100, 2),
                acc[key]['total'],
                "-",
                "-"
            ], divider=key == "<10%")
        else:
            table.add_row([
                key,
                "-",
                "0",
                "-",
                "-"
            ], divider=key == "<10%")
    table.add_row(["All", round(correct/total * 100, 2), total, round(input_token_num/total, 2), round(token_num/total, 2)])
    print(table)
    
if __name__ == "__main__":
    fire.Fire(main)