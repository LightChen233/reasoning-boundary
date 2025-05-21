'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2024-03-03 15:00:53
LastEditTime: 2024-04-01 13:38:06
Description: 

'''
import fire
from matplotlib import pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from utils.request_tool import RequestOutput
from utils.tools import get_combined_boundary

N = 1.6e5
M = 7.0
SIGMA = 20000
PARAM_DICT = {
    "gpt35-turbo": {
        "K": 0.106,
        "K2": 0.425,
        "mode": "nl",
        "result_path": "experiments/RBF/mathematical-reasoning/gpt35-CoT.jsonl"
    },
    "gpt-4o": {
        "K": 0.12,
        "K2": 0.71,
        "mode": "nl",
        "result_path": "experiments/RBF/mathematical-reasoning/gpt4o-CoT.jsonl"
    },
    "o1-preview": {
        "K": 0.301,
        "K2": 0.92,
        "mode": "nl",
        "result_path": "experiments/RBF/mathematical-reasoning/o1-CoT.jsonl"
    },
    "deepseek-r1": {
        "K": 0.52,
        "K2": 39.06,
        "mode": "nl",
        "result_path": "experiments/RBF++/zero-shot/biggsm++/deepseek-r1-tp-00.jsonl"
    },
}
def run(data_split = "deepseek-r1"):
    result_path = PARAM_DICT[data_split]["result_path"]
    K = PARAM_DICT[data_split]["K"]
    K2 = PARAM_DICT[data_split]["K2"]
    response_list = RequestOutput(result_path)
    acc, total = 0, 0
    MAX_CAL = 8e8
    MAX_PLAN = 500
    x_list, y_list, z_list = [], [], []
    print(len(response_list))
    for idx in range(len(response_list)):
        total += 1
        boundary = get_combined_boundary(response_list.get_origin_input(idx), return_dict=True)
        if boundary["calculate_boundary"] > MAX_CAL or boundary["plan_boundary"] > MAX_PLAN:
            continue
        y_list.append(boundary["plan_boundary"])
        x_list.append(boundary["calculate_boundary"])
        if response_list.judge_correct(idx):
            z = "Y"
            acc += 1
        else:
            z = "N"
        z_list.append(z)
    print(f"ACC: {round(acc/len(z_list), 2)}\t Total: {len(z_list)}")
    
    x = np.linspace(max(K*M+0.001, 0), MAX_PLAN, 500)
    x = x[x != K*M]
    x = x[N*K + N*K*M / ((x)/K - M) - SIGMA > 0]
    x = x[N*K + N*K*M / ((x)/K - M) - SIGMA < MAX_CAL]

    y = N*K + N*K*M / ((x)/K - M) - SIGMA
    df = pandas.DataFrame({"x": y_list, "y": x_list, "z": z_list})
    sns.relplot(
        data=df,
        x="x", y="y",
        hue="z",
        style=[0 if x == "N" else 0 for x in z_list],
        palette=["C0", "C3"] if z_list[0] == "Y" else ["C3", "C0"],
        s=60,
    )
    plt.plot(x, y)
    x = np.linspace(K2*M, MAX_PLAN, 500)
    x = x[(x)/K2 - M != 0]
    x = x[N*K2 + N*K2*M / ((x)/K2 - M) - SIGMA < MAX_CAL]
    x = x[x > K2*M]
    x = x[N*K2 + N*K2*M / ((x)/K2 - M) - SIGMA >-1000]
    y = N*K2 + N*K2*M / ((x)/K2 - M) - SIGMA
    plt.plot(x, y)
    plt.savefig(f'{data_split}.svg', format='svg')
    plt.show()

if __name__ == "__main__":
    fire.Fire(run)