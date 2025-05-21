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
from utils.request_tool import RequestOutput
from utils.tools import get_mm_combined_boundary

N = 1.6e5
M = 7.0
SIGMA = 20000
PARAM_DICT = {
    "gpt-4o": {
        "K": 0.134,
        "K2": 1.392,
        "mode": "nl",
        "result_path": "experiments/RBF++/zero-shot/m3cot/gpt-4o-tp-00.jsonl"
    },
}
def run(data_split = "gpt-4o"):
    result_path = PARAM_DICT[data_split]["result_path"]
    K = PARAM_DICT[data_split]["K"]
    K2 = PARAM_DICT[data_split]["K2"]
    response_list = RequestOutput(result_path)
    acc, total = 0, 0
    x_list, y_list, z_list, c_list = [], [], [], []
    print(len(response_list))
    for idx in range(len(response_list)):
        total += 1
        boundary = get_mm_combined_boundary(response_list.get_origin_input(idx), return_dict=True)
        x_list.append(boundary["global_boundary"])
        y_list.append(boundary["local_boundary"])
        z_list.append(boundary["domain_boundary"])
        if response_list.judge_multi_choice_correct(idx):
            c = "b"
            acc += 1
        else:
            c = "r"
        c_list.append(c)
    response_list = RequestOutput("experiments/olympiadbench/result/gpt-4o-tp-00.jsonl")
    for idx in range(len(response_list)):
        total += 1
        response_list.data[idx]["origin"]["topic"] = response_list.data[idx]["origin"]["topic"] if response_list.data[idx]["origin"]["topic"] in ["geometry", "algebra"] else "natural-science"
        boundary = get_mm_combined_boundary(response_list.get_origin_input(idx), return_dict=True)
        if boundary["global_boundary"] > 30 or boundary["local_boundary"] > 40:
            continue
        x_list.append(boundary["global_boundary"])
        y_list.append(boundary["local_boundary"])
        z_list.append(boundary["domain_boundary"])
        if response_list.judge_math_correct(idx):
            c = "b"
            acc += 1
        else:
            c = "r"
        c_list.append(c)
    print(f"ACC: {round(acc/len(z_list), 2)}\t Total: {len(z_list)}")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(x_list, y_list, z_list, c=c_list)


    k = K

    x_min, x_max = min(x_list), max(x_list)
    y_min, y_max = min(y_list), max(y_list)
    print(x_max)
    x = np.concatenate((np.linspace(x_min, x_min+2, 90),np.linspace(x_min+2, x_max, 10)))
    y = np.concatenate((np.linspace(y_min, y_min+2, 90),np.linspace(y_min+2, y_max, 10)))
    X, Y = np.meshgrid(x, y)
    N1, B1 = 3, 0
    N2, B2 = 7, 0
    N3, B3 = 1, -10

    denominator = (1/k) - (N1/(X+B1)) - (N2/(Y+B2))
    Z = np.divide(1, denominator, where=denominator > 0)
    Z = N3 * (Z-B3)
    Z[denominator <= 0] = np.nan
    Z[Z > max(z_list)] = np.nan
    
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, rstride=1, cstride=1)

    k = K2
    x_min, x_max = min(x_list), max(x_list)
    y_min, y_max = min(y_list), max(y_list)
    x = np.concatenate((np.linspace(x_min, x_max/5, 50),np.linspace(x_max/5, x_max, 50)))
    y = np.concatenate((np.linspace(y_min, x_max/5, 50),np.linspace(y_min/5, y_max, 50)))
    X, Y = np.meshgrid(x, y)
    
    denominator = (1/k) - (N1/(X+B1)) - (N2/(Y+B2))
    Z = np.divide(1, denominator, where=denominator > 0)
    Z = N3 * (Z-B3)
    Z[denominator <= 0] = np.nan
    Z[Z > max(z_list)] = np.nan
    
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, rstride=1, cstride=1)

    ax.view_init(elev=25, azim=160)
    plt.show()

if __name__ == "__main__":
    fire.Fire(run)