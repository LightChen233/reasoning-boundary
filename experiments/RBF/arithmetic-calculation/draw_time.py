'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2024-03-03 15:00:53
LastEditTime: 2024-03-30 16:19:21
Description: 

'''
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from utils.request_tool import RequestOutput

response_list = RequestOutput("experiments/arithmetic-calculation/data/time.jsonl")
total = 0
acc = 0
K = 220000
K2 = 2e6
x_list = []
y_list = []
correct_list = []
Counter = {">90%": {"total": 0, "correct": 0}, "10%~90%": {"total": 0, "correct": 0}, "<10%": {"total": 0, "correct": 0}}
for idx in range(len(response_list)):
    total += 1
    pred1 = [s for s in re.findall(r'-?\d+\.?\d*', response_list.get_last_pred_text(idx).replace(",", "").strip(".").split("=")[-1])][-1]
    x = response_list.get_origin_input(idx)["x"]
    y = response_list.get_origin_input(idx)["y"]
    if x > 1e16 or y > 1e16:
        continue
    x_list.append(x)
    y_list.append(y)
    answer = response_list.get_origin_input(idx)["answer"]
    if x * y <= K:
        Counter[">90%"]["total"] += 1
    if x * y > K2:
        Counter["<10%"]["total"] += 1
    else:
        Counter["10%~90%"]["total"] += 1
    if round(float(pred1), 2) == round(answer, 2):
        acc += 1
        if x * y <= K:
            Counter[">90%"]["correct"] += 1
        if x * y > K2:
            Counter["<10%"]["correct"] += 1
        else:
            Counter["10%~90%"]["correct"] += 1
        correct_list.append("Y")
    else:
        
        correct_list.append("N")

df = pandas.DataFrame({"x": x_list, "y": y_list, "c": correct_list})
sns.relplot(
    data=df,
    x="x", y="y",
    hue="c",
    style="c",
    sizes=(0,10000),
    palette=["C0", "C3"]
)
x = np.linspace(30, 2000)
y = K/x
temp_x = x[y<2000]
temp_y = y[y<2000]
plt.plot(temp_x, temp_y, linewidth=2.0,color="r")
x = np.linspace(30, 2000)
y = K2/x
temp_x = x[y<2000]
temp_y = y[y<2000]

plt.plot(temp_x,temp_y,linewidth=2.0,color="b")

plt.show()