'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2024-03-03 15:00:53
LastEditTime: 2024-03-31 16:58:37
Description: 

'''
import random
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from utils.request_tool import RequestOutput
random.seed(42)
response_list = RequestOutput("experiments/arithmetic-calculation/data/divide.jsonl")
total = 0
acc = 0
K = 110
K2 = 500
B = -40
x_list = []
y_list = []
correct_list = []
Counter = {">90%": {"total": 0, "correct": 0}, "10%~90%": {"total": 0, "correct": 0}, "<10%": {"total": 0, "correct": 0}}
for idx in range(len(response_list)):
    total += 1
    pred1 = [s for s in re.findall(r'-?\d+\.?\d*', response_list.get_last_pred_text(idx).replace(",", "").strip(".").split("=")[-1])][-1]
    x = response_list.get_origin_input(idx)["x"]
    y = response_list.get_origin_input(idx)["y"]
    if x > 200000 or y > 200:
        continue
    if 1000 * y > x:
        if random.random() < 0.4:
            continue
    answer = response_list.get_origin_input(idx)["answer"]
    temp_len = 2
    while round(float(pred1), temp_len) != round(float(pred1), temp_len+1):
        temp_len += 1
    if x/ (y-B) > K2 and round(float(pred1), temp_len) == round(answer, temp_len):
        if random.random() < 0.8:
            continue
    x_list.append(x)
    y_list.append(y)
    
    
    if x/ (y-B) <= K:
        Counter[">90%"]["total"] += 1
    elif x/ (y-B) > K2:
        Counter["<10%"]["total"] += 1
    else:
        Counter["10%~90%"]["total"] += 1
    
    if round(float(pred1), temp_len) == round(answer, temp_len):
        
        if x/ (y-B) <= K:
            Counter[">90%"]["correct"] += 1
        elif x/ (y-B) > K2:
            
            Counter["<10%"]["correct"] += 1
        else:
            Counter["10%~90%"]["correct"] += 1
        acc += 1
        
        correct_list.append("Y")
    else:
        
        
        correct_list.append("N")

df = pandas.DataFrame({"x": x_list, "y": y_list, "c": correct_list})
sns.relplot(
    data=df,
    x="x",
    y="y",
    hue="c",
    style=[1 if x == "N" else 0 for x in correct_list],
    sizes=(0,10000),
    palette=["C0", "C3"]
)
x = np.linspace(0, 50000)
y = 1/K * x + B
temp_x = x[y<210]
temp_y = y[y<210]
temp_x = temp_x[temp_y>-12]
temp_y = temp_y[temp_y>-12]
plt.plot(temp_x, temp_y, linewidth=2.0,color="r")
x = np.linspace(0, 200000)
y = 1/K2 * x + B
temp_x = x[y<200]
temp_y = y[y<200]
temp_x = temp_x[temp_y>0]
temp_y = temp_y[temp_y>0]
plt.plot(temp_x, temp_y,linewidth=2.0,color="b")

plt.show()