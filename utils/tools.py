'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-10-13 14:24:40
LastEditTime: 2025-05-14 21:48:48
Description: 

'''
import json
import os
import re
import time


def read_jsonl(data_path):
    input_data = []
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf8") as f:
            for line in f:
                input_data.append(json.loads(line.strip()))
    else:
        print(f"Missing {data_path}")
    return input_data


def write_jsonl(save_path, save_object, mode="a"):
    with open(save_path, mode, encoding="utf8") as f:
        for obj in save_object:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def evaluate_expression(expression):
    max_dict = {"plus": 0, "time": 0}
    def parse_expression(i):
        value, i = parse_term(i)
        while i < len(expression) and expression[i] in '+-':
            if expression[i] == '+':
                i += 1
                right_value, i = parse_term(i)
                value += right_value
                if abs(value) > abs(max_dict["plus"]):
                    max_dict["plus"] = abs(value)
            elif expression[i] == '-':
                i += 1
                right_value, i = parse_term(i)
                value -= right_value
                
        return value, i
    
    def parse_term(i):
        value, i = parse_factor(i)
        while i < len(expression) and expression[i] in '*/':
            if expression[i] == '*':
                i += 1
                right_value, i = parse_factor(i)
                value *= right_value
                if abs(value) > abs(max_dict["time"]):
                    max_dict["time"] = abs(value)
            elif expression[i] == '/':
                i += 1
                right_value, i = parse_factor(i)
                value /= right_value
        return value, i
    
    def parse_factor(i):
        if expression[i] == '(':
            i += 1  # Skip '('
            value, i = parse_expression(i)
            i += 1  # Skip ')'
        else:
            start_i = i
            while i < len(expression) and (expression[i] == "." or expression[i].isdigit()):
                i += 1
            if "." in expression:
                value = float(expression[start_i:i])
            else:
                value = int(expression[start_i:i])
        return value, i
    
    
    expression = expression.replace(' ', '')
    if expression.startswith("-"):
        expression = "0" + expression
    value, _ = parse_expression(0)
    return value, max_dict


def get_combined_boundary(origin_data, return_dict=False):
    N = 1.6e5
    M = 7.0
    SIGMA = 20000
    origin_eqs = [s for s in re.findall(r'<<(.*)?>>', origin_data["answer"])]
    
    operation_list = [operation for eq1 in origin_eqs for operation in re.findall(r'[\+\-\*/]', eq1.split("=")[0])]
    
    max_time = 0
    
    for eq0 in origin_eqs:
        _, max_dict = evaluate_expression(eq0.split("=")[0])
        if max_time < max_dict["time"]:
            max_time = max_dict["time"]
    
    calculate_boundary = max_time
    
    if len(operation_list) == len(origin_eqs):
        plan_boundary= len([x for x in origin_eqs if not x.strip("0.").startswith("0")])
    else:
        plan_boundary = len(operation_list)
    if return_dict:
        return {
            "plan_boundary": plan_boundary,
            "calculate_boundary": calculate_boundary,
            "combined_boundary": 1/(M/plan_boundary + N/(calculate_boundary + SIGMA)) if plan_boundary != 0 and calculate_boundary + SIGMA != 0 else 0
        }
    return 1/(M/plan_boundary + N/(calculate_boundary + SIGMA))


def get_mm_combined_boundary(origin_data, return_dict=False):
    N1, B1 = 3, 0
    N2, B2 = 7, 0
    N3, B3 = 1, -10
    DOMAIN_PERFORMANCE_DICT = {
        "language-science": 92.20,
        "natural-science": 56.06,
        "social-science": 40.17,
        "physical-commonsense": 88.64,
        "social-commonsense": 74.89,
        "temporal-commonsense": 74.38,
        "algebra": 39.57,
        "geometry": 14.29,
        "theory": 35.29 
    }
    global_boundary = origin_data['global_bound']
    local_boundary = origin_data['local_bound']
    domain_boundary = 100-DOMAIN_PERFORMANCE_DICT[origin_data['topic']]
    if return_dict:
        return {
            "global_boundary": global_boundary,
            "local_boundary": local_boundary,
            "domain_boundary": domain_boundary,
            "combined_boundary": 1/(N1/(global_boundary + B1) + N2/(local_boundary + B2) + N3/(domain_boundary + B3))
        }
    return 1/(N1/(global_boundary + B1) + N2/(local_boundary + B2) + N3/(domain_boundary + B3))


def get_multilingual_combined_boundary(origin_data, return_dict=False):
    N1, B1 = 7, 0
    N2, B2 = 1.6e5, 20000
    N3, B3 = 1, 0
    LANG_DICT = {"en": 72.8, "bn": 33.6, "de": 56.0,
                 "es": 61.2, "fr": 62.0, "ja": 52.8,
                 "ru": 62.0, "sw": 48.0, "te": 7.6, 
                 "th": 42.4, "zh": 60.0}
    global_boundary = origin_data['global_bound']
    max_time = 2
    origin_eqs = [s for s in re.findall(r'<<(.*)?>>', origin_data["answer"])]
    for eq0 in origin_eqs:
        _, max_dict = evaluate_expression(eq0.split("=")[0])
        if max_time < max_dict["time"]:
            max_time = max_dict["time"]
    
    calculate_boundary = max_time
    local_boundary = calculate_boundary# origin_data['local_bound']
    language_boundary = 100-LANG_DICT[origin_data['lang']]
    if return_dict:
        return {
            "global_boundary": global_boundary,
            "local_boundary": local_boundary,
            "language_boundary": language_boundary,
            "combined_boundary": 1/(N1/(global_boundary + B1) + N2/(local_boundary + B2) + N3/(language_boundary + B3)) if global_boundary + B1 != 0 and local_boundary + B2 != 0 and language_boundary + B3 != 0 else 0
        }
    return 1/(N1/(global_boundary + B1) + N2/(local_boundary + B2) + N3/(language_boundary + B3))

def get_multi_hop_boundary(origin_data, return_dict=False):
    N1 = 10
    N2 = 10
    B1 = 0.
    B2 = 0.2
    global_boundary = origin_data['global_bound']
    local_boundary = origin_data['local_bound']
    if return_dict:
        return {
            "global_boundary": global_boundary,
            "local_boundary": local_boundary,
            "combined_boundary": 1/(N1/(global_boundary + B1) + N2/(local_boundary + B2))
        }
    return 1/(N1/(global_boundary + B1) + N2/(local_boundary + B2))