'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-10-31 14:35:46
LastEditTime: 2025-05-14 17:38:25
Description: 

'''
import base64
from collections import defaultdict
from copy import deepcopy
import json
import os
import re
import shutil

import PIL.Image
from tqdm import tqdm

def write_jsonl(save_path, file_name, res_list):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, file_name), "w", encoding="utf8") as f:
        for r in res_list:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
       


def read_jsonl(load_path):
    if not os.path.exists(load_path):
        print("Missing PATH: ", load_path)
        return []
    with open(load_path, "r", encoding="utf8") as f:
        res_list = []
        for i, line in enumerate(f):
            try:
                res_list.append(json.loads(line.strip()))
            except:
                print("Error in line :", i)
    return res_list

class M3CoT():
    def __init__(self, data_path="data", enable_image_prepare=False, dataset=None) -> None:
        base_dir = data_path
        data_list = []
        if not os.path.exists("data/images"):
            os.makedirs("data/images", exist_ok=True)
        if dataset is None:
            for sp in ["train", "dev", "test"]:
                obj_list = read_jsonl(os.path.join(base_dir, f"{sp}.jsonl"))
                for obj in obj_list:
                    if "image" not in obj:
                        obj["image"] = obj["image_path"]
                    if isinstance(obj["image"], str):
                        obj["image_path"] = obj["image"]
                        if enable_image_prepare:
                            obj["image"] = PIL.Image.open(obj["image_path"]).convert("RGB")
                    else:
                        obj["image_path"] = os.makedirs(f"data/images/{obj['id']}.png")
                        obj["image"].save(f"data/images/{obj['id']}.png", 'PNG')
                    data_list.append(obj)
        else:
            for sp in ["train", "validation", "test"]:
                obj_list = dataset[sp]
                for obj in obj_list:
                    data_list.append(obj)
            
        self.data = data_list
        self.origin_data = data_list
        self.domains = list(set([data["domain"] for data in data_list]))
        ids = self.get_ids()
        assert len(ids) == len(set(ids))

    def get_domain(self):
        return self.domains
    
    def reset(self):
        self.data = self.origin_data
    
    def select_by_category(self, category_list):
        temp_data = []
        for data in self.data:
            if data["category"] in category_list:
                temp_data.append(data)
        self.data = temp_data
    
    def select_by_split(self, split):
        temp_data = []
        for data in self.data:
            if data["split"] == split:
                temp_data.append(data)
        self.data = temp_data
    
    def select_by_domain(self, domain):
        temp_data = []
        for data in self.data:
            if data["domain"] == domain:
                temp_data.append(data)
        self.data = temp_data
    
    def select_by_topic(self, topic):
        temp_data = []
        for data in self.data:
            if data["topic"] == topic:
                temp_data.append(data)
        self.data = temp_data
    
    def get_category(self, topic_list=None):
        if topic_list is None:
            return list(set([x["category"] for x in self.data]))
        else:
            return list(set([x["category"] for x in self.data if x["topic"] in topic_list]))
    
    def get_topic(self, domain_list=None):
        if domain_list is None:
            return list(set([x["topic"] for x in self.data]))
        else:
            return list(set([x["topic"] for x in self.data if x["domain"] in domain_list]))
    
    def get_ids(self):
        return [x["id"] for x in self.data]
    
    def save_as_scienceqa_format(self, save_dir):
        res_list = {}
        ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
        if not os.path.exists(os.path.join(save_dir, "images")):
            os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        splits = {}
        for data in self.data:
            data = deepcopy(data)
            if "image" in data:
                del data["image"]
            data["answer"] = ANSWER_MAP[data["answer"]]
            data["solution"] = data["rationale"]
            data.pop("rationale")
            data["hint"] = data["context"]
            data.pop("context")
            data["image"] = "image.png"
            data["task"] = "closed choice"
            temp_save_dir = os.path.join(save_dir, "images", data["id"])
            if not os.path.exists(temp_save_dir):
                os.makedirs(temp_save_dir, exist_ok=True)
            if not os.path.exists(os.path.join(temp_save_dir, "image.png")):
                shutil.copy(data["image_path"], os.path.join(temp_save_dir, "image.png"))
            res_list[data["id"]] = data
            if data["split"] not in splits:
                splits[data["split"]] = []
            splits[data["split"]].append(data["id"])
        with open(os.path.join(save_dir, "problems.json"), "w") as f:
            json.dump(res_list, f)
        with open(os.path.join(save_dir, "pid_splits.json"), "w") as f:
            json.dump(splits, f)
        
    def save_as_m3it_format(self, save_dir):
        ALPHA_MAP = ["A", "B", "C", "D", "E", "F", "G"]
        
        for split in ["train", "dev", "test"]:
            self.select_by_split(split)
            res_list = []
            for data in self.data:
                
                obj = {}
                with open(data["image_path"], "rb") as image_file:
                    if "image" in data:
                        del data["image"]
                    obj["image_str"] = base64.b64encode(image_file.read()).decode("utf-8")
                    if data["context"] != "":
                        inp = "[Context]\n" + data["context"] + "\n\n"
                    else:
                        inp = ""
                    inp += "[Question]\n" + data["question"] + "\n[Choices]\n"
                    for i, c in enumerate(data["choices"]):
                        inp += f"({ALPHA_MAP[i]}) {c}\n"
                    obj["input"] = inp
                    obj["outputs"] = data["rationale"]
                    obj["meta"] = {
                        "image_id": data["image_id"],
                        "instance_id": data["id"],
                        "split": data["split"]
                    }
                res_list.append(obj)
            write_jsonl(save_dir, f"{split}.jsonl", res_list)
            self.reset()
            
    def save_as_m3cot_format(self, save_dir):
        
        if not os.path.exists(os.path.join(save_dir, "images")):
            os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        splits = {"train": [], "dev": [], "test": []}
        for data in tqdm(self.data):
            data = deepcopy(data)
            temp_save_path = os.path.join(save_dir, "images", data["id"] +".png")
            if not os.path.exists(temp_save_path):
                if "image_path" in data:
                    shutil.copy(data["image_path"], temp_save_path)
                else:
                    if data["image"] is None:
                        continue
                    data["image"].save(temp_save_path)
                    del data["image"]
            if "image" in data:
                del data["image"]
            data["image_path"] = temp_save_path
            
            splits[data["split"]].append(data)
        for sp in ["train", "dev", "test"]:
            write_jsonl(save_dir, f"{sp}.jsonl", splits[sp])
            
    def save_as_preview_format(self, save_dir):
        res_list = []
        ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
        if not os.path.exists(os.path.join(save_dir, "images")):
            os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        for data in tqdm(self.data):
            data = deepcopy(data)
            if "image" in data:
                del data["image"]
            data["answer"] = ANSWER_MAP[data["answer"]]
            data["solution"] = data["rationale"]
            data.pop("rationale")
            data["hint"] = data["context"]
            data.pop("context")
            data["image"] = "image.png"
            data["task"] = "closed choice"
            data["path"] = f"https://cdn.jsdelivr.net/gh/LightChen233/blog-img/m3cot-image/{data['id']}.png"
            
            temp_save_dir = os.path.join(save_dir, "images")
            if not os.path.exists(os.path.join(temp_save_dir, f"{data['id']}.png")):
                shutil.copy(data["image_path"], os.path.join(temp_save_dir, f"{data['id']}.png"))
            res_list.append(data)
            
        with open(os.path.join(save_dir, "problems.js"), "w", encoding="utf8") as f:
            json.dump(res_list, f, ensure_ascii=False, indent=4)
        

def sort_dict(a_dict,option="value"):
    if option in ["value","key"]:
        result_dict={}
        if option=="key":
            temp_list=list(a_dict.keys())
            temp_list.sort()
            for item in temp_list:
                result_dict[item]=a_dict[item]
        else:
            temp_value_list=list(a_dict.values())
            temp_key_list=list(a_dict.keys())
            for i in range(len(temp_key_list)):
                for j in range(len(temp_key_list)-i-1):
                    if temp_value_list[j]>temp_value_list[j+1]:
                        temp=temp_key_list[j]
                        temp_key_list[j]=temp_key_list[j+1]
                        temp_key_list[j+1]=temp
                        temp=temp_value_list[j]
                        temp_value_list[j]=temp_value_list[j+1]
                        temp_value_list[j+1]=temp
            for key,value in zip(temp_key_list,temp_value_list):
                result_dict[key]=value
        return result_dict
    raise ValueError(option+" is not in option list——[key,value]")

def sort_metric(a_dict):
    res_dict = {}
    res_dict["science"] = sort_dict(a_dict["science"], option="key")
    res_dict["commonsense"] = sort_dict(a_dict["commonsense"], option="key")
    res_dict["mathematics"] = sort_dict(a_dict["mathematics"], option="key")
    return res_dict

ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]
def judge_answer(text, choices, answer):
    if isinstance(answer, int):
        answer = ALPHA_MAP[answer]
    if "[Answer]" in text:
        text = text.split("[Answer]")[-1].split("[Rationale]")[0].split("[Context]")[0]
    pattern = re.compile(r'\(([A-Za-z])\)')
    res = pattern.findall(text)
    if len(res) >= 1:
        pred = res[-1].upper()  # 'A', 'B', ...
    else:
        res = []
        for i, choice in enumerate(choices):
            if choice.lower() in text.lower():
                res.append(ALPHA_MAP[i])
        if len(res) >= 1:
            pred = res[-1]
        else:
            for i, choice in enumerate(choices):
                text = re.sub(r'[\n.,!?]', ' ', text)
                if ALPHA_MAP[i] in text.split(" "):
                    res.append(ALPHA_MAP[i])
            if len(res) >= 1:
                pred = res[-1]
            else:
                for i, choice in enumerate(choices):
                    text = re.sub(r'[\n.,!?]', ' ', text)
                    if ALPHA_MAP[i].lower() in text.split(" "):
                        res.append(ALPHA_MAP[i])
                if len(res) >= 1:
                    pred = res[-1]
                else:
                    pred = "FAILED"
    
    if pred == answer:
        return True
    else:
        return False
    
class MetricData():
    def __init__(self, base_dir, file_name=None) -> None:
        data_list = []
        if file_name is None or file_name == "":
            data_list = read_jsonl(base_dir)
        elif os.path.exists(os.path.join(base_dir, f"{file_name}")):
            data_list = read_jsonl(os.path.join(base_dir, f"{file_name}"))
        else:
            for domain in os.listdir(base_dir):
                for topic in os.listdir(os.path.join(base_dir, domain)):
                    obj_list = read_jsonl(os.path.join(base_dir, domain, topic, f"{file_name}"))
                    for obj in obj_list:
                        obj["domain"] = domain
                        obj["topic"] = topic
                        data_list.append(obj)
        self.data = data_list
        self.origin_data = data_list
        self.domain_list = ["science", "commonsense", "mathematics"]
        self.topic_list = ["language-science", "natural-science", "social-science", "physical-commonsense", "social-commonsense", "temporal-commonsense", "algebra", "geometry", "theory"]

    def select_by_ids(self, id_list):
        temp_data = []
        for data in self.data:
            if data["id"] in id_list:
                temp_data.append(data)
        self.data = temp_data
    
    def set_by_id(self, id, value):
        for i, d in enumerate(self.data):
            if d["id"] == id:
                self.data[i] = value
                break
        for i, d in enumerate(self.origin_data):
            if d["id"] == id:
                self.origin_data[i] = value
                break
        
    
    def reset(self):
        self.data = self.origin_data

    def get_domain_list(self):
        return self.domain_list

    def get_topic_list(self):
        return self.topic_list

    def metric(self,
               by="all",
               map:M3CoT=None):
        total = {}
        correct = {}
        if len(self.data) == 0:
            return 0
        idx_list = []
        for obj_ in self.data:
            obj = obj_['origin']
            if map is not None:
                if obj["id"] in idx_list:
                    continue
                else:
                    idx_list.append(obj["id"])
            if map is not None:
                flag = False
                for d in map.data:
                    if d["id"] == obj["id"]:
                        obj["domain"] = d["domain"]
                        obj["topic"] = d["topic"]
                        if "choices" not in obj:
                            obj["choices"] = d["choices"]
                        obj["answer"] = d["answer"]
                        flag = True
                        break
                if not flag:
                    continue
            
            pred_text = obj_['pred'][1]['content'][0]['text']
            if obj["domain"] not in correct.keys():
                correct[obj["domain"]] = {}
            if obj["topic"] not in correct[obj["domain"]].keys():
                correct[obj["domain"]][obj["topic"]] = 0
            if obj["domain"] not in total.keys():
                total[obj["domain"]] = {}
            if obj["topic"] not in total[obj["domain"]].keys():
                total[obj["domain"]][obj["topic"]] = 0    
            
            if judge_answer(pred_text, obj["choices"], obj["answer"]):
                correct[obj["domain"]][obj["topic"]] += 1
            # elif obj["domain"] == "commonsense":
            #     print(obj)
            total[obj["domain"]][obj["topic"]] += 1
        if by == "all":
            _total = sum([total[key1][key2] for key1 in total for key2 in total[key1]])
            _correct = sum([correct[key1][key2] for key1 in total for key2 in correct[key1]])
            return {"total": {"acc": _correct * 1.0 / _total, "total": _total, "correct": _correct}}
        elif by == "domain":
            res_dict = {}
            for key1 in total:
                _total = sum([total[key1][key2] for key2 in total[key1]])
                _correct = sum([correct[key1][key2] for key2 in correct[key1]])
                res_dict[key1] = {"acc": _correct * 1.0 / _total, "total": _total, "correct": _correct}
            res_dict = sort_metric(res_dict)
            return res_dict
        elif by == "topic":
            res_dict = defaultdict(defaultdict)
            for key1 in total:
                for key2 in total[key1]:
                    _total = sum([total[key1][key2]])
                    _correct = sum([correct[key1][key2]])
                    res_dict[key1][key2] = {"acc": _correct * 1.0 / _total, "total": _total, "correct": _correct}
            res_dict = sort_metric(res_dict)
            return res_dict
        else:
            raise ValueError

