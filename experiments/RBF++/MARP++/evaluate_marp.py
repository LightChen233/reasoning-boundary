import datasets
import fire
from prettytable import PrettyTable
from tqdm import tqdm

from utils.mm_tool import M3CoT, MetricData


def run(
        print_latex_format=False,
        metric_by= "topic", # ["topic", "domain", "all"]
        ):
    m3cot = M3CoT(data_path="data/m3cot")
    path_dict = {
        "baseline": "experiments/RBF++/zero-shot/m3cot/gpt-4o-tp-00.jsonl",
        "least-to-most": "experiments/RBF++/least-to-most/result/gpt-4o-tp-00.jsonl",
        "complex-cot": "experiments/RBF++/complex-cot/result/gpt-4o-tp-00.jsonl",
        "MARP": "experiments/RBF++/MARP++/result/marp-gpt-4o-tp-00.jsonl",
        "MARP++": "experiments/RBF++/MARP++/result/marp++-gpt-4o-tp-00.jsonl",
    }
    tb = None
    topic_list = None
    domain_list = None
    for path_key in tqdm(list(path_dict.keys())):
        path = path_dict[path_key]
        metric_data = MetricData(path, None)
        if tb is None:
            if metric_by == "topic":
                topic_list = metric_data.get_topic_list()
                tb = PrettyTable(["prompt"] + topic_list + ["total"])
            elif metric_by == "domain":
                domain_list = metric_data.get_domain_list()
                tb = PrettyTable(["prompt"] + domain_list + ["total"])
        m3cot.select_by_split("test")
        res = metric_data.metric(by=metric_by, map=m3cot)
        if metric_by == "topic":
            topic_performance_list = ["" for x in topic_list]
            for domain in res:
                for topic in res[domain]:
                    topic_performance_list[topic_list.index(topic)]=f"{res[domain][topic]['acc']*100.0:.2f}"
            all_res = metric_data.metric(by="all", map=m3cot)["total"]
            topic_performance_list.append(round(all_res["acc"]*100, 2))
            tb.add_row([path_key] + topic_performance_list)
        elif metric_by == "domain":
            domain_performance_list = ["" for x in domain_list]
            for domain in res:
                domain_performance_list[domain_list.index(domain)] = f"{res[domain]['acc']*100.0:.2f}"
            all_res = metric_data.metric(by="all", map=m3cot)["total"]
            domain_performance_list.append(round(all_res["acc"]*100, 2))
            tb.add_row([path_key] + domain_performance_list)
            
    print(tb)
"""
python evaluate.py --setting zero-shot \
                   --model gpt4v \
                   --prompt cot \
                   --metric_by topic

"""
if __name__ == "__main__":
    fire.Fire(run)