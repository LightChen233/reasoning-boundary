<!--
 * @Author: Qiguang Chen
 * @LastEditors: Qiguang Chen
 * @Date: 2024-05-23 20:24:16
 * @LastEditTime: 2024-05-26 18:09:00
 * @Description: 
 * 
-->
<p align="center">
<h1 align="center"> <img src="imgs/image.png" alt="SVG Image" width="40px"> Unlocking the Boundaries of Thought: A Reasoning Granularity Framework to Quantify and Optimize Chain-of-Thought</h1>
</p>
<p align="center">
  	<a href="https://img.shields.io/badge/version-v0.0.1-blue">
      <img alt="version" src="https://img.shields.io/badge/version-v0.0.1-blue?color=FF8000?color=009922" />
    </a>
    <a href="https://github.com/LightChen233/reasoning-boundary/pulls">
       <img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-blue" />
  	</a>
   	<a href="https://github.com/LightChen233/reasoning-boundary/stargazers">
       <img alt="stars" src="https://img.shields.io/github/stars/LightChen233/reasoning-boundary" />
  	</a>
  	<a href="https://github.com/LightChen233/reasoning-boundary/network/members">
       <img alt="FORK" src="https://img.shields.io/github/forks/LightChen233/reasoning-boundary?color=FF8000" />
  	</a>
    <a href="https://github.com/LightChen233/reasoning-boundary/issues">
      <img alt="Issues" src="https://img.shields.io/github/issues/LightChen233/reasoning-boundary?color=0088ff"/>
    </a>
    <br />
</p>

<p align="center">
  	<b>
    | [<a hred="https://proceedings.neurips.cc/paper_files/paper/2024/hash/62ab1c2cb4b03e717005479efb211841-Abstract-Conference.html">NeurIPS(Oral)</a>] | [<a href="https://arxiv.org/abs/2410.05695">ArXiv-RBF</a>] | [<a href="https://arxiv.org/abs/2505.13307">ArXiv-RBF++</a>] | [<a href="https://huggingface.co/datasets/LightChen2333/BigGSM">🤗HuggingFace</a>] |
    </b>
    <br />
</p>

🌟 Any contributions via PRs, issues, emails or other methods are greatly appreciated.

## 🔥News
- 🎖️ **We update our work to [RBF++](https://arxiv.org/abs/2505.13307) and introduce BigGSM++ for quantify the reasoning boundary in multimodal and long chain-of-thought reasoning scenarios (Data is available at [Google Drive](https://drive.google.com/file/d/15tsQfyg6lRzv1Gd3MvmUf-MvlBJj7zHj/view?usp=drive_link)).**
- 🎖️ **Our work is accepted by NeurIPS 2024 (<span style="color:red">Oral</span>).**
- 🔥 **We have release benchmark on \[[🤗HuggingFace](https://huggingface.co/datasets/LightChen2333/BigGSM)\].**
- 🔥 **The paper is also available on \[[ArXiv](https://arxiv.org/abs/2410.05695)\].**

## 💡 Motivation
Chain-of-Thought (CoT) reasoning has emerged as a promising approach for enhancing the performance of large language models (LLMs) on complex reasoning tasks. Recently, a series of studies attempt to explain the mechanisms underlying CoT, aiming to deepen the understanding and enhance its efficacy. Nevertheless, the existing research faces two major challenges:
- (1) **A lack of quantitative metrics to assess CoT capabilities**
- (2) **A dearth of guidance on optimizing CoT performance**.

Motivated by this, in this work, we introduce a novel reasoning granularities (RG) methodological framework to address these challenges. To solve the lack of quantification, we first define an RG to quantify the upper bound of CoT and establish a combination law for RG, enabling a practical quantitative approach applicable to various real-world CoT tasks.
To address the lack of optimization, we propose three categories of RGs. We further optimize these categories with combination laws focused on RG promotion and reasoning path optimization for CoT improvement.
Through extensive experiments on 25 models and 4 tasks, the study validates the existence and rationality of the proposed framework. Furthermore, it explains the effectiveness of 10 CoT strategies and guides optimization from two perspectives.

We hope this work can provide a comprehensive understanding of the boundaries and optimization strategies for reasoning in LLMs.



## 🎯 Installation

### 1. Dataset Preparation
#### Load Dataset from Huggingface
```python 
import datasets
dataset = datasets.load_dataset("LightChen2333/BigGSM")
```

### 2. Install from git
Our code requires `Python>=3.10`
```bash 
git clone https://github.com/LightChen233/reasoning-boundary.git && cd reasoning-boundary/
pip install -r requirements.txt
```
### 3. Evaluation for reproduction
```bash
python evaluate.py --data_split CoT
```
where `--data_split` can be selected from `[CoT, Tool-Usage, PoT, Complex-CoT, LtM, MARP, PoT-MARP, gpt-4o, gpt-4o-MARP, o1-preview]`. 

### 4. Evaluation for your results
```bash
python evaluate.py --data_split custom \
                   --K 0.301 \
                   --K2 0.92 \
                   --mode nl \
                   --result_path [PREDICTION_PATH]
```
`PREDICTION_PATH` consists the results predicted by model which save as `jsonl` format. Among them, each line of file  must meet the following format:
```json
{
    "index": "str",
    "pred": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "str"}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "str"}]
        }
    ],
    "origin": {
        "index": "str",
        "question": "str",
        "answer": "str",
    }
}
```

## 🖨️File Structure

```yaml
root
├── data                            # data folder where the BigGSM dataset is loaded
├── experiment                      # All experimental data
│   ├── RBF                         # Experimental results for RBF.
│   └── RBF++                       # Experimental results under RBF++.
├── utils                           # Tool library folder
│   ├── data.py                     # Dataset loading class
│   ├── request_tool.py             # API request tool
│   └── tools.py                    # Common-used tools
├── draw_bound_*.py                      # Draw reasoning boundary script
└── evaluate_*.py                     # Evaluation script
```

## ✒️ Reference
If you find this project useful for your research, please kindly consider citing the following paper:

```
@inproceedings{chen-etal-2024-rg,
    title = "Unlocking the Boundaries of Thought: A Reasoning Granularity Framework to Quantify and Optimize Chain-of-Thought",
    author = "Chen, Qiguang  and
      Qin, Libo  and
      Jiaqi, Wang  and
      Jinxuan, Zhou  and
      Che, Wanxiang",
    booktitle = "Proc. of NeurIPS",
    year = "2024",
}
```

## 📲 Contact

Please create Github issues here or email [Qiguang Chen](mailto:charleschen2333@gmail.com) if you have any questions or suggestions. 

