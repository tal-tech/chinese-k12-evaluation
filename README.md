
## CK12: A Rounded K12 Knowledge Graph Based Benchmark for Chinese Holistic Cognition Evaluation
Weihao You, Pengcheng Wang, Changlong Li, Zhilong Ji†, Jinfeng Bai

## Overview

我们利用了k12教育领域知识图谱构建了一个LLM的评估基准。
该评测集包括500+个一级知识点和1900+个二级知识点，基本涵盖了K12教育领域的全面知识。
本次评估的主要目的是全面评估LLM在中文语境下的高级理解能力和推理能力。
我们的评估包括九个学科、五个不同的问题类型，共有41k个问题。
我们使用了四种不同的测试模式来测试当前主流的LLM（zero-shot、few-shot、 zero-shot-cot、 few-shot-cot）。并对数学的cot结果进行了推理步骤的评测。

![example.jpg](https://github.com/tal-tech/chinese-k12-evaluation/blob/main/images/overview.png)


## News
* **[2023.12.09]** CK12 has been accepted to AAAI 2024 🎉🎉🎉

## Leaderboard
我们的测试集在四种测试模式下，不同模型的表现
single，multi， filling， juding， sorting分别代表单选，多选，填空，判断，排序；single*代表选项被打乱，single**代表增加了10个干扰选项
![example.jpg](https://github.com/tal-tech/chinese-k12-evaluation/blob/main/images/result.png)

### 数学推理子集步骤准确性评估
![example.jpg](https://github.com/tal-tech/chinese-k12-evaluation/blob/main/images/math_results.png)


## data
https://drive.google.com/drive/folders/1pvvhI_y1o5olcxHGGWFqt3SDv3PRX0rd?usp=drive_link

## run testing

### 环境依赖：

torch 2.0.1

transformers 4.33.3

vllm 0.1.3



### huggingface inference

原始数据是 data/original_data 

打乱选项的数据是 data/shuffle_options

增加干扰选项的数据是 data/adding_10_distractor_options

`bash code/release_code_aaai_ck12/inference_hf.sh data/original_data sava_name`

### vllm inference

`bash code/release_code_aaai_ck12/inference_vllm.sh data/original_data sava_name`

### get metric

`python code/release_code_aaai_ck12/metric.py`



## 例子

文科和理科的题目例子
### 政治
![example.jpg](https://github.com/tal-tech/chinese-k12-evaluation/blob/main/images/exp5.png)

### 生物
![example.jpg](https://github.com/tal-tech/chinese-k12-evaluation/blob/main/images/exp4.png)
