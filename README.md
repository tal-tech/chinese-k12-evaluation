# chinese-k12-evaluation
## 中文k12领域大模型评测集

我们利用了k12教育领域知识图谱构建了一个LLM的评估基准。
该评测集包括500+个一级知识点和1900+个二级知识点，基本涵盖了K12教育领域的全面知识。
本次评估的主要目的是全面评估LLM在中文语境下的高级理解能力和推理能力。
我们的评估包括九个学科、五个不同的问题类型，共有41k个问题。
对于单选和多选我们打乱了选项，并进行了添加干扰项的特殊操作，以进一步评估LLM的知识利用和推理能力。
我们使用了四种不同的测试模式来测试当前主流的LLM（zero-shot、few-shot、 zero-shot-cot、 few-shot-cot）。
我们希望这个评测集可以帮助评估LLM在知识点上的优势和不足，从而促进其在中文语境下的发展。

![benchmark.jpg](https://github.com/youweihao-tal/chinese-k12-evaluation/blob/main/images/benchmark.png)

## 例子
文科和理科的例子
![example.jpg](https://github.com/youweihao-tal/chinese-k12-evaluation/blob/main/images/examples.png)

## 学科分布比例以及数学学科一级知识点分布比例
我们列出了每个学科题目的占比，以及每个一级知识点的题目占比。并且单独列出来数学学科的一级知识点以及二级知识点的占比
![stat.jpg](https://github.com/youweihao-tal/chinese-k12-evaluation/blob/main/images/stat.png)

## 主流模型上的测试结果
我们对主流9个模型进行了测试，其中✳代表结果即将更新
![res.jpg](https://github.com/youweihao-tal/chinese-k12-evaluation/blob/main/images/result.png)
