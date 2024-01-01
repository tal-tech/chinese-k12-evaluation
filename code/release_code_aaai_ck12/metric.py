import argparse,os
from evaluator_opt import args
from evaluators.evaluator import Evaluator
import glob
if __name__ =='__main__':
    
    Evaluator_object = Evaluator(args)

    data_list = glob.glob('eval_results/*json')
    for data_ in data_list:
        total_answers = []
        lines = open(data_,'r',encoding='utf-8').readlines()
        for line in lines:
            single_que_dict = eval(line)
            model_answer = Evaluator_object.get_metrics(single_que_dict)
            if len(model_answer) != 0:
                single_que_dict['model_answer'] = model_answer
                total_answers.append(single_que_dict)


        for task, data in Evaluator_object.total_count.items():
            if task == '完形填空':
                continue
            task_acc = data['correct_count'] / data['total_count']
            print (f'{task}:  ACC: {task_acc}')
        
        print(data_)
        print('x')