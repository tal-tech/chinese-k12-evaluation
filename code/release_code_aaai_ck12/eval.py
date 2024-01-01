from tqdm import tqdm
import json,os
import torch
from glob import glob
from itertools import cycle
from evaluator_opt import args

def run(data):
    from evaluator_opt import EVALUATOR
    ques_list, gpu_id = data
    EVALUATOR.eval_single_gpu(ques_list, gpu_id)

def main(args):

    # 读取数据，获得所有题目
    all_questions = []
    questions_list_for_every_gpu = []
    all_json_files = glob(args.eval_test_file_path + '/*/*/*.jsonl')
    for single in all_json_files:
        with open(single, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            all_questions.extend(lines)

    # all_questions = all_questions[:800]
    per_gpu_eval_cnt = ((len(all_questions) // args.world_size * args.world_size) + args.world_size) // args.world_size
    for gpu_id in range(args.world_size):
        questions_list_for_every_gpu.append(all_questions[per_gpu_eval_cnt*(gpu_id) : (gpu_id+1)*per_gpu_eval_cnt])

    # 多卡推理
    ctx = torch.multiprocessing.get_context("spawn")
    p = ctx.Pool(args.world_size)
    device_ids = args.cuda_device
    device_ids = cycle(device_ids)
    for data in tqdm(p.imap_unordered(run, zip(questions_list_for_every_gpu, device_ids))):
        None

    # 合并结果并保存
    all_response = []
    for gpu_id in args.cuda_device:
        res_file = f'{args.save_path}/{args.model_name}-{gpu_id}-{args.save_mode_name}.json'
        with open(res_file, 'r') as f:
            lines = f.readlines()
            all_response.extend(lines)
        os.remove(res_file)
 
    save_path = f'{args.save_path}/{args.model_name}_{args.save_mode_name}.json'
    with open(save_path, 'w') as f:
        for item in all_response:
            f.write(json.dumps(eval(item), ensure_ascii=False) + '\n')



if __name__ == "__main__":
    main(args)
    
    

