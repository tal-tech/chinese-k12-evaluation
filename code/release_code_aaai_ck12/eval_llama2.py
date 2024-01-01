from glob import glob
from evaluator_opt import args,EVALUATOR

def main(args):
    # 读取数据，获得所有题目
    all_questions = []
    all_json_files = glob(args.eval_test_file_path + '/*/*/*.jsonl')
    for single in all_json_files:
        with open(single, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            all_questions.extend(lines)
    # all_questions = all_questions[:10]
    EVALUATOR.eval_single_gpu(all_questions)

if __name__ == "__main__":
    main(args)