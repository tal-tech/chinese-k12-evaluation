
import argparse,os
from evaluators.chatglm_wo_compute_metric import ChatGLM_Evaluator
from evaluators.baichuan_wo_compute_metric import Baichuan_Evaluator
from evaluators.llama2 import LLAMA2_Evaluator

parser = argparse.ArgumentParser()
parser.add_argument("--ntrain", "-k", type=int, default=5)
parser.add_argument("--openai_key", type=str,default="xxx")
parser.add_argument("--minimax_group_id", type=str,default="xxx")
parser.add_argument("--minimax_key", type=str,default="xxx")

parser.add_argument("--compute_metric", default=False, type=bool)
parser.add_argument("--model_name",default='chatglm2',type=str)
parser.add_argument("--model_path",default='',type=str)

parser.add_argument("--zero_shot", default=True, type=bool)
parser.add_argument("--few_shot", default=False, type=bool)
parser.add_argument("--zero_shot_cot",default=False, type=bool)
parser.add_argument("--few_shot_cot",default=False, type=bool)

parser.add_argument("--max_tokens",default=128, type=int)

parser.add_argument("--subject","-s",type=str,default="operating_system")

parser.add_argument("--cuda_device", default= ['0','1','2','3','4','5','6','7'], type=list)  
parser.add_argument("--world_size", default = 8, type=int)  
parser.add_argument("--eval_test_file_path", default = '', type=str)  
parser.add_argument("--few_shot_examples_file_path", default = '', type=str)  
parser.add_argument("--few_shot_examples_cot_file_path", default = '', type=str)  
    
parser.add_argument("--save_path",type=str,default="eval_results")
parser.add_argument("--save_mode_name",type=str,default="zero_shot")

args = parser.parse_args()
os.makedirs(args.save_path, exist_ok=True)


if "chatglm2" in args.model_name:
    EVALUATOR=ChatGLM_Evaluator(args=args)
    
elif "baichuan" in args.model_name:
    EVALUATOR=Baichuan_Evaluator(args=args)

elif "llama2" in args.model_name:
    EVALUATOR=LLAMA2_Evaluator(args=args)
