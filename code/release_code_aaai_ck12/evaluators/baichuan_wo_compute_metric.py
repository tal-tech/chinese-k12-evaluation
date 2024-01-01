import os
import re,json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from evaluators.evaluator import Evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

class Baichuan_Evaluator(Evaluator):
    def __init__(self, args):
        super(Baichuan_Evaluator, self).__init__(args)
        if not args.compute_metric:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)

        self.args = args
        self.baichuan_model_path = args.model_path

    def eval_single_gpu(self,ques_list, gpu_id):
        self.model = AutoModelForCausalLM.from_pretrained(self.baichuan_model_path, device_map={"":int(gpu_id)}, torch_dtype=torch.float16, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(self.baichuan_model_path)
            
        new_save_list = []
        for line in tqdm(ques_list):
            single_que_dict = eval(line)
            prompt = self.gen_prompt(single_que_dict,task_type=single_que_dict['task_type'])
            messages = []
            messages.append({"role": "user", "content": prompt})
            response = self.model.chat(self.tokenizer, messages)
            single_que_dict['response'] = response
            new_save_list.append(single_que_dict)
        save_path = f'{self.args.save_path}/{self.args.model_name}-{gpu_id}-{self.args.save_mode_name}.json'
        
        # save_path = f'{self.args.save_path}/{self.args.model_name}-{gpu_id}.json'
        with open(save_path, 'w', encoding='utf-8') as f:
            for item in new_save_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f'save on gpu {gpu_id}')



        
        
        

   