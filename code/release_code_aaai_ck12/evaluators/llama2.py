



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
from transformers import AutoTokenizer, LlamaForCausalLM
from vllm import LLM, SamplingParams


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

class LLAMA2_Evaluator(Evaluator):
    def __init__(self, args):
        super(LLAMA2_Evaluator, self).__init__(args)
        if not args.compute_metric:
            self.model = LLM(model=args.model_path,tensor_parallel_size=8)

        self.args = args

    def eval_single_gpu(self,ques_list):
        # model = self.model.to('cuda')
            
        new_save_list = []
        for line in tqdm(ques_list):
            single_que_dict = eval(line)
            prompt = self.gen_prompt_llama2(single_que_dict,task_type=single_que_dict['task_type'])
            if single_que_dict['task_subject'] == '英语':
                prompt = '请用英语回答下面的问题：\n' + prompt

            # Generate
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=self.args.max_tokens )
            completions = self.model.generate(prompt, sampling_params)
            
            res_completions = []
            # print('---',len(completions))
            for output in completions:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                single_que_dict['response'] = generated_text
                res_completions.append(single_que_dict)

            new_save_list.append(res_completions)
        # 
        save_path = f'{self.args.save_path}/{self.args.model_name}-{self.args.save_mode_name}.json'
        with open(save_path, 'w', encoding='utf-8') as f:
            for item in new_save_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f'saved to {save_path}')