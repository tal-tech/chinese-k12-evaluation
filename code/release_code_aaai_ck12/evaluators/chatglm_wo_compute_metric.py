import os
import re,json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from evaluators.evaluator import Evaluator

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

class ChatGLM_Evaluator(Evaluator):
    def __init__(self, args):
        super(ChatGLM_Evaluator, self).__init__(args)
        if not args.compute_metric:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, mirror="tuna")
            # self.model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, mirror="tuna", resume_download=True).half()
        self.args = args

    def gen_prompt(self, single_que_dict, task_type):
        zero_shot = self.args.zero_shot
        few_shot = self.args.few_shot
        zero_shot_cot = self.args.zero_shot_cot
        few_shot_cot = self.args.few_shot_cot

        que = single_que_dict['prompt']
        if few_shot:
            pre_tag = self.gen_few_shot_examples(single_que_dict)
            que = pre_tag + '\n' + que

        if few_shot_cot:
            pre_tag = self.gen_few_shot_cot_examples(single_que_dict)
            que = pre_tag + '\n' + que


        options = single_que_dict['answer_option']

        if task_type == '单选':
            prompt = f'下面是一道选择题，题干之后是四个选项，请从ABCD中选择出正确的选项，直接给出正确选项对应的结果即可。\n题目: {que}'
            prompt += ''.join([option for option in options])
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，并且回答的格式应为“正确答案是：”'
                if zero_shot_cot:
                    prompt += '\n让我们一步一步思考。'

        elif task_type == '多选':
            prompt = f'下面是一道多选题，注意答案可能不止一个，题干之后是四个选项，请从ABCD中选择出正确的选项，直接给出正确选项对应的结果即可。\n题目: {que}'
            prompt += ''.join([option for option in options])
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，并且回答的格式应为“正确答案是：”'
                if zero_shot_cot:
                    prompt += '\n让我们一步一步思考。'

        elif task_type == '填空':
            prompt = '下面是一道填空题，直接给出题目中 ”$$\\underline{}$$“ 处应该填写的正确答案即可。\n题目:' + que
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，每个”$$\\underline{}$$“都需要回答，并且回答的格式应为“正确答案是：”'
                if zero_shot_cot:
                    prompt += '\n让我们一步一步思考。'

        elif task_type == '判断':
            prompt = f'下面是一道或几道判断题，每道题判断描述是否正确，正确的回答‘T’，错误的回答‘F’。\n题目: {que}'
            prompt += ''.join([option for option in options])
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，仅需要回答正确或者错误，你给出的答案格式应为“1.  2.  3.”'
                if zero_shot_cot:
                    prompt += '\n让我们一步一步思考。'
                    
        elif task_type == '排序':
            prompt = f'题目: {que}'
            prompt += ''.join([option for option in options])
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，仅需要回答正确顺序对应的选项，你给出的答案格式应为“正确的顺序应该是：”'
                if zero_shot_cot:
                    prompt += '\n让我们一步一步思考。'


        elif task_type == '完形填空':
            prompt = f'下面是一道完形填空题，首先会给出一段文字，其中有若干数量的空白需要填充，题干之后是所有空白处的选项，每个空白处对应四个选项，请从这些选项中选择出每个空白处应该填写正确的选项，直接给出正确选项对应的结果即可。\n题目: {que}'
            
            que_cnt = len(single_que_dict['answer'])
            for i in range(que_cnt):
                prompt += f'{i+1}. '
                prompt += ' '.join([option for option in options[i*4:(i+1)*4]]) + '\n'
                
            # prompt += ''.join([option for option in options])
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，并且回答的格式应为“正确答案是：'
                if zero_shot_cot:
                    prompt += '\n让我们一步一步思考。'

        else:
            prompt = f'下面是一道试题，请直接给出该题目的答案。\n题目: {que}. \n特别注意，不需要给解析，并且回答的格式应为“正确答案是：（）”'

        if few_shot or few_shot_cot:
            prompt += '\n正确答案是：'
            if few_shot_cot:
                prompt += '\n让我们一步一步思考。'


        prompt.replace('\n\n', '\n')
        return prompt

    def eval_single_gpu(self,ques_list, gpu_id):

        self.model = AutoModel.from_pretrained(self.args.model_path, device_map={"":int(gpu_id)}, resume_download=True, trust_remote_code=True).half()
        model = self.model.to(f'cuda:{gpu_id}')

        new_save_list = []
        for line in tqdm(ques_list):
            single_que_dict = eval(line)
            prompt = self.gen_prompt(single_que_dict,task_type=single_que_dict['task_type'])
            if single_que_dict['task_subject'] == '英语':
                prompt = '请用英语回答下面的问题：\n' + prompt
            response, history = model.chat(self.tokenizer, prompt, history=[])
            single_que_dict['response'] = response
            new_save_list.append(single_que_dict)
        
        save_path = f'{self.args.save_path}/{self.args.model_name}-{gpu_id}-{self.args.save_mode_name}.json'

        with open(save_path, 'w', encoding='utf-8') as f:
            for item in new_save_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f'save on gpu {gpu_id}')



        
        
        

   