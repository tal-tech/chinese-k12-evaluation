import re,json
import string

class Evaluator:
    def __init__(self, args):
        self.args = args
        out_file_name = args.few_shot_examples_file_path
        f =  open(out_file_name, 'r')
        ex = json.load(f)
        self.few_shot_examples_all = ex

        f2 = open(args.few_shot_examples_cot_file_path, 'r')
        ex2 = json.load(f2)
        self.few_shot_examples_cot_all = ex2

        self.total_count = {}


    def gen_few_shot_examples(self, single_que_dict):
        subject = single_que_dict['task_subject']
        task_type = single_que_dict['task_type']
        options = single_que_dict['answer_option']
        all_options = ' '.join(options)
        tokens_cnt = 0
        tokens_cnt_self = len(all_options) + len(single_que_dict['prompt'])
        need_cnt = 2048 - tokens_cnt_self

        few_shot_examples = self.few_shot_examples_all[subject][task_type]
        few_shot_cnt = len(' '.join(ex['prompt'] for ex in few_shot_examples))
        tokens_cnt += few_shot_cnt

        if tokens_cnt > need_cnt:
            few_shot_examples = few_shot_examples[:3]
        tokens_cnt2 = 0
        few_shot_cnt = len(' '.join(ex['prompt'] for ex in few_shot_examples))
        tokens_cnt2 += few_shot_cnt
        if tokens_cnt2 > need_cnt:
            few_shot_examples = [few_shot_examples[0]]


        pre_tag = ''
        for i in range(len(few_shot_examples)):
            pre_tag += f'第{i+1}个例子：\n'
            prompt = few_shot_examples[i]['prompt']
            ans = few_shot_examples[i]['answer']
            pre_tag += f'{prompt} \n正确答案是：{ans} \n'
        pre_tag += '问题：\n'
        return pre_tag

    def gen_few_shot_cot_examples(self, single_que_dict):
        subject = single_que_dict['task_subject']
        task_type = single_que_dict['task_type']

        few_shot_cot_examples = self.few_shot_examples_cot_all[subject][task_type][:5]

        options = single_que_dict['answer_option']
        all_options = ' '.join(options)
        tokens_cnt = 0
        tokens_cnt_self = len(all_options) + len(single_que_dict['prompt'])
        need_cnt = 2048 - tokens_cnt_self

        few_shot_cnt = len(' '.join(ex['prompt'] for ex in few_shot_cot_examples))
        few_shot_cnt += len(' '.join(ex['analysis'] for ex in few_shot_cot_examples))
        tokens_cnt += few_shot_cnt

        if tokens_cnt > need_cnt:
            few_shot_cot_examples = few_shot_cot_examples[:3]
        tokens_cnt2 = 0
        few_shot_cnt = len(' '.join(ex['prompt'] for ex in few_shot_cot_examples))
        few_shot_cnt += len(' '.join(ex['analysis'] for ex in few_shot_cot_examples))

        tokens_cnt2 += few_shot_cnt
        if tokens_cnt2 > need_cnt:
            if len(few_shot_cot_examples ) == 0:
                few_shot_cot_examples = []
            else:
                few_shot_cot_examples = [few_shot_cot_examples[0]]
                
        pre_tag = ''
        # try:
        for i in range(len(few_shot_cot_examples)):
            pre_tag += f'第{i+1}个例子：\n'

            prompt = few_shot_cot_examples[i]['prompt']
            ans = few_shot_cot_examples[i]['answer']
            analysis_list = few_shot_cot_examples[i]['analysis'].split('\n\n')

            analysis = '1. ' + analysis_list[0] + '\n'
            ana_cnt = len(analysis_list)
            for i in range(1,ana_cnt):
                analysis += f'{i+1}. {analysis_list[i]}\n'

            pre_tag += f'\n{prompt} \n正确答案是：\n让我们一步一步思考，\n{analysis}所以答案是：{ans} \n'
        # except:
        #     print('$'*100)
        #     print(few_shot_cot_examples)
        pre_tag += '问题：\n'
    
        return pre_tag


    def gen_prompt_llama2(self, single_que_dict, task_type):
        zero_shot = self.args.zero_shot
        few_shot = self.args.few_shot
        zero_shot_cot = self.args.zero_shot_cot
        few_shot_cot = self.args.few_shot_cot

        # print('test-mode', '*'*20)
        # print('zero_shot','few_shot','zero_shot_cot','few_shot_cot')
        # print(zero_shot,few_shot,zero_shot_cot,few_shot_cot)
        
        que = single_que_dict['prompt']
        if few_shot:
            pre_tag = self.gen_few_shot_examples(single_que_dict)
            que = pre_tag + '\n' + que

        if few_shot_cot:
            pre_tag = self.gen_few_shot_cot_examples(single_que_dict)
            que = pre_tag + '\n' + que


        options = single_que_dict['answer_option']

        if task_type == '单选':
            if few_shot or few_shot_cot:
                prompt = f'请选出下列单选题的正确答案，直接给出正确选项对应的结果即可。\n{que}'
            else:  
                prompt = f'下面是一道选择题，题干之后是四个选项，请从ABCD中选择出正确的选项，直接给出正确选项对应的结果即可。\n题目: {que}'
            prompt += ''.join([option for option in options])
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，并且回答的格式应为“正确答案是：”\n答案：'
                if zero_shot_cot:
                    prompt += '\n答案：让我们一步一步思考。'

        elif task_type == '多选':
            if few_shot or few_shot_cot:
                prompt = f'请选出下列多选题的正确答案，直接给出正确选项对应的结果即可。\n{que}'
            else: 
                prompt = f'下面是一道多选题，注意答案可能不止一个，题干之后是四个选项，请从ABCD中选择出正确的选项，直接给出正确选项对应的结果即可。\n题目: {que}'
            prompt += ''.join([option for option in options])
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，并且回答的格式应为“正确答案是：”\n答案：'
                if zero_shot_cot:
                    prompt += '\n答案：让我们一步一步思考。'

        elif task_type == '填空':
            if few_shot or few_shot_cot:
                prompt = f'请给出下列填空题的正确答案，直接给出空白处应该填写的内容即可。\n{que}'
            else: 
                prompt = '下面是一道填空题，直接给出题目中 ”$$\\underline{}$$“ 处应该填写的正确答案即可。\n题目:' + que
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，每个”$$\\underline{}$$“都需要回答，并且回答的格式应为“正确答案是：”\n答案：'
                if zero_shot_cot:
                    prompt += '\n答案：让我们一步一步思考。'

        elif task_type == '判断':
            if few_shot or few_shot_cot:
                prompt = f'请给出下列判断题的正确答案。\n{que}'
            else: 
                prompt = f'下面是一道或几道判断题，每道题判断描述是否正确，正确的回答‘T’，错误的回答‘F’。\n题目: {que}'
            prompt += ''.join([option for option in options])
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，仅需要回答正确或者错误，你给出的答案格式应为“1.  2.  3.”\n答案：'
                if zero_shot_cot:
                    prompt += '\n答案：让我们一步一步思考。'
                    
        elif task_type == '排序':
            if few_shot or few_shot_cot:
                prompt = f'请给出下列排序题的正确答案。\n{que}'
            else: 
                prompt = f'题目: {que}'
            prompt += ''.join([option for option in options])
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，仅需要回答正确顺序对应的选项，你给出的答案格式应为“正确的顺序应该是：”\n答案：'
                if zero_shot_cot:
                    prompt += '\n答案：让我们一步一步思考。'


        elif task_type == '完形填空':
            if few_shot or few_shot_cot:
                prompt = f'请给出下列完形填空题的正确答案。\n{que}'
            else: 
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
            prompt += '\n正确答案是：\n答案：'
            if few_shot_cot:
                prompt += '\n答案：让我们一步一步思考。'


        prompt.replace('\n\n', '\n')
        return prompt
    


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
            if few_shot or few_shot_cot:
                prompt = f'请选出下列单选题的正确答案，直接给出正确选项对应的结果即可。\n{que}'
            else:  
                prompt = f'下面是一道选择题，题干之后是四个选项，请从ABCD中选择出正确的选项，直接给出正确选项对应的结果即可。\n题目: {que}'
            prompt += ''.join([option for option in options])
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，并且回答的格式应为“正确答案是：”'
                if zero_shot_cot:
                    prompt += '\n让我们一步一步思考。'

        elif task_type == '多选':
            if few_shot or few_shot_cot:
                prompt = f'请选出下列多选题的正确答案，直接给出正确选项对应的结果即可。\n{que}'
            else: 
                prompt = f'下面是一道多选题，注意答案可能不止一个，题干之后是四个选项，请从ABCD中选择出正确的选项，直接给出正确选项对应的结果即可。\n题目: {que}'
            prompt += ''.join([option for option in options])
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，并且回答的格式应为“正确答案是：”'
                if zero_shot_cot:
                    prompt += '\n让我们一步一步思考。'

        elif task_type == '填空':
            if few_shot or few_shot_cot:
                prompt = f'请给出下列填空题的正确答案，直接给出空白处应该填写的内容即可。\n{que}'
            else: 
                prompt = '下面是一道填空题，直接给出题目中 ”$$\\underline{}$$“ 处应该填写的正确答案即可。\n题目:' + que
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，每个”$$\\underline{}$$“都需要回答，并且回答的格式应为“正确答案是：”'
                if zero_shot_cot:
                    prompt += '\n让我们一步一步思考。'

        elif task_type == '判断':
            if few_shot or few_shot_cot:
                prompt = f'请给出下列判断题的正确答案。\n{que}'
            else: 
                prompt = f'下面是一道或几道判断题，每道题判断描述是否正确，正确的回答‘T’，错误的回答‘F’。\n题目: {que}'
            prompt += ''.join([option for option in options])
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，仅需要回答正确或者错误，你给出的答案格式应为“1.  2.  3.”'
                if zero_shot_cot:
                    prompt += '\n让我们一步一步思考。'
                    
        elif task_type == '排序':
            if few_shot or few_shot_cot:
                prompt = f'请给出下列排序题的正确答案。\n{que}'
            else: 
                prompt = f'题目: {que}'
            prompt += ''.join([option for option in options])
            if zero_shot or zero_shot_cot:
                prompt += '\n特别注意，不需要给解析，仅需要回答正确顺序对应的选项，你给出的答案格式应为“正确的顺序应该是：”'
                if zero_shot_cot:
                    prompt += '\n让我们一步一步思考。'


        elif task_type == '完形填空':
            if few_shot or few_shot_cot:
                prompt = f'请给出下列完形填空题的正确答案。\n{que}'
            else: 
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
    
    def retrieve_answer(self,model_response,task_type):
        ans_list = []
        pattern = []
        if task_type == '单选':
            pattern=[
                r"答案是：([A-Z]+)(?=[。.“\"\s]|$)",
                r"([A-Z]+)(?=[.。“\"\s]|$)",
            ]
            ans_list = self._match(pattern, model_response, type='Search')
            # print (model_response)
            # print (ans_list)
        elif task_type == '多选':
            # pattern=[
            #     r"([A-Z]+)(?=[.。“\"\s])",
            # ]
            pattern=[
                r"正确答案：([A-Z,、，.\s和]+)(?=[.。“\"\s,，/、（]|$)",
                r"正确答案是：([A-Z,、，.\s和]+)(?=[.。“\"\s,，/、（]|$)",
                r"正确答案是\s?([A-Z,、，.\s和]+)(?=[.。“\"\s,，/、（]|$)",
                r"Correct answer:([A-Z,、，.\s和]+)(?=[.。“\"\s,，/、（]|$)",
                r"Correct answer is\s?([A-Z,、，.\s和]+)(?=[.。“\"\s,，/、（]|$)",
                r"correct answer is\s?([A-Z,、，.\s和]+)(?=[.。“\"\s,，/、（]|$)",
                r"本题选\s?([A-Z,、，.\s和]+)(?=[.。“\"\s,，/、（]|$)",
                r"正确答案为\s?([A-Z,、，.\s和]+)(?=[.。“\"\s,，/、（]|$)",
                r"([A-Z,、，.\s和]+)\s?正确",
                r"选\s?([A-Z,、，.\s和]+)",
                r"([A-Z,、，.\s和]+)(?=[.。“\"\s,，/、（]|$)",
            ]
            ans_list = self._match(pattern, model_response, type='Search')
            # print (model_response)
            # print (ans_list)


        elif task_type == '填空':
            '''
            填空题还是不好提取，特殊情况太多了
            '''
            pattern=[
                r"答案是：(.+?)(?=[。（>]|$)",
                r"(.+?)(?=[。（>]|$)",
            ]
            ans_list = self._match(pattern, model_response, type='Search')
            # print (model_response)
            # print (ans_list)
            
        elif task_type == '判断':
            '''
            判断还需要将每一个小题给拆开
            '''
            pattern=[
                r"[FT]",
                r"[错对]",
                r"[×√]",
            ] 
            ans_list = self._match(pattern, model_response, type='FindAll') 
            # print (model_response)
            # print (ans_list)
        elif task_type == '排序':
            '''
            排序有时候需要去重一下
            '''
            pattern = [
                r"：([A-Z,、，.\s]+)(?=[。.\s]|$)",
                r":([A-Z,、，.\s]+)(?=[。.\s]|$)",
                r"^([A-Z,、，.\s]+)(?=[。.\s]|$)",
                r"([A-Z,、，.\s]+)",
                r"(\d+?)\."
            ]
            ans_list = self._match(pattern, model_response, type='FindAll')
            # print (model_response)
            # print (ans_list)
        elif task_type == '完形填空':
            '''
            只有英语有完形填空，题目有点问题，我这边跑出来的基本都是错误的，需要增加指代
            '''
            pass
        else:
            print(task_type)

        return ans_list

    def _match(self, pattern, string, type='Search'):
        ans_list = []
        for p in pattern:
            if type == 'Search': ## 只匹配一次
                match=re.search(p,string)
                if match:
                    ans_list = match.group() ## 匹配到就跳出
                    break
            elif type == 'FindAll': ## 匹配多次
                ans_list=re.findall(p,string)
                if len(ans_list) != 0: ## 一旦匹配到就跳出返回
                    break
        return ans_list

    def compute_acc(self,RightAnswer, ModelAnswer, type):
        
        if type not in self.total_count:
            self.total_count[type] = {'correct_count': 0 , 'total_count': 0}
            
        if type == '单选':
            
            self.total_count[type]['total_count'] += 1
            ModelAnswer = ''.join(ModelAnswer)
            ModelAnswer = re.findall(r"[A-Z]+",ModelAnswer)
            ModelAnswer = ''.join(ModelAnswer)
            # print (f'right: {RightAnswer}')
            # print (f'model: {ModelAnswer}')
            if RightAnswer == ModelAnswer:
                self.total_count[type]['correct_count'] += 1
            #     print ('correct')
            # print ('-----------------------')
        elif type == '多选':
            '''多选题要全部选项都对才算对
            '''
            
            self.total_count[type]['total_count'] += 1
            ## split modelanswer 
            ModelAnswer_ = []
            
            if isinstance(ModelAnswer, list):
                ModelAnswer = ''.join(ModelAnswer)
            ModelAnswer = re.findall(r"[A-Z]+",ModelAnswer)
            ModelAnswer = ''.join(ModelAnswer)
            

            if len(ModelAnswer) > 1:
                ModelAnswer_.extend(list(ModelAnswer))
            elif len(ModelAnswer) == 1:
                ModelAnswer_.append(ModelAnswer)
            else:
                pass
            ## set 一下
            ModelAnswer_ = list(set(ModelAnswer_))
            
            # print (f'right: {RightAnswer}')
            # print (f'model: {ModelAnswer_}')
            ## 判定每一个modelanswer是否存在rightanswer里面，并且是否全  
            incorrect_flag = 0
            count = 0
            for split_modelanswer in ModelAnswer_:
                if split_modelanswer not in RightAnswer:
                    incorrect_flag = 1
                    break
                else:
                    count += 1
            if not incorrect_flag and count == len(RightAnswer):
                self.total_count[type]['correct_count'] += 1
            #     print ('correct')
            # print ('-----------------------')
                
        elif type == '填空':
            '''填空题的子题分开判定？
            填空题只要rightanswer在解析出来的modelanswer里面就算对
            '''
            self.total_count[type]['total_count'] += 1
            RightAnswer = RightAnswer.split(';') if ';' in RightAnswer else RightAnswer
            # print (f'right: {RightAnswer}')
            # print (f'model: {ModelAnswer}')
            count = 0
            for rightanswer in RightAnswer:
                if rightanswer in ModelAnswer:
                    count += 1
            if count == len(RightAnswer):
                self.total_count[type]['correct_count'] += 1
            #     print ('correct')
            # print ('-----------------------')
            
        elif type == '判断':
            '''判断题拆开小题判断, 同时把一些其他符号映射到T和F上面
            '''
            map2TF = {'错':'F', '对':'T', '√':'T', '×':'F'}
            for rightanswer, modelanswer in zip(RightAnswer,ModelAnswer):
                self.total_count[type]['total_count'] += 1
                modelanswer = map2TF[modelanswer] if modelanswer not in ['T','F'] else modelanswer
                # print (f'right: {rightanswer}')
                # print (f'model: {modelanswer}')
                if rightanswer == modelanswer:
                    self.total_count[type]['correct_count'] += 1
                #     print('correct')
                # print ('-----------------------')
                    
        elif type == '排序':
            '''排序全对才算对
            '''
            self.total_count[type]['total_count'] += 1
            ModelAnswer = ''.join(ModelAnswer)
            ModelAnswer = re.findall(r"[A-Z]+",ModelAnswer)
            ModelAnswer = ''.join(ModelAnswer)
            
            # print (f'right: {RightAnswer}')
            # print (f'model: {ModelAnswer}')

            if RightAnswer == ModelAnswer:
                self.total_count[type]['correct_count'] += 1
            #     print('correct')
            # print ('-----------------------')
        elif type == '完形填空':
            '''之后再完善
            '''
            pass


    def get_metrics(self,single_que_dict):
        
        # task_type = single_que_dict['task_type']
        # subject = single_que_dict['task_subject']
        # right_answer = single_que_dict['answer']
        # model_response = single_que_dict['response'].replace('\n',' ')
        # print(single_que_dict)
        task_type = single_que_dict['task_type']
        subject = single_que_dict['task_subject']
        right_answer = single_que_dict['answer']
        # model_response = single_que_dict['response'].replace('\n',' ')  #这里改动了，因为chatglm3可能生成字典格式，存在response中
        model_response = single_que_dict['response']
        if isinstance(model_response, str):
            model_response = model_response.replace('\n',' ')
        else:
            model_response = model_response['content'].replace('\n',' ')
            
            print('sigle_que_dict',single_que_dict)
            print(model_response)
            print(f"Unexpected type: {type(model_response)}")



        model_answer = self.retrieve_answer(model_response,task_type)
        self.compute_acc(right_answer,model_answer,task_type)

        # print (f'type: {task_type}')
        # print (f'subject: {subject}')
        # print (f'Right answer: {right_answer}\n')
        # print (f'Model response : {model_response} \n')
        # print (f'Model answer : {model_answer} \n')
        # print ('--------------------------------\n')

        return model_answer