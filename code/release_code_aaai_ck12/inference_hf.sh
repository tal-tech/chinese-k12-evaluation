python eval.py \
    --eval_test_file_path $1 \
    --model_name "chatglm2" \
    --model_path model_path \
    --few_shot_examples_file_path few_shot_examples.json \
    --few_shot_examples_cot_file_path few_shot_cot_examples.json \
    --save_path "eval_results" \
    --save_mode_name $2 \
    --zero_shot True
