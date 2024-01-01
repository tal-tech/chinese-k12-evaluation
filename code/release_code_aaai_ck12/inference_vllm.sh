python eval_llama2.py \
    --eval_test_file_path test_data \
    --model_name llama2 \
    --model_path model_path \
    --few_shot_examples_file_path few_shot_examples.json \
    --few_shot_examples_cot_file_path few_shot_cot_examples.json \
    --save_path eval_results \
    --zero_shot True \
    --max_tokens 128 \
    --save_mode_name zero_shot_128

