import argparse
import json
import os
import random
import time
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DATASET_KEYS = {
    'openai/gsm8k': {'question': 'question', 'answer': 'answer'},
    'hendrycks/competition_math': {'question': 'problem', 'answer': 'solution'},
    'datasets/converted_aime_dataset': {'question': 'problem', 'answer': 'solution'},
    'aime_2025': {'question': 'problem', 'answer': 'solution'},
    'di-zhang-fdu/MATH500': {'question': 'problem', 'answer': 'solution'},
    'datasets/compression_dataset': {'question': 'problem', 'answer': 'solution'},
    "cais/mmlu": {'question': 'question', 'answer': 'answer'},
    "amc": {'question': 'question', 'answer': 'answer'},
    "olympiadbench": {'question': 'question', 'answer': 'final_answer'},
    "minerva_math": {'question': 'problem', 'answer': 'solution'}, 
}


os.environ['TOKENIZERS_PARALLELISM'] = "false"

# ============= Tool functions ==============

def set_random_seed(seed: int) -> None: 
     random.seed(seed) 
     np.random.seed(seed) 
     torch.manual_seed(seed) 
     if torch.cuda.is_available(): 
         torch.cuda.manual_seed_all(seed) 

def read_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results

def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def get_most_common(solns):
    soln_counts = {}
    for soln in solns:
        if len(soln) == 0:
            continue
        found = False
        for key in soln_counts.keys():
            if verify(soln, key):
                soln_counts[key] += 1
                found = True
                break
        if not found:
            # if 判断是list
            if isinstance(soln, list):
                try:
                    soln_counts[soln[1]] = 1
                except:
                    soln_counts[soln[0]] = 1
            else:
                soln_counts[soln] = 1
    if len(soln_counts) == 0:
        return ""
    return max(soln_counts, key=soln_counts.get)


def parse_output(ds, outputs, dataset_name, save_file_name=None):
    QUESTION_KEY = DATASET_KEYS[dataset_name]["question"]
    ANSWER_KEY = DATASET_KEYS[dataset_name]["answer"]
    
    predictions, golds, results = [], [], []
    for input, output in zip(ds, outputs):
        gold = input[ANSWER_KEY]
        if isinstance(gold, list):
            gold = gold[0]
        gold = str(gold)
        prediction = [parse(resp.text)[1] if len(parse(resp.text)) > 1 else '' for resp in output.outputs]
        predictions.append(prediction)
        golds.append(gold)
        results.append({
            QUESTION_KEY: input[QUESTION_KEY],
            ANSWER_KEY: input[ANSWER_KEY],
            "responses": [resp.text for resp in output.outputs],
            "prediction": prediction,
            "gold": gold,
            "tokens": [len(resp.token_ids) for resp in output.outputs],
            "accuracy": [verify(gold, pred) for pred in prediction]
        })
    if save_file_name:
        with open(save_file_name, 'w') as f:
            json.dump(results, f, indent=4)
    return results

def process_single_result(args):
    result, budget, dataset_name, tokenizer = args

    if dataset_name == "olympiadbench":
        gold = str(result['final_answer'][0])
    else:
        gold = str(result['gold'])

    ans_format = "\\boxed{{{}}}"
    if "boxed" in gold:
        gold_extracted = parse(gold)
    else:
        gold_extracted = parse(ans_format.format(gold))

    responses = result["responses"]
    response_ids = result["response_ids"]

    truncated = truncate_responses_by_budget(responses, response_ids, budget, tokenizer)

    preds = []
    for r in truncated:
        try:
            preds.append(parse(r))
        except:
            preds.append(r)

    match_flags = [verify(p, gold_extracted) for p in preds]

    pass_1 = int(match_flags[0]) if match_flags else 0
    most_common = parse("\\boxed{{{}}}".format(get_most_common(preds)))
    maj = int(verify(preds, gold_extracted)) if preds else 0
    avg = float(np.mean(match_flags)) if match_flags else 0

    return pass_1, maj, avg

def process_single_result_no_truncation(args):
    result, dataset_name, n_list = args

    if dataset_name == "olympiadbench":
        gold = str(result['final_answer'][0])
    else:
        gold = str(result['gold'])

    
    ans_format = "\\boxed{{{}}}"
    if "boxed" in gold:
        gold_extracted = parse(gold)
    else:
        gold_extracted = parse(ans_format.format(gold))

    responses = result["responses"]

    preds = []
    for r in responses:
        try:
            preds.append(parse(r))
        except:
            preds.append(r)

    match_flags = [verify(p, gold_extracted) for p in preds]
    
    pass_at_n = {}
    avg_at_n = {}
    maj_at_n = {}

    for n in n_list:
        pass_at_n[f"pass@{n}"] = 1 if any(match_flags[:n]) else 0 # pass@n is first prediction
        avg_at_n[f"avg@{n}"] = float(np.mean(match_flags[:n]))  # average accuracy over top-n

        if n == 1:
            maj_at_n[f"maj@{n}"] = avg_at_n[f"avg@{n}"]

        else:
            maj_pred = parse("\\boxed{{{}}}".format(get_most_common(preds[:n])))

            try:
                maj_at_n[f"maj@{n}"] = int(verify(maj_pred, gold_extracted)) if maj_pred else 0
            except:
                maj_at_n[f"maj@{n}"] = 0
            
    avg_lens = np.mean(result['tokens'])

    return pass_at_n, avg_at_n, maj_at_n, avg_lens

def get_scores_parallel(results, tokenizer, n_list=[], dataset_name="", num_workers=None):
    
    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)

    # no budget and compute average lengths
    args_list = [(result, dataset_name, n_list) for result in results]
    native_scores = []
    
    for result in results:
        native_scores.append(process_single_result_no_truncation((result, dataset_name, n_list)))

    passk, avg_k, maj_k, avg_lengths = zip(*native_scores)

    pass_at_all = {}
    avg_at_all = {}
    maj_at_all = {}
    
    for n in n_list:
        pass_key = f"pass@{n}"
        avg_key = f"avg@{n}"
        maj_key = f"maj@{n}"

        pass_values = [d[pass_key] for d in passk]
        avg_values = [d[avg_key] for d in avg_k]
        maj_values = [d[maj_key] for d in maj_k]

        pass_at_all[pass_key] = float(np.mean(pass_values))
        avg_at_all[avg_key] = float(np.mean(avg_values))
        maj_at_all[maj_key] = float(np.mean(maj_values))

        if n == 1:
            assert np.isclose(pass_at_all[pass_key], avg_at_all[avg_key]), f"pass@1 and avg@1 should be the same, now pass@1={pass_at_all[pass_key]} and avg@1={avg_at_all[avg_key]}"
            assert np.isclose(pass_at_all[pass_key], maj_at_all[maj_key]), f"pass@1 and maj@1 should be the same, now pass@1={pass_at_all[pass_key]} and maj@1={maj_at_all[maj_key]}"


        
    saved_native_scores = {
        **pass_at_all,
        **avg_at_all,
        **maj_at_all,
        'average_length': float(np.mean(avg_lengths))
    }
    
    return saved_native_scores


def evaluate_model(model_path, dataset_config, dataset_name, temperature, max_tokens=8000, gpu_nums=1, llm_instance=None, tokenizer_instance=None):
    dataset = dataset_config['loader'](dataset_name)
    MAX_TOKENS = max_tokens
    # TEST_TEMPERATURE = dataset_config['temperature']
    TEST_TEMPERATURE = limit_toketens
    MAX_TEST_SAMPLES = dataset_config['max_samples']
    dataset_save_name = dataset_config['save_name']
    n_list = dataset_config['n_list']
    TEST_N = n_list[-1]

    QUESTION_KEY = DATASET_KEYS[dataset_name]["question"]
    ANSWER_KEY = DATASET_KEYS[dataset_name]["answer"]
    
    saved_model_name = clean_model_name(model_path)
    print(f"Evaluating model: {saved_model_name}")
    save_path = f"outputs_entropy_2/{saved_model_name}/{dataset_save_name}/{temperature}.json"
    
    if tokenizer_instance:
        tokenizer = tokenizer_instance
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        
    if os.path.exists(save_path):
        print("Found existing results, loading...")
        start_time = time.time()
        test_scores = get_scores_from_file(dataset_name, n_list, save_path, tokenizer)
        end_time = time.time()
        print("Test:", test_scores)
        print("Time taken:", end_time - start_time)
        test_scores['time'] = end_time - start_time
        return test_scores

    test_prompts = []
    if llm_instance:
        model = llm_instance
    else:
        model = LLM(model_path, gpu_memory_utilization=0.85, tensor_parallel_size=gpu_nums)
    try:
        test_ds = dataset['test'].shuffle(seed=0).select(range(min(MAX_TEST_SAMPLES, len(dataset['test']))))
    except:
        try:
            test_ds = dataset['train'].shuffle(seed=0).select(range(min(MAX_TEST_SAMPLES, len(dataset['train']))))
        except:
            test_ds = dataset
    
    for x in test_ds:
        if "llama" in model_path.lower():
            prompt = tokenizer.apply_chat_template([{        "role": "system",        "content": "Please reason step by step, and put your final answer within \\boxed{}."   },{        "role": "user",       "content": x[QUESTION_KEY]  }], tokenize=False, add_generation_prompt=True)
        else:
        
            prompt = apply_qwen_math_template(x[QUESTION_KEY])
        
        
        test_prompts.append(prompt)

    sampling_params = SamplingParams(seed=42, temperature=TEST_TEMPERATURE, max_tokens=MAX_TOKENS, n=TEST_N)
    sampling_params.stop_token_ids = [tokenizer.eos_token_id]

    print("Generating test outputs...")
    # print(tokenizer.decode(test_prompts[0], skip_special_tokens=False))
    print(test_prompts[0])

    start_time = time.time()
    test_outputs = model.generate(test_prompts, sampling_params=sampling_params, use_tqdm=True)

    os.makedirs(f"outputs_entropy_2/{saved_model_name}/{dataset_save_name}", exist_ok=True)

    save_path = f"outputs_entropy_2/{saved_model_name}/{dataset_save_name}/{temperature}.json"
    
    results = parse_output(test_ds, test_outputs, dataset_name, save_path)
    test_scores = get_scores_parallel(results, tokenizer, n_list=n_list, dataset_name=dataset_name, num_workers=cpu_count() - 1)
    end_time = time.time()
    print("Test:", test_scores)
    print("Time taken:", end_time - start_time)
    test_scores['time'] = end_time - start_time
    return test_scores


def clean_model_name(model_path):
    if "checkpoints" in model_path:
        saved_model_name = model_path[model_path.find("checkpoints") + len("checkpoints") + 1:]
        saved_model_name = saved_model_name[saved_model_name.find("/") + 1:]
    elif "ckpt" in model_path:
        saved_model_name = model_path[model_path.find("ckpt") + len("ckpt") + 1:]
        saved_model_name = saved_model_name[saved_model_name.find("/") + 1:]
    else:
        saved_model_name = model_path.split("/")[-1]
    return saved_model_name.replace("/", "_")


def get_scores_from_file(dataset_name, n_list, save_path, tokenizer):
    start_time = time.time()
    with open(save_path, 'r') as f:
        results = json.load(f)
    
    test_scores = get_scores_parallel(results, tokenizer=tokenizer, n_list=n_list, dataset_name=dataset_name, num_workers=cpu_count() - 1)
    end_time = time.time()
    print("Test:", test_scores)
    print("Time taken:", end_time - start_time)
    test_scores['time'] = end_time - start_time
    return test_scores



# ============= Main function ==============

def main(args):
    model_path = args.model_path
    dataset_type = args.dataset
    scale = args.scale
    max_tokens = args.tok_limit
    temperature = args.temperature
    gpu_nums = args.gpus

    print("Dataset:", dataset_type, "\nScale:", scale)        
        
    LlAMA_DATASET_CONFIGS = {
        'all': {
        'aime_2025':{
            'loader': lambda name: load_dataset('path/to/aime_2025'),
            'test_n': 10, 
            'temperature': 1, 
            'max_samples': 100,
            'n': 32,
            'n_list': [1, 4, 8, 16, 32],
            # 'n_list': [1, 16, 32, 64, 128, 256],
            'save_name': 'aime_2025'
        },
        'datasets/converted_aime_dataset': {
            'loader': lambda name: load_from_disk(name),
            'test_n': 10, 
            'temperature': 1, 
            'max_samples': 100,
            'n': 32,
            'n_list': [1, 4, 8, 16, 32],
            # 'n_list': [1, 16, 32, 64, 128, 256],
            'save_name': 'aime'
        },
        'di-zhang-fdu/MATH500': {
            'loader': lambda name: load_dataset('di-zhang-fdu/MATH500'),
            'test_n': 3, 
            'temperature': 1,
            'max_samples': 500,
            'n': 8,
            'n_list': [1, 2, 4, 8, 16],
            'save_name': 'math500'
        },
        'amc': {
            'loader': lambda _: read_jsonl_file("datasets/aimo-validation-amc.jsonl"),
            'test_n': 10, 'temperature': 1, 'max_samples': -1,
            'n_list': [1, 4, 8, 16, 32],
            'n': 32,
            # 'n_list': [1, 8, 16, 32, 64, 128],
            'save_name': 'amc'
        },
        'olympiadbench': {
            'n': 8,
            'loader': lambda _: read_jsonl_file("datasets/olympiadbench/test.jsonl"),
            'test_n': 10, 'temperature': 1, 'max_samples': -1,
            'n_list': [1, 2, 4, 8],             
            # 'n_list': [1, 8, 16, 32, 64, 128],
            'save_name': 'olympiadbench'
        },
    }
    }
    
    DATASET_CONFIGS = {
        'id': {
        'aime_2025':{
            'loader': lambda name: load_dataset('path/to/aime_2025'),
            'test_n': 10, 
            'temperature': 1, 
            'max_samples': 100,
            'n': 32,
            # 'n_list': [1, 4, 8, 16, 32],
            'n_list': [1, 16, 32, 64, 128, 256],
            'save_name': 'aime_2025'
        },
        'datasets/converted_aime_dataset': {
            'loader': lambda name: load_from_disk(name),
            'test_n': 10, 
            'temperature': 1, 
            'max_samples': 100,
            'n': 32,
            # 'n_list': [1, 4, 8, 16, 32],
            'n_list': [1, 16, 32, 64, 128, 256],
            'save_name': 'aime'
        },
        'di-zhang-fdu/MATH500': {
            'loader': lambda name: load_dataset('di-zhang-fdu/MATH500'),
            'test_n': 3, 
            'temperature': 1,
            'max_samples': 500,
            'n': 8,
            'n_list': [1, 2, 4, 8, 16],
            'save_name': 'math500'
        },
        'amc': {
            'loader': lambda _: read_jsonl_file("datasets/aimo-validation-amc.jsonl"),
            'test_n': 10, 'temperature': 1, 'max_samples': -1,
            # 'n_list': [1, 4, 8, 16, 32],
            'n': 32,
            'n_list': [1, 8, 16, 32, 64, 128],
            'save_name': 'amc'
        },
        'olympiadbench': {
            'n': 8,
            'loader': lambda _: read_jsonl_file("datasets/olympiadbench/test.jsonl"),
            'test_n': 10, 'temperature': 1, 'max_samples': -1,
            # 'n_list': [1, 2, 4, 8, 16],             
            'n_list': [1, 8, 16, 32, 64, 128],
            'save_name': 'olympiadbench'
        },
        },
    }
    
    llm_instance = LLM(model_path, gpu_memory_utilization=0.85, tensor_parallel_size=gpu_nums)
    tokenizer_instance = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    
    if 'llama' in model_path.lower():
        DATASET_CONFIGS = LlAMA_DATASET_CONFIGS
    
    if dataset_type == "all":
        results = {}
        
        dataset_name_list = DATASET_CONFIGS.keys()

        print("="*20)
        print()
        print(f"Evaluating datasets: {list(dataset_name_list)}")
        print()
        print("="*20)
        
        for dataset_type in dataset_name_list:
            all_scores = {}
            for dataset_name in DATASET_CONFIGS[dataset_type].keys():
                
                print("-"*20)
                print("Evaluating dataset:", dataset_name)
                
                config = DATASET_CONFIGS[dataset_type][dataset_name]
                dataset_save_name = config['save_name']

                scores = evaluate_model(model_path, config, dataset_name, temperature, max_tokens, gpu_nums, llm_instance, tokenizer_instance)
                print("Finished evaluating dataset:", dataset_name)
                if scores:
                    scores["model_path"] = model_path
                    saved_model_name = clean_model_name(model_path)
                    os.makedirs(f"results_entropy_2/{saved_model_name}/{dataset_save_name}", exist_ok=True)
                    print(scores)

                    save_path = f"results_entropy_2/{saved_model_name}/{dataset_save_name}/{temperature}.json"
                    with open(save_path, 'w') as f:
                        json.dump(scores, f, indent=4)
                    all_scores[dataset_name] = scores
            
            print("All scores:", all_scores)
            print(f"Average scores for **{dataset_type}**")
            
            items = list(all_scores.values())
            # select last item for each dataset 

            all_last_scores = {}
            for key, value in all_scores.items():
                print(key, value)
                n = DATASET_CONFIGS[dataset_type][key]['n_list'][-1]
                all_last_scores[key] = {
                    f"pass@{n}" : value[f"pass@{n}"],
                    f"avg@{n}" : value[f"avg@{n}"],
                    f"maj@{n}" : value[f"maj@{n}"],
                    "average_length": value["average_length"]
                }

            items = list(all_last_scores.values())

            avg_pass = np.mean([list(item.values())[0] for item in items])
            avg_avg = np.mean([list(item.values())[1] for item in items])
            avg_maj = np.mean([list(item.values())[2] for item in items])
            avg_length = np.mean([list(item.values())[3] for item in items])
            print("Pass:", avg_pass)
            print("Avg:", avg_avg)
            print("Maj:", avg_maj)
            print("Length:", avg_length)

            results[dataset_type] = {
                "average_pass": avg_pass,
                "average_avg": avg_avg,
                "average_maj": avg_maj,
                "average_length": avg_length,
                "scores": all_last_scores
            }

        save_path = f"results_entropy_2/{saved_model_name}/all.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
            
            # compute avg here



    else:
        if dataset_type not in DATASET_CONFIGS:
            raise ValueError(f"Dataset config for {dataset_type} is not defined.")

        dataset_name_list = DATASET_CONFIGS[dataset_type]

        print("="*20)
        print()
        print(f"Evaluating datasets: {dataset_name_list}")
        print()
        print("="*20)
        
        all_scores = {}
        for dataset_name in dataset_name_list.keys():
            
            print("-"*20)
            print("Evaluating dataset:", dataset_name)
            
            config = dataset_name_list[dataset_name]
            dataset_save_name = config['save_name']


            scores = evaluate_model(model_path, config, dataset_name,temperature, max_tokens, gpu_nums, llm_instance, tokenizer_instance)
            print("Finished evaluating dataset:", dataset_name)
            if scores:
                scores["model_path"] = model_path
                saved_model_name = clean_model_name(model_path)
                os.makedirs(f"results_entropy_2/{saved_model_name}/{dataset_save_name}", exist_ok=True)
                print(scores)

                save_path = f"results_entropy_2/{saved_model_name}/{dataset_save_name}/{temperature}.json"
                with open(save_path, 'w') as f:
                    json.dump(scores, f, indent=4)
                all_scores[dataset_name] = scores


        results = {}
        print("All scores:", all_scores)
        print(f"Average scores for **{dataset_type}**")
        
        items = list(all_scores.values())
        # select last item for each dataset 

        all_last_scores = {}
        for key, value in all_scores.items():
            print(key, value)
            n = DATASET_CONFIGS[dataset_type][key]['n']
            all_last_scores[key] = {
                f"pass@{n}" : value[f"pass@{n}"],
                f"avg@{n}" : value[f"avg@{n}"],
                f"maj@{n}" : value[f"maj@{n}"],
                "average_length": value["average_length"]
            }

        items = list(all_last_scores.values())

        avg_pass = np.mean([list(item.values())[0] for item in items])
        avg_avg = np.mean([list(item.values())[1] for item in items])
        avg_maj = np.mean([list(item.values())[2] for item in items])
        avg_length = np.mean([list(item.values())[3] for item in items])
        print("Pass:", avg_pass)
        print("Avg:", avg_avg)
        print("Maj:", avg_maj)
        print("Length:", avg_length)

        results[dataset_type] = {
            "average_pass": avg_pass,
            "average_avg": avg_avg,
            "average_maj": avg_maj,
            "average_length": avg_length,
            "scores": all_last_scores
        }

        save_path = f"results_entropy_2/{saved_model_name}/{dataset_type}.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
            


if __name__ == "__main__":

    set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--scale', type=str, default='1.5B')
    parser.add_argument('--tok_limit', type=int, default=32768)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--gpus', type=int, default=1)
    
    

    args = parser.parse_args()
    main(args)
