from math_verify import parse, verify
import numpy as np

def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    
    delta_score = 0
    try:
        acc = verify(parse(solution_str), parse(ground_truth))
    except Exception as e:
        acc = 0

    return {
        "score": acc,
        "delta_score": 0,
        "acc": acc,
        "length": extra_info['valid_response_length'],
    }
