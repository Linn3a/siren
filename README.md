# Selective Entropy Regularization (SIREN) 🧜‍♀️

<div align="center">
    <p>
    </p>
    <a href="https://arxiv.org/abs/2509.25133"><img src="https://img.shields.io/badge/arXiv-2509.25133-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv:2509.25133"></a>
</div>

This repository contains the official implementation of **Selective Entropy Regularization (SIREN)**, introduced in our paper:
[Rethinking Entropy Regularization in Large Reasoning Models](https://arxiv.org/abs/2509.25133).

SIREN addresses the issue of *entropy collapse* in Reinforcement Learning with Verifiable Reasoning (RLVR) when applying naive entropy regularization to large reasoning models.
Built upon the [veRL](https://github.com/volcengine/verl) framework, our implementation introduces key modifications to entropy computation, aggregation, and the overall training objective.

## Installation ⚙️

We recommend creating a clean conda environment to avoid dependency conflicts.

```
conda create -n siren python=3.10
conda activate siren
pip install -r requirements.txt

# install verl
cd verl
pip install -e .
```


## Usage 🍽️

### Prepare data

```bash
huggingface-cli download --repo-type dataset --resume-download Elliott/Openr1-Math-46k-8192 --local-dir data
```

### Running

We provide example scripts for both training and evaluation.

```bash
# training
bash exp_scripts/siren.sh

# evaluation
bash exp_scripts/eval.sh

```

- The training script (siren.sh) contains default hyperparameters and can be customized according to your experimental setup. 

# Acknowledgement 🫰

We thank the open-source communities behind the following projects for their valuable contributions:
- Frameworks: [veRL](https://github.com/volcengine/verl), [vLLM](https://github.com/vllm-project/vllm) , [Math-Verify](https://github.com/huggingface/Math-Verify)
- Datasets: [MATH](https://github.com/hendrycks/math), [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- Backbones: [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math), [Llama-3.1](https://huggingface.co/meta-llama/Llama-3.1-8B)

## Citation 📜

If you find our work useful in your research, please consider citing:

```bibtex
@misc{jiang2025rethinkingentropyregularizationlarge,
      title={Rethinking Entropy Regularization in Large Reasoning Models}, 
      author={Yuxian Jiang and Yafu Li and Guanxu Chen and Dongrui Liu and Yu Cheng and Jing Shao},
      year={2025},
      eprint={2509.25133},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.25133}, 
}
```
