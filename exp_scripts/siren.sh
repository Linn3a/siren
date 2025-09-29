export WANDB_MODE=offline

MODEL_PATH=path/to/model
DATA_PATH=data/openr1.parquet

TEST_DATA_PATH=data/validation.parquet
MODEL_TYPE=qwen

loss_agg_mode="seq-mean-token-sum-norm"

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 3))

enable_filter_groups=False
select_batch_metric=acc
max_num_gen_batches=20
train_prompt_bsz=128
train_prompt_mini_bsz=8
n_resp_per_prompt=8
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio=0.28

temperature=1.0
test_temperature=0.6
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))

entropy_coeff=0.0
entropy_coeff=0.005
kl_coeff=0.000
entropy_update_steps=800

experiment_type=siren

mask_ratio=0.8
entropy_topk=10000
entropy_topp=0.8

wandb_name=${experiment_type}_${entropy_coeff}_token${mask_ratio}_topk${entropy_topk}_topp${entropy_topp}

python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files=${DATA_PATH} \
data.val_files=${TEST_DATA_PATH} \
data.train_batch_size=${train_prompt_bsz} \
actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
data.max_prompt_length=$max_prompt_length \
data.max_response_length=$max_response_length \
data.filter_overlong_prompts=True \
data.truncation="left" \
data.return_raw_chat=True \
actor_rollout_ref.model.path=${MODEL_PATH} \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
actor_rollout_ref.actor.optim.weight_decay=0.1 \
actor_rollout_ref.actor.clip_ratio=${clip_ratio} \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=$train_prompt_mini_bsz \
actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
actor_rollout_ref.actor.use_kl_loss=False \
actor_rollout_ref.actor.kl_loss_coef=0.0 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
+actor_rollout_ref.actor.experiment_type=${experiment_type} \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
+actor_rollout_ref.rollout.multi_turn.enable=False \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
algorithm.use_kl_in_reward=False \
algorithm.norm_adv_by_std_in_grpo=False \
actor_rollout_ref.rollout.enable_chunked_prefill=True \
actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
actor_rollout_ref.rollout.temperature=${temperature} \
actor_rollout_ref.rollout.top_p=${top_p} \
actor_rollout_ref.rollout.top_k=${top_k} \
actor_rollout_ref.rollout.val_kwargs.temperature=${test_temperature} \
actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
actor_rollout_ref.rollout.val_kwargs.do_sample=True \
actor_rollout_ref.rollout.val_kwargs.n=16 \
actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
reward_model.reward_manager=naive \
custom_reward_function.path=verl/recipe/entropy/reward_score.py \
custom_reward_function.name=reward_func \
trainer.critic_warmup=0 \
trainer.logger="['console','wandb']" \
trainer.project_name='siren' \
trainer.experiment_name=${wandb_name} \
trainer.val_before_train=False \
+trainer.reward_mode=baseline \
trainer.n_gpus_per_node=1 \
trainer.nnodes=1 \
trainer.save_freq=10 \
trainer.test_freq=10 \
trainer.total_epochs=2 

