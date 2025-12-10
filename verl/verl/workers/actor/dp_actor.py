# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from flash_attn.bert_padding import (index_first_axis, pad_input, rearrange,
                                     unpad_input)
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (agg_loss, compute_policy_loss,
                                         compute_policy_loss_clip_cov,
                                         compute_policy_loss_kl_cov,
                                         compute_policy_loss_noclip,
                                         kl_penalty)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import (get_reverse_idx,
                                         rearrange_micro_batches)
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (gather_outpus_and_unpad,
                                ulysses_pad_and_slice_inputs)
from verl.workers.actor import BasePPOActor

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def efficient_top_p_sampling(logits, k=15000, p=0.95):
    probs = torch.softmax(logits, dim=-1)  # shape: [vocab_size]

    topk_logits, topk_indices = torch.topk(logits, k=k)

    topk_probs = torch.gather(probs, dim=1, index=topk_indices)
    cumulative_probs = torch.cumsum(topk_probs, dim=-1)

    # compute topk entropy
    with torch.no_grad():
        full_entropy = torch.logsumexp(logits, dim=-1) - torch.sum(torch.nn.functional.softmax(logits, dim=-1) * logits, dim=-1)

    tensor_p = p * torch.ones([probs.shape[0], k]).to(logits.device)

    cutoff_idx = torch.unsqueeze(torch.max(torch.searchsorted(cumulative_probs,tensor_p), dim = -1).values, dim=1)

    max_idx = cumulative_probs.shape[-1]

    B, K = topk_indices.shape

    range_tensor = torch.arange(K).unsqueeze(0).expand(B, K).to(logits.device)  # [B, K]

    cutoff_idx_expanded = cutoff_idx.expand_as(topk_indices)  # [B, K]
    mask = range_tensor < cutoff_idx_expanded  # [B, K]

    filtered_indices = [row[mask_row] for row, mask_row in zip(topk_indices, mask)]
    mask = torch.zeros_like(logits, dtype=torch.bool)
    
    for i in range(B):
        mask[i].scatter_(0, filtered_indices[i], True)

    pd_masked = probs * mask.float()   

    pd_masked = pd_masked / (pd_masked.sum(dim=-1, keepdim=True) + 1e-12)
    entropy = -torch.sum(pd_masked * torch.log(pd_masked + 1e-12), dim=-1)

    return entropy, full_entropy


def mask_by_quantile(x: torch.Tensor, response_mask: torch.Tensor, upper: float, lower: float):

    x_masked = x.masked_fill(~response_mask.to(dtype=torch.bool),float('nan')).float()

    lower_bound = torch.nanquantile(x_masked, lower, dim=-1, keepdim=True)

    quantile_mask = (x > lower_bound)
    masked_x = torch.where(quantile_mask, x, torch.zeros_like(x))
    return masked_x, quantile_mask, lower_bound

class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1


        self.experiment_type = self.config.get("experiment_type", "baseline")
        self.entropy_agg_mode = self.config.get("entropy_agg_mode", "seq-mean-token-mean")
        
        if self.experiment_type.startswith("naive"):
            self.entropy_agg_mode = self.config.loss_agg_mode
            
        self.mask_ratio = self.config.get("mask_ratio", 0.8)
        
        self.use_siren = self.config.get("use_siren", True)
        self.no_entropy_loss = self.config.get("no_entropy_loss", False)

        self.entropy_topk = self.config.get("entropy_topk", 10000)
        self.entropy_topp = self.config.get("entropy_topp", 0.8)
        
        self.entropy_coeff_lr = self.config.get("entropy_coeff_lr", 0)
        self.entropy_coeff = self.config.entropy_coeff
        

        print(f"entropy_topk: {self.entropy_topk}, entropy_topp: {self.entropy_topp}")

        if "entropy_anchor" in self.config:
            self.entropy_anchor = torch.tensor(self.config.entropy_anchor)
        
        if "mask_ratio_upper" in self.config:
            self.mask_ratio_upper = self.config.mask_ratio_upper

            print(f"clip entropy from {self.mask_ratio} to {self.mask_ratio_upper}")

        if not self.use_siren and (self.experiment_type == "baseline_entropy_adv" or self.experiment_type == "baseline_eighty_twenty" or self.experiment_type == "naive_entropy_reg" or self.experiment_type == "naive_entropy_reg_adaptive" or self.experiment_type == "abla_wo_topp" or self.experiment_type == "baseline" or self.experiment_type == "abla_only_anchor"):
            print("use naive entropy calculation")
            self.compute_entropy_from_logits = (
                torch.compile(verl_F.entropy_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)  #  use torch compile by default
                else verl_F.entropy_from_logits
            )
        else:
            print("use **topp** entropy calculation")
            self.compute_entropy_from_logits_w_topp = efficient_top_p_sampling
      
    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)

                # compute entropy
                if calculate_entropy:
                    full_entropy_rmpad = None
                    # entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                    try:
                        entropy_rmpad, full_entropy_rmpad = self.compute_entropy_from_logits_w_topp(logits_rmpad, self.entropy_topk, self.entropy_topp)
                    except:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        if full_entropy_rmpad is not None:
                            full_entropy_rmpad = gather_outpus_and_unpad(full_entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        # top_p_entropy_rmpad = gather_outpus_and_unpad(top_p_entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    if full_entropy_rmpad is not None:
                        full_entropy_full = pad_input(hidden_states=full_entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                    if full_entropy_rmpad is not None:
                        full_entropy_full = full_entropy_full.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            if full_entropy_rmpad is not None:
                return entropy, log_probs, full_entropy_full
            return entropy, log_probs, None

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def clean_entropy_anchor(self):
        if hasattr(self, 'entropy_anchor'):
            del self.entropy_anchor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        full_entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, full_entropy = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
                if full_entropy is not None:
                    full_entropy_lst.append(full_entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        full_entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
            if full_entropy is not None:
                full_entropys = torch.concat(full_entropy_lst, dim=0)

        
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys, full_entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error

        
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "token_level_scores"]
            
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if 'entropy_anchor' in data.non_tensor_batch:
            self.entropy_anchor = torch.tensor(data.non_tensor_batch['entropy_anchor'][0])


        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                
                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]
                    
                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    loss_agg_mode = self.config.loss_agg_mode

                    
                    calculate_entropy = True
                    entropy, log_prob, full_entropy = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    if self.use_siren:
                        masked_entropy, value_mask, lower_bound = mask_by_quantile(entropy, response_mask, 1, self.mask_ratio)
                        entropy = masked_entropy
                        
                    if self.no_entropy_loss:
                        
                        if self.experiment_type == "baseline_entropy_adv":
                            if "adv_alpha" in self.config:
                                alpha = self.config.adv_alpha
                            else:
                                alpha = 0.4
                            print(f"baseline entropy advantage with alpha={alpha}")
                            kappa = 1
                            advantages += torch.min(alpha * entropy.detach(), advantages.abs()/kappa)

                        if self.experiment_type == "baseline_kl_cov":
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_kl_cov(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                        elif self.experiment_type == "baseline_clip_cov":
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_clip_cov(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                        
                        elif self.experiment_type == "baseline_eighty_twenty":
                            masked_entropy, value_mask, lower_bound = mask_by_quantile(entropy, response_mask, 1, 0.8)
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=value_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                        else:
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )

                        entropy_loss = torch.tensor(0.0)
                        entropy_loss_mse = torch.tensor(0.0)
                    
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                        if self.experiment_type == "naive_entropy_reg" or self.experiment_type == "naive_entropy_reg_adaptive":
                            value_mask = response_mask
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=value_mask, loss_agg_mode=self.entropy_agg_mode)
                            entropy_loss_mse = -1 * entropy_loss
                            
                            if not hasattr(self, 'entropy_anchor'): 
                                self.entropy_anchor = entropy_loss.detach()

                        elif self.experiment_type == "abla_wo_anchor":
                            masked_entropy, value_mask, lower_bound = mask_by_quantile(entropy, response_mask, 1, self.mask_ratio)
                            entropy_loss = agg_loss(loss_mat=masked_entropy, loss_mask=value_mask, loss_agg_mode=self.entropy_agg_mode)
                            entropy_loss_mse = -1 * entropy_loss
                
                        # use anchor
                        elif self.experiment_type == "siren" or self.experiment_type == "abla_wo_topp" or self.experiment_type == "abla_token_first" or self.experiment_type == "abla_wo_token" or self.experiment_type == "abla_only_anchor":
                            
                            if self.experiment_type == "siren" or self.experiment_type == "abla_wo_topp":
                                if hasattr(self, "mask_ratio_upper"):
                                    masked_entropy, value_mask, lower_bound = mask_by_quantile(entropy, response_mask, self.mask_ratio_upper, self.mask_ratio)
                                else:
                                    masked_entropy, value_mask, lower_bound = mask_by_quantile(entropy, response_mask, 1, self.mask_ratio)
                                
                            elif self.experiment_type == "abla_token_first":
                                masked_entropy, value_mask, lower_bound = mask_by_quantile(full_entropy, response_mask, 1, self.mask_ratio)
                                                        
                        
                            elif self.experiment_type == "abla_wo_token" or self.experiment_type == "abla_only_anchor":
                                value_mask = response_mask
                            
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=value_mask, loss_agg_mode=self.entropy_agg_mode)
                        
                        
                            if hasattr(self, 'entropy_anchor'):
                                entropy_change = entropy_loss - self.entropy_anchor.to(entropy_loss.device)
                                entropy_loss_mse = entropy_change.pow(2).mean()  
                            # else:
                            #     self.entropy_anchor = entropy_loss.detach()
                            #     entropy_loss_mse = torch.tensor(0.0)
                            
                                
                        else:
                            raise ValueError(f"Invalid experiment_type {self.experiment_type}")
                        
                        
                    policy_loss = pg_loss + entropy_loss_mse * self.entropy_coeff
                    

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        if self.experiment_type == "siren_kl":
                            masked_entropy, value_mask, lower_bound = mask_by_quantile(full_entropy, response_mask, 1, self.mask_ratio)
                        else:
                            value_mask = response_mask
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=value_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                        
                    loss.backward()
                    
                    
                    if self.experiment_type == "naive_entropy_reg_adaptive":
                    # update entropy coeff
                        if hasattr(self, 'entropy_anchor'):
                            alpha_gradient = self.entropy_anchor - entropy_loss.detach()
                        else:
                            alpha_gradient = torch.tensor(0.0)
                        
                        self.entropy_coeff += self.entropy_coeff_lr * alpha_gradient.item()
                        if self.entropy_coeff < 0:
                            self.entropy_coeff = 0.0

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/mse_entropy_loss": entropy_loss_mse.detach().item(),
                        "actor/entropy_coeff": self.entropy_coeff,
                    }

                    if hasattr(self, 'entropy_anchor'):
                        data["actor/entropy_anchor"] = self.entropy_anchor.item()
                    append_to_dict(metrics, data)
                    

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
