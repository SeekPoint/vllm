"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.parallel_utils.tensor_parallel import (
    gather_from_tensor_model_parallel_region)
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceOutputs

_SAMPLING_EPS = 1e-5

'''
0x8. Sampler & Copy-On-Write 机制
一般来说模型执行之后的输出Tensor维度是[batch_size, hidden_dim]，也就是下图中的hidden_states。
然后hidden_states会被传入到到sampler进行处理，
在sampler这里包含一个 lm_head proj 将hidden_states的维度映射成 [batch_size, vocab_size]。
        002.webp

Sampler的具体过程有些复杂，我们还是要仔细分析一下代码，
对应vllm/vllm/model_executor/layers/sampler.py这个文件。
'''

# Sampler是一个nn.Module子类，负责从模型的输出中采样下一个token。
# 该类的功能基于预定义的一系列步骤来调整和处理模型的输出。
class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence and frequency penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Dict[int, SequenceOutputs]:
        # Get the hidden states that we use for sampling.
        # 按输入元数据剪枝隐藏状态，只保留用于采样的状态。
        hidden_states = _prune_hidden_states(hidden_states, input_metadata)

        # Get the logits for the next tokens.
        # 这是通过将hidden_states与embedding矩阵的转置进行矩阵乘法得到的。
        # 如果存在embedding_bias，则将其添加到logits上。
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        # 由于可能使用了模型并行，这里调用gather通信原语同步模型参数
        logits = gather_from_tensor_model_parallel_region(logits)
        # Remove paddings in vocab (if any).
        # 删除词汇表中的任何padding
        logits = logits[:, :self.vocab_size]

        # Apply presence and frequency penalties.
        # 应用存在和频率惩罚：这是通过一个函数_apply_penalties来完成的，它使用input_metadata中的信息。
        output_tokens = _get_output_tokens(input_metadata)
        assert len(output_tokens) == logits.shape[0]
        presence_penalties, frequency_penalties = _get_penalties(
            input_metadata)
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        logits = _apply_penalties(logits, output_tokens, presence_penalties,
                                  frequency_penalties, self.vocab_size)

        # Apply temperature scaling.
        # 应用温度缩放：temperatures是根据input_metadata获得的。
        # 如果temperatures中的任何值不为1.0，则将逻辑值除以对应的温度。
        temperatures = _get_temperatures(input_metadata)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(temperatures,
                             dtype=logits.dtype,
                             device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        # 计算概率：通过对逻辑值应用softmax函数来计算。
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities (before applying top-p and top-k).
        # 计算对数概率：通过取概率的对数来计算。
        logprobs = torch.log(probs)

        # Apply top-p and top-k truncation.
        # 应用top-p和top-k截断：这是通过函数_apply_top_p_top_k完成的，它使用input_metadata中的信息。
        top_ps, top_ks = _get_top_p_top_k(input_metadata, self.vocab_size)
        assert len(top_ps) == len(top_ks) == probs.shape[0]
        do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
        do_top_k = any(k != self.vocab_size for k in top_ks)
        if do_top_p or do_top_k:
            probs = _apply_top_p_top_k(probs, top_ps, top_ks)

        # Sample the next tokens.
        # 采样下一个token：这是通过函数_sample完成的。
        return _sample(probs, logprobs, input_metadata)


# 这个函数_prune_hidden_states的目的是从给定的hidden_states中选择特定的隐藏状态，
# 这些状态对应于输入序列中每个prompt的最后一个token以及所有生成的token。
def _prune_hidden_states(
    hidden_states: torch.Tensor,
    input_metadata: InputMetadata,
) -> torch.Tensor:
    # 初始化一个start_idx为0，这个变量将用于迭代和获取每个prompt的最后一个token的索引。
    start_idx = 0
    # 初始化一个空列表last_token_indicies，它将用于存储每个prompt的最后一个token的索引。
    last_token_indicies: List[int] = []
    # 通过循环迭代input_metadata.prompt_lens中的每个prompt长度来更新last_token_indicies
    for prompt_len in input_metadata.prompt_lens:
        last_token_indicies.append(start_idx + prompt_len - 1)
        start_idx += prompt_len
    # 在处理完所有prompt后，添加所有生成的token的索引到last_token_indicies列表。
    last_token_indicies.extend(
        range(start_idx, start_idx + input_metadata.num_generation_tokens))
    return hidden_states[last_token_indicies]


# 这个函数_get_penalties的目的是获取输入元数据中的存在性惩罚和频率惩罚，并将它们组织为两个列表。
def _get_penalties(
        input_metadata: InputMetadata) -> Tuple[List[float], List[float]]:
    # Collect the presence and frequency penalties.
    # 初始化两个空列表presence_penalties和frequency_penalties。
    # 这些列表将分别用于存储存在性惩罚和频率惩罚的值。
    presence_penalties: List[float] = []
    frequency_penalties: List[float] = []
    # 循环迭代input_metadata.seq_groups中的每个序列组：
    for i, seq_group in enumerate(input_metadata.seq_groups):
        # 从seq_group中提取seq_ids（序列标识符）和sampling_params（采样参数）。
        seq_ids, sampling_params = seq_group
        # 从sampling_params中提取presence_penalty（存在性惩罚）
        # 到变量p和frequency_penalty（频率惩罚）到变量f。
        p = sampling_params.presence_penalty
        f = sampling_params.frequency_penalty
        # 检查当前迭代的索引i是否小于input_metadata.num_prompts，以确定当前的序列组是否为prompt输入。
        if i < input_metadata.num_prompts:
            # A prompt input.
            presence_penalties.append(p)
            frequency_penalties.append(f)
        else:
            # A generation token.
            presence_penalties += [p] * len(seq_ids)
            frequency_penalties += [f] * len(seq_ids)
    return presence_penalties, frequency_penalties


# 这个函数的目的是从input_metadata中提取输出token，并将它们组织为一个列表。
def _get_output_tokens(input_metadata: InputMetadata) -> List[List[int]]:
    # 初始化一个空列表output_tokens，用于存储输出token的列表。
    output_tokens: List[List[int]] = []
    # 循环迭代input_metadata.seq_groups中的每个序列组：
    for i, seq_group in enumerate(input_metadata.seq_groups):
        # 从seq_group中提取seq_ids（序列标识符）。_表示我们不使用sampling_params。
        seq_ids, _ = seq_group
        # 检查当前迭代的索引i是否小于input_metadata.num_prompts，以确定当前的序列组是否为prompt输入。
        if i < input_metadata.num_prompts:
            # A prompt input.
            # NOTE: While the prompt input usually has no output tokens,
            # it may have output tokens in the case of recomputation.
            # 这里的recompute指的是什么？
            seq_id = seq_ids[0]
            seq_data = input_metadata.seq_data[seq_id]
            output_tokens.append(seq_data.output_token_ids)
        else:
            # A generation token.
            for seq_id in seq_ids:
                seq_data = input_metadata.seq_data[seq_id]
                output_tokens.append(seq_data.output_token_ids)
    return output_tokens


# 这个函数_apply_penalties是用于在logits上应用特定的惩罚，具体来说是频率惩罚
#（frequency penalties）和存在惩罚（presence penalties）。
# 这些惩罚通常用于对模型生成的输出进行微调，从而控制模型的行为。
def _apply_penalties(
    logits: torch.Tensor, # 一个张量，表示模型输出的原始得分。
    output_tokens: List[List[int]], # 一个整数列表的列表，表示每个序列的输出token IDs。
    presence_penalties: List[float], # 浮点数列表，表示存在和频率惩罚。
    frequency_penalties: List[float],
    vocab_size: int,
) -> torch.Tensor:
    num_seqs = logits.shape[0]
    # Collect the indices of sequences that have non-zero penalties.
    # 首先，函数检查每个序列是否具有非零存在或频率惩罚，并收集这些序列的索引。
    indices = []
    for i in range(num_seqs):
        if not output_tokens[i]:
            continue
        p = presence_penalties[i]
        f = frequency_penalties[i]
        if p < _SAMPLING_EPS and f < _SAMPLING_EPS:
            continue
        indices.append(i)

    # Return early if all sequences have zero penalties.
    # 如果所有序列的惩罚都为零，则直接返回原始logits，无需应用任何惩罚
    if not indices:
        return logits

    # 对于每个具有非零惩罚的序列，使用np.bincount计算输出token的频率计数，并将它们堆叠到一个张量中。
    bin_counts = []
    for i in indices:
        bin_counts.append(np.bincount(output_tokens[i], minlength=vocab_size))
    bin_counts = np.stack(bin_counts, axis=0)
    bin_counts = torch.from_numpy(bin_counts).to(dtype=logits.dtype,
                                                 device=logits.device)

    # 从非零惩罚的序列索引中选择对应的频率和存在惩罚，并将它们转换为张量。
    frequency_penalties = [frequency_penalties[i] for i in indices]
    frequency_penalties = torch.tensor(frequency_penalties,
                                       dtype=logits.dtype,
                                       device=logits.device)
    presence_penalties = [presence_penalties[i] for i in indices]
    presence_penalties = torch.tensor(presence_penalties,
                                      dtype=logits.dtype,
                                      device=logits.device)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    # 对于每个具有非零惩罚的序列，频率惩罚乘以bin计数，并从对应的logits中减去结果。
    logits[indices] -= frequency_penalties.unsqueeze(dim=1) * bin_counts
    # 创建一个掩码，该掩码表示哪些token在序列中出现（即bin计数大于0）。
    # 然后，存在惩罚乘以此掩码，并从对应的logits中减去结果。
    presence_mask = (bin_counts > 0.0).to(dtype=logits.dtype)
    logits[indices] -= presence_penalties.unsqueeze(dim=1) * presence_mask
    return logits

# 这个函数_get_temperatures的目的是从输入元数据中获取每个序列的温度值。
# 温度是在随机采样过程中使用的一个参数，它控制了模型输出的多样性。
# 较高的温度会使输出更加随机，而较低的温度会使模型更加确定性地选择最有可能的输出。
def _get_temperatures(input_metadata: InputMetadata) -> List[float]:
    # Collect the temperatures for the logits.
    temperatures: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        temperature = sampling_params.temperature
        if temperature < _SAMPLING_EPS:
            # NOTE: Zero temperature means deterministic sampling
            # (i.e., greedy sampling or beam search).
            # Set the temperature to 1 to avoid division by zero.
            temperature = 1.0

        if i < input_metadata.num_prompts:
            # A prompt input.
            temperatures.append(temperature)
        else:
            # A generation token.
            temperatures += [temperature] * len(seq_ids)
    return temperatures


# 这个函数 _get_top_p_top_k 的目的是从输入元数据中提取每个序列的 top_p 和 top_k 值。
# 这两个参数都是用于控制模型在随机采样时如何裁剪概率分布的。
def _get_top_p_top_k(
    input_metadata: InputMetadata,
    vocab_size: int,
) -> Tuple[List[float], List[int]]:
    top_ps: List[float] = []
    top_ks: List[int] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        top_p = sampling_params.top_p
        # k should not be greater than the vocab size.
        top_k = min(sampling_params.top_k, vocab_size)
        # k=-1 means no truncation.
        top_k = vocab_size if top_k == -1 else top_k
        if i < input_metadata.num_prompts:
            # A prompt input.
            top_ps.append(top_p)
            top_ks.append(top_k)
        else:
            # A generation token.
            top_ps += [top_p] * len(seq_ids)
            top_ks += [top_k] * len(seq_ids)
    return top_ps, top_ks

# 这个函数的 _apply_top_p_top_k 主要目的是在随机采样过程中裁剪模型的输出概率分布。
# 其基本思路是根据 top_p 和 top_k 的值，保留概率最高的一些token，而不是根据其完整
# 的概率分布来随机选择token。这是一个控制模型输出多样性的常见方法。
def _apply_top_p_top_k(
    probs: torch.Tensor,
    top_ps: List[float],
    top_ks: List[int],
) -> torch.Tensor:
    p = torch.tensor(top_ps, dtype=probs.dtype, device=probs.device)
    k = torch.tensor(top_ks, dtype=torch.int, device=probs.device)
    # 对 probs Tensor按降序排序，得到排序后的概率 probs_sort 和相应的索引 probs_idx。
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

    # Apply top-p.
    # 计算累积和 probs_sum。
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 创建一个遮罩 top_p_mask，表示哪些token的概率应该被设置为0，以便保证累积概率低于给定的 top_p 值。
    top_p_mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
    # 使用此遮罩将不满足条件的token的概率设置为0。
    probs_sort[top_p_mask] = 0.0

    # Apply top-k.
    # Create a mask for the top-k elements.
    # 创建一个遮罩 top_k_mask，表示哪些token的概率应该被设置为0，以便只保留 top_k 最高的概率的token。
    top_k_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
    top_k_mask = top_k_mask.expand(probs_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze(dim=1)
    # 使用此遮罩将不满足条件的token的概率设置为0。
    probs_sort[top_k_mask] = 0.0

    # Re-sort the probabilities.
    # 使用原始索引 probs_idx 将裁剪后的概率 probs_sort 还原到其原始顺序，得到最终的裁剪后的概率 probs。
    probs = torch.gather(probs_sort,
                         dim=-1,
                         index=torch.argsort(probs_idx, dim=-1))
    return probs

# 这个函数 _get_topk_logprobs 的主要目的是从输入的 logprobs Tensor中
# 提取最大的 num_logprobs 个对数概率值，并返回一个字典，其中键是token的ID，值是对应的对数概率值。
def _get_topk_logprobs(
    logprobs: torch.Tensor,
    num_logprobs: Optional[int],
) -> Dict[int, float]:
    if num_logprobs is None or num_logprobs == 0:
        return {}

    # 使用 torch.topk 函数从 logprobs 中提取最大的 num_logprobs 个
    # 对数概率值和对应的token的ID。这个函数返回两个值：最大的对数概率值和对应的token的ID。
    topk_logprobs, topk_ids = torch.topk(logprobs, num_logprobs)
    # 如果 num_logprobs 是1，那么 topk_logprobs 和 topk_ids 都是标量。
    # 在这种情况下，我们需要调用 .item() 来从这些Tensor中获取Python的数值。
   # 否则，我们调用 .tolist() 将这两个Tensor转换为Python的列表。
    if num_logprobs == 1:
        topk_logprobs = [topk_logprobs.item()]
        topk_ids = [topk_ids.item()]
    else:
        topk_logprobs = topk_logprobs.tolist()
        topk_ids = topk_ids.tolist()

    # 初始化一个空字典 token_to_logprob。
    token_to_logprob: Dict[int, float] = {}
    # 使用 zip 函数遍历 topk_ids 和 topk_logprobs，并将每个token的ID和对应的对数概率值添加到字典中。
    for token_id, logprob in zip(topk_ids, topk_logprobs):
        token_to_logprob[token_id] = logprob
    return token_to_logprob


# 这个函数 _sample_from_prompt 的目的是根据给定的概率分布 prob 和
# 采样参数 sampling_params 从一个prompt中进行采样，并返回采样得到的token的ID列表。
def _sample_from_prompt(
    prob: torch.Tensor,
    sampling_params: SamplingParams,
) -> List[int]:
    if sampling_params.use_beam_search:
        # Beam search.
        # beam_width 设置为 sampling_params.best_of 的值。
        # 束搜索通常会选择多个可能的路径，并在每一步保持这些路径的数量。
        beam_width = sampling_params.best_of
        # 使用 torch.topk 函数从 prob 中选择 beam_width 个具有最大概率的tokens。
        _, next_token_ids = torch.topk(prob, beam_width)
        # 将得到的tokens转化为Python列表。
        next_token_ids = next_token_ids.tolist()
    elif sampling_params.temperature < _SAMPLING_EPS:
        # Greedy sampling.
        # 确认 sampling_params.best_of 的值为1。因为贪心采样只选择一个具有最高概率的token。
        assert sampling_params.best_of == 1
        # 使用 torch.argmax 函数从 prob 中选择具有最大概率的token。
        next_token_id = torch.argmax(prob)
        next_token_ids = [next_token_id.item()]
    else:
        # Random sampling.
        # Sample `best_of` tokens for the prompt.
        # 设置 num_seqs 为 sampling_params.best_of 的值。这表示我们要采样多少个tokens。
        num_seqs = sampling_params.best_of
        # 使用 torch.multinomial 函数从 prob 中随机采样 num_seqs 个tokens。
        # replacement=True 表示可以重复采样同一个token。
        next_token_ids = torch.multinomial(prob,
                                           num_samples=num_seqs,
                                           replacement=True)
        next_token_ids = next_token_ids.tolist()
    # 返回 next_token_ids 列表，这是从提示中采样得到的token的ID列表。
    return next_token_ids

# 这个函数 _sample_from_generation_tokens 主要是根据给定的概率分布和参数来对生成的token进行采样。
def _sample_from_generation_tokens(
    seq_ids: List[int],
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    seq_logprobs: List[float],
    sampling_params: SamplingParams,
) -> Tuple[List[int], List[int]]:
    # NOTE(woosuk): sampling_params.best_of can be greater than
    # len(seq_ids) because some sequences in the group might have
    # been already terminated.
    if sampling_params.use_beam_search:
        # Beam search.
        # Add cumulative logprobs for the sequences in the group.
        # 首先，将 seq_logprobs 添加到每个token的 logprobs 中，
        # 因为束搜索的计算需要考虑整个序列的累计对数概率。
        seq_logprobs = torch.tensor(seq_logprobs,
                                    dtype=torch.float,
                                    device=logprobs.device)
        logprobs = logprobs + seq_logprobs.unsqueeze(dim=1)

        # 确定词汇表大小和束宽。
        vocab_size = logprobs.size(-1)
        beam_width = len(seq_ids)
        # 从扁平化的 logprobs 中获取最高的 beam_width 个对数概率及其对应的索引。
        _, topk_ids = torch.topk(logprobs.flatten(), beam_width)
        # 根据获取的最高的索引，计算出对应的序列索引 seq_idx 和token索引 token_ids。
        topk_ids = topk_ids.tolist()
        seq_idx = [i // vocab_size for i in topk_ids]
        beam_seq_ids = [seq_ids[i] for i in seq_idx]
        token_ids = [i % vocab_size for i in topk_ids]

        # 初始化一个空的字典 beam_outputs 来存储当前步的最佳序列和token。
        beam_outputs: Dict[int, Tuple[int, int]] = {}
        outstanding_beams: List[Tuple[int, int]] = []
        # If a beam survives, continue with it.
        # 遍历序列和token，将它们添加到 beam_outputs 中。
        # 如果某个序列已经存在于 beam_outputs 中，则将它添加到 outstanding_beams 列表中。
        for seq_id, token_id in zip(beam_seq_ids, token_ids):
            if seq_id not in beam_outputs:
                beam_outputs[seq_id] = (seq_id, token_id)
            else:
                outstanding_beams.append((seq_id, token_id))

        # If a beam is discarded, fork another beam.
        # 如果某个序列没有在 beam_outputs 中，
        # 则从 outstanding_beams 中弹出一个并添加到 beam_outputs 中。
        for seq_id in seq_ids:
            if seq_id not in beam_outputs:
                beam_outputs[seq_id] = outstanding_beams.pop()
        assert not outstanding_beams

        # 最后，从 beam_outputs 中提取父序列ID和下一个token ID。
        parent_seq_ids = [beam_outputs[seq_id][0] for seq_id in seq_ids]
        next_token_ids = [beam_outputs[seq_id][1] for seq_id in seq_ids]
    elif sampling_params.temperature < _SAMPLING_EPS:
        # Greedy sampling.
        assert len(seq_ids) == 1
        next_token_id = torch.argmax(probs, dim=-1)
        next_token_ids = [int(next_token_id.item())]
        parent_seq_ids = seq_ids
    else:
        # Random sampling.
        # Sample 1 token for each sequence in the group.
        next_token_ids = torch.multinomial(probs,
                                           num_samples=1,
                                           replacement=True)
        next_token_ids = next_token_ids.squeeze(dim=-1).tolist()
        parent_seq_ids = seq_ids
    return parent_seq_ids, next_token_ids

# 这个函数 _sample 的目的是基于给定的概率分布和输入元数据来对token进行采样，并为每个序列返回一个采样结果。
def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    input_metadata: InputMetadata,
) -> Dict[int, SequenceOutputs]:
    seq_outputs: Dict[int, SequenceOutputs] = {}

    # TODO(woosuk): Optimize.
    idx = 0
    # 对于input_metadata.seq_groups中的每个seq_group:
    for i, seq_group in enumerate(input_metadata.seq_groups):
        # 提取seq_ids（序列ID）和sampling_params（采样参数）。
        seq_ids, sampling_params = seq_group
        # 处理prompt
        if i < input_metadata.num_prompts:
            # Generate the next tokens for a prompt input.
            assert len(seq_ids) == sampling_params.best_of
            prob = probs[idx]
            logprob = logprobs[idx]
            idx += 1

            # Sample the next tokens.
            next_token_ids = _sample_from_prompt(prob, sampling_params)
            # Get top-k log probabilities for the next tokens.
            next_logprobs = _get_topk_logprobs(logprob,
                                               sampling_params.logprobs)

            # Build the output.
            for seq_id, next_token_id in zip(seq_ids, next_token_ids):
                output_logprobs = next_logprobs.copy()
                output_logprobs[next_token_id] = logprob[next_token_id].item()
                seq_outputs[seq_id] = SequenceOutputs(seq_id, seq_id,
                                                      next_token_id,
                                                      output_logprobs)
        # 处理生成的token
        else:
            # Generate the next tokens for generation tokens.
            prob = probs[idx:idx + len(seq_ids)]
            logprob = logprobs[idx:idx + len(seq_ids)]
            idx += len(seq_ids)

            # Sample the next tokens.
            seq_logprobs = [
                input_metadata.seq_data[seq_id].cumulative_logprob
                for seq_id in seq_ids
            ]
            parent_seq_ids, next_token_ids = _sample_from_generation_tokens(
                seq_ids, prob, logprob, seq_logprobs, sampling_params)

            # Get top-k log probabilities for the next tokens.
            next_logprobs: Dict[int, Dict[int, float]] = {}
            for j, seq_id in enumerate(seq_ids):
                next_logprobs[seq_id] = _get_topk_logprobs(
                    logprob[j], sampling_params.logprobs)

            # Build the output.
            for seq_id, parent_seq_id, next_token_id in zip(
                    seq_ids, parent_seq_ids, next_token_ids):
                j = seq_ids.index(parent_seq_id)
                output_logprobs = next_logprobs[parent_seq_id].copy()
                output_logprobs[next_token_id] = logprob[j,
                                                         next_token_id].item()
                seq_outputs[seq_id] = SequenceOutputs(
                    seq_id,
                    parent_seq_id,
                    next_token_id,
                    output_logprobs,
                )

    return seq_outputs

'''
在_sample这个函数中，分别对prompt和generation进行了处理。
我们可以从_sample_from_prompt和_sample_from_generation_tokens这两个函数中发现，
只有在beam search方法时对prompt和generation的处理方式不一样。

如果我们输入有n个prompt，也就是n个seq_group，然后每个seq_group有 best_of 个 seqs。
我们在prompt阶段要做的就是对于每个 seq_group，我们从对应分布中取 top best_of 的结果，然后把它们和每个 seq_id 对应。
而对于后续的 generation 阶段，注意现在每一次下一步都有 best_of * vocab_size 的空间，
然后会在这个空间中取topk（_, topk_ids = torch.topk(logprobs.flatten(), beam_width)）。
然后我们知道这些 topk 可能有多个来自同一个 parent seq。
然后它的策略是，一个 parent id 对应的第一个 output 会放到对应的位置，其它的 output 会被放到某个驱逐的 parent id 对应的位置。
这里也就是了VLLM的fork机制，Fork 的过程在 Block Manager 里实现就是，child seq 会拷贝 parent seq 的 block table，
同时给对应的 block 加上 ref counter。
这个fork不是在sampler阶段做的，而是在scheduler update 的位置 fork。

在scheduler的update函数里面，涉及到针对beam search采样的fork调用：
      003.webp
以及Block Manager的fork函数：
    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
    
可以看到fork只是增加parent seq的物理块引用计数，而在给新的 token allocate 物理块的时候，
如果发现之前的 block 是引用，则会在此时 allocate 真正的 block，
并且拷贝数据（拷贝是先在 scheduler 的_schedule函数里生成 blocks_to_copy 这个信息包，
传到 engine 部分进一步传给worker由对应的 cuda kernel真正执行）。
这个过程也就是VLLM的copy on write，对应了以下的几张图：

            004-1.webp
            ===
            004-7.webp



0x10. 总结
这篇文章主要是对VLLM的流程进行了梳理，还没有解析Paged Attention的实现，等下次有空再研究一下。



    
'''