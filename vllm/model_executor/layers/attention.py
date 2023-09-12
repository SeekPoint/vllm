"""Multi-head attention."""
from typing import List, Optional

import torch
import torch.nn as nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm import attention_ops
from vllm import cache_ops
from vllm import pos_encoding_ops
from vllm.model_executor.input_metadata import InputMetadata

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]


class PagedAttention(nn.Module):
    # pylint: disable=line-too-long
    """GPT-style multi-head PagedAttention.

    This class takes flattened 1D query, key, and value tensors as input. The
    input 1D tensors can either contain prompt tokens or generation tokens, in
    addition to paddings.

    If the input tensors contain prompt tokens, the layout is as follows:

    |<---------------------- num_valid_tokens ---------------------->|
    |<--------------- num_prompt_tokens -------------->|
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|<--padding-->|

    Otherwise, the layout is as follows:

    |<------------------ num_valid_tokens ------------------->|
    |<------- num_generation_tokens (M) ------->|
    |<--generation_0-->|...|<--generation_M-1-->|<--padding-->|

    The prompts might have different lengths, while the generation tokens always
    have length 1. The paddings are appended to make the input length a multiple
    of 8, which is desirable for Tensor Cores.

    The class does the following:
    1. Perform multi_query_kv_attention for the prompts. This operation does
        not use the KV cache.
    2. Wait for the cache operations (e.g., swap, copy) to finish. The cache
        operations are issued by the cache engine before executing the forward
        pass of the model, and they are executed asynchronously.
    3. Reshape and store the input key and value tensors in the KV cache.
    4. Perform single_query_cached_kv_attention for the generation tokens.
        This operation reads the previous key and value tensors from the KV
        cache.
    5. Output a flattened 1D tensor.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 num_kv_heads: Optional[int] = None) -> None:
        super().__init__()
        self.num_heads = num_heads # 多头注意力中的“头”的数量。
        self.head_size = head_size # 每个“头”的大小。
        self.scale = float(scale) # 缩放因子。
        # 使用了一个外部库xops.fmha.cutlass.FwOp()来实现注意力操作，但这里没有提供具体细节。
        self.attn_op = xops.fmha.cutlass.FwOp()
        # 键值对头的数量。默认为None，这意味着它的默认值与num_heads相同。
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        # 对num_heads和num_kv_heads之间的关系进行了断言，确保num_heads是num_kv_heads的倍数。
        assert self.num_heads % self.num_kv_heads == 0
        # 一个query共用多少个k, v
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        # 根据num_kv_heads和num_queries_per_kv来计算head_mapping。
        self.head_mapping = torch.repeat_interleave(
            torch.arange(self.num_kv_heads, dtype=torch.int32, device="cuda"),
            self.num_queries_per_kv)

        # 检查是否支持head_size，如果不支持则抛出错误。
        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    # 此成员函数set_attn_bias的目的是设置注意力偏置（attention bias）。
    # 这里的注意力偏置就是mask矩阵
    def set_attn_bias(self, input_metadata: InputMetadata) -> None:
        # 检查input_metadata对象中的attn_bias属性是否已经设置。
        # 如果attn_bias是真值（例如，它是一个非空列表或其他真值），则进入此条件语句。
        if input_metadata.attn_bias:
            # Already set by a previous layer.
            # 如果注意力偏置已经被设置，则提前退出函数。
            return
        # 从input_metadata对象中获取prompt_lens属性，该属性可能表示各提示的长度。
        prompt_lens = input_metadata.prompt_lens
        # 使用BlockDiagonalCausalMask.from_seqlens方法从prompt_lens（提示的长度列表）
        # 生成一个块对角因果掩码，并将其赋值给attn_bias。这个掩码可能被用于实现自回归模型中的
        # 因果注意力，确保一个位置的输出只依赖于该位置之前的输入。
        attn_bias = BlockDiagonalCausalMask.from_seqlens(prompt_lens)
        # 将新生成的attn_bias添加到input_metadata对象中的attn_bias属性。
        input_metadata.attn_bias.append(attn_bias)

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """对 prompt tokens 做正常的attention.
	Normal attention for the prompt tokens.

        Args:
            output: shape = [num_prompt_tokens, num_heads, head_size]
            query: shape = [num_prompt_tokens, num_heads, head_size]
            key: shape = [num_prompt_tokens, num_kv_heads, head_size]
            value: shape = [num_prompt_tokens, num_kv_heads, head_size]
            input_metadata: metadata for paged attention.  paged attention的元信息.
        """

        # 判断num_kv_heads（键值对头的数量）是否不等于num_heads（注意力的头的数量）。
        if self.num_kv_heads != self.num_heads:
            # Project the key and value tensors to the desired number of heads.
            # 如果上述条件为真，那么对key和value进行调整，使其头的数量与num_heads匹配。
            # 具体来说，它会使用torch.repeat_interleave函数沿着第1个维度重复
            # 每个key和value self.num_queries_per_kv次。
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=1)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=1)

        # TODO(woosuk): The unsqueeze op may incur some CPU overhead. Optimize.
        # 使用xtransformer库中的memory_efficient_attention_forward方法来计算注意力。
        # 输入的query、key、value张量都增加了一个额外的维度（使用unsqueeze(0)）。
        # 此外，它还接受了input_metadata.attn_bias[0]作为注意力偏置、
        # self.scale作为缩放因子以及self.attn_op作为操作符。
        out = xops.memory_efficient_attention_forward(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias=input_metadata.attn_bias[0],
            p=0.0,
            scale=self.scale,
            op=self.attn_op,
        )
        # TODO(woosuk): Unnecessary copy. Optimize.
        output.copy_(out.squeeze(0))
        return output

    # 这段代码定义了single_query_cached_kv_attention方法，它用于单个query时的KV缓存注意力操作。
    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        """PagedAttention for the generation tokens.

        Args:
            output: shape = [num_generation_tokens, num_heads, head_size]
            query: shape = [num_generation_tokens, num_heads, head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
        """
        block_size = value_cache.shape[3]
        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            self.head_mapping,
            self.scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            None,  # alibi_slopes
        )

    # 这是PagedAttention类的forward方法，表示该模块的前向传播操作。
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        NOTE: The query, key, and value tensors must be sliced from a qkv
        tensor of shape [num_tokens, 3 * num_heads * head_size].

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
            cache_event: event to wait for the cache operations to finish.

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        # Reshape the query, key, and value tensors.
        # 将输入的query, key, value张量重新调整为具有三个维度的形状。
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Pre-allocate the output tensor.
        # 创建一个与查询张量形状相同但未初始化的输出张量。
        output = torch.empty_like(query)

        # Compute the attention op for prompts.
        # 这段代码处理prompt tokens的注意力：
        num_prompt_tokens = input_metadata.num_prompt_tokens
        # 如果存在prompt token，则使用multi_query_kv_attention方法为prompt tokens计算注意力。
        if num_prompt_tokens > 0:
            # Prompt run.
            assert input_metadata.num_generation_tokens == 0
            self.set_attn_bias(input_metadata)
            self.multi_query_kv_attention(
                output[:num_prompt_tokens],
                query[:num_prompt_tokens],
                key[:num_prompt_tokens],
                value[:num_prompt_tokens],
                input_metadata,
            )

        # Wait until the cache op is done.
        # 等待缓存操作完成
        if cache_event is not None:
            cache_event.wait()

        # Reshape the keys and values and store them in the cache.
        # When key_cache and value_cache are not provided, the new key
        # and value vectors will not be cached.
        # 这里的input_metadata.num_valid_tokens对应了这个类开头的解释
        num_valid_tokens = input_metadata.num_valid_tokens
        if (num_valid_tokens > 0 and key_cache is not None
                and value_cache is not None):
            # The stride is 3 because the key and value are sliced from qkv.
            cache_ops.reshape_and_cache(
                key[:num_valid_tokens],
                value[:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata.slot_mapping,
            )

        if input_metadata.num_generation_tokens > 0:
            # Decoding run.
            assert input_metadata.num_prompt_tokens == 0
            assert key_cache is not None and value_cache is not None, (
                "key_cache and value_cache must be provided when "
                "generating tokens.")
            # Compute the attention op for generation tokens.
            self.single_query_cached_kv_attention(
                output[num_prompt_tokens:num_valid_tokens],
                query[num_prompt_tokens:num_valid_tokens], key_cache,
                value_cache, input_metadata)

        # Reshape the output tensor.
        # NOTE(woosuk): The output tensor may include paddings.
        return output.view(-1, self.num_heads * self.head_size)
'''
这里的key_cache有些特殊，暂时还不清楚为什么它的维度和value_cache不一样，
相比于value_cache的shape，它在head_size维度拆分了一个x出来，
感觉是一种加速策略后续在解析cuda kernel实现时希望可以找到答案。

然后可以看到对于prompt的attention的计算是由xtransformers库的算子来完成的，
在生成阶段则是由VLLM自己实现的cuda kernel（attention_ops.single_query_cached_kv_attention）来完成计算，
因为这里涉及到Paged Attention特有的cache访问策略。
'''

class PagedAttentionWithRoPE(PagedAttention):
    """PagedAttention with GPT-NeoX style rotary embedding."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,
        base: int = 10000,
        num_kv_heads: Optional[int] = None,
    ) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads)

        # Create the cos and sin cache.
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2) / rotary_dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum("i,j -> ij", t, inv_freq.float())
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        # FIXME(woosuk): This assumes that we configure the default dtype when
        # initializing the model.
        # TODO(woosuk): Make it more robust.
        torch_dtype = torch.get_default_dtype()
        cache = cache.to(torch_dtype)
        # Embedding size: [max_position, rotary_dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        """ PagedAttention forward pass with rotary embedding.

        Args:
            positions: shape = [num_tokens]
                        query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
            cache_event: event to wait for the cache operations to finish.

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        pos_encoding_ops.rotary_embedding_neox(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
        )
        return super().forward(
            query,
            key,
            value,
            key_cache,
            value_cache,
            input_metadata,
            cache_event,
        )


# 这段代码定义了一个名为 PagedAttentionWithALiBi 的类，
# 它扩展了另一个名为 PagedAttention 的类，增加了ALiBi注意力偏置（attention bias）的功能。
class PagedAttentionWithALiBi(PagedAttention):
    """PagedAttention with ALiBi attention bias."""

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 slopes: List[float],
                 num_kv_heads: Optional[int] = None) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads)
        # 确保 slopes 的长度与 num_heads 相等。
        assert len(slopes) == num_heads

        slopes = torch.tensor(slopes, dtype=torch.float32)
        # 在PyTorch模型中，如果你有一个你不希望被认为是模型参数的张量，你可以使用
        # register_buffer 将它注册为一个缓冲区。这里，它注册了一个名为 "alibi_slopes" 的缓冲区。
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    # 这个方法的目的是根据输入的元数据（input_metadata）设置注意力偏置。
    def set_attn_bias(self, input_metadata: InputMetadata) -> None:
        # 如果注意力偏置已经被之前的层设置，直接返回。
        if input_metadata.attn_bias:
            # Already set by a previous layer.
            return
        # Generates ALiBi mask for each prompt.
        # 遍历每个提示的长度，生成ALiBi的掩码。
        for prompt_len in input_metadata.prompt_lens:
            # 创建一个偏置矩阵，其基于两种方式的差异形成注意力偏置。
            bias = torch.arange(prompt_len)
            # Note(zhuohan): HF uses
            #     `bias = bias[None, :].repeat(prompt_len, 1)`
            # here. We find that both biases give the same results, but
            # the bias below more accurately follows the original ALiBi
            # paper.
            bias = bias[None, :] - bias[:, None]
            bias = bias.to(self.alibi_slopes.device)

            # When using custom attention bias, xformers requires the bias to
            # be sliced from a tensor whose length is a multiple of 8.
            # 确保偏置张量的长度是8的倍数。
            padded_len = (prompt_len + 7) // 8 * 8
            # 用 alibi_slopes 修改偏置。
            bias = torch.empty(
                self.num_heads,
                padded_len,
                padded_len,
                device=self.alibi_slopes.device,
            )[:, :prompt_len, :prompt_len].copy_(bias)
            bias.mul_(self.alibi_slopes[:, None, None])
            # 使用偏置创建一个下三角掩码。
            attn_bias = LowerTriangularMaskWithTensorBias(bias)
            # 将新的注意力偏置添加到输入元数据中。
            input_metadata.attn_bias.append(attn_bias)

    # 这个函数 multi_query_kv_attention 是一个执行注意力机制的函数，
    # 特别是使用了ALiBi的偏置来处理prompt tokens。
    def multi_query_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Attention with ALiBi bias for the prompt tokens.

        Args:
            output: shape = [num_prompt_tokens, num_heads, head_size]
            query: shape = [num_prompt_tokens, num_heads, head_size]
            key: shape = [num_prompt_tokens, num_kv_heads, head_size]
            value: shape = [num_prompt_tokens, num_kv_heads, head_size]
            input_metadata: metadata for paged attention.
        """
        # 如果key/value的头数量与query的头数量不同，则它会对key和value进行
        # 重复扩展，使其与query的头数量匹配。
        if self.num_kv_heads != self.num_heads:
            # Project the key and value tensors to the desired number of heads.
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=1)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=1)

        # FIXME(woosuk): Because xformers does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        start = 0
        # 它遍历每个提示的长度，并对每个prompt执行以下步骤：
        for i, prompt_len in enumerate(input_metadata.prompt_lens):
            end = start + prompt_len
            # 使用当前的 start 和 end 索引切片 query, key, value 张量。
            # 使用 xops.memory_efficient_attention_forward 执行注意力操作。
            # 将结果复制到 output 张量的相应位置。
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=input_metadata.attn_bias[i],
                p=0.0,
                scale=self.scale,
                op=self.attn_op,
            )
            # TODO(woosuk): Unnecessary copy. Optimize.
            output[start:end].copy_(out.squeeze(0))
            start += prompt_len
        return output

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        """PagedAttention with ALiBi bias for the generation tokens.

        Args:
            output: shape = [num_generation_tokens, num_heads, head_size]
            query: shape = [num_generation_tokens, num_heads, head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
        """
        block_size = value_cache.shape[3]
        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            self.head_mapping,
            self.scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            self.alibi_slopes,
        )
'''
通过这种方法， 就不需要把所有序列都padding到一个长度了浪费计算资源了，并且可以塞下更多的序列使得吞吐量增加。

但是VLLM这里也有个缺点就是无法显示控制Batch的数量，
只要Cache Engine还有空闲空间就可以持续往scheduler里面塞序列。
'''