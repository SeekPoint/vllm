"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple

import torch

from vllm import cache_ops
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl

logger = init_logger(__name__)
'''
0x3. cache engine
上面的worker实现和普通的PyTorch执行模型前向最大的一个区别是它维护了一个cache engine，它是用管理模型的KV Cache的。
下面仍然是对它的实现进行解析，它的实现在vllm/vllm/worker/cache_engine.py这个文件。
'''
KVCache = Tuple[torch.Tensor, torch.Tensor]


# CacheEngine类的主要责任是初始化和管理GPU和CPU上的KV Cache，并为KV Cache 操作如交换和复制提供方法。
class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    # 这个构造函数接受三个参数：cache_config、model_config和parallel_config，
    # 分别对应缓存配置、模型配置和并行配置。
    def __init__(
            self,
            cache_config: CacheConfig,
            model_config: ModelConfig,
            parallel_config: ParallelConfig,
    ) -> None:
        # 下面三行代码保存了传入构造函数的配置信息，供类的其他方法使用。
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        # 根据模型配置提取了头的大小、层数、头的数量和数据类型，并保存为类的成员变量。
        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_heads(parallel_config)
        self.dtype = model_config.dtype

        # 这里从缓存配置中获取块的大小、GPU上的块数量和CPU上的块数量。
        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        # Initialize the cache.
        # 这两行代码调用了这个类的2个成员函数来分配GPU和CPU缓存，并将结果保存为类的成员变量。
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        # 首先创建了一个新的CUDA流并保存。接下来，它使用assert确保新创建的流不是当前的CUDA流。
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        # 这行代码为每层创建了一个CUDA事件，并保存为一个列表。CUDA事件主要用于同步CUDA流。
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    # 这是CacheEngine类的一个成员函数get_key_block_shape，该函数的目的是返回KV Cache中key block的shape（维度）
    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        # torch.tensor([], dtype=self.dtype)：创建一个空的Tensor，其数据类型由类的dtype属性指定。
        # .element_size()：此方法返回该Tensor的数据类型的元素大小（以字节为单位）。
        # 例如，如果dtype是torch.float32（即32位浮点数），那么element_size将是4（字节）。
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        # 这行代码将16除以前面计算得到的element_size（并执行整数除法），得到的结果赋值给变量x。
        # 假设dtype是torch.float32（元素大小为4字节），那么x将是4。
        x = 16 // element_size
        # 这里构建并返回一个由四个整数构成的元组，这些整数描述了key block的形状。具体来说，形状的每个维度如下：
        # 头的数量（由类的num_heads属性指定）。
        # 头的大小除以x。
        # 块的大小（由类的block_size属性指定）。
        # x。
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )

    # 返回value block的形状
    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )

    # 在GPU上申请key_block和value_block的内存
    def allocate_gpu_cache(self) -> List[KVCache]:
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache

    # 在CPU上申请key_block和value_block的内存
    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache

    def _swap(
            self,
            src: List[KVCache],
            dst: List[KVCache],
            src_to_dst: Dict[int, int],
    ) -> None:
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(src_value_cache, dst_value_cache,
                                      src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    # paged attention的swap操作，有点像操作系统里的 swap 概念。in 就是 cpu to gpu，
    # out 就是 gpu to cpu。内部实现由专门的 cu 函数 swap_blocks 支持。
    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    # paged attention的copy操作，由专门的 cu 函数 copy_blocks 支持。
    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)

    # 这个函数get_cache_block_size是CacheEngine类的静态方法，用于计算缓存块的大小。
    @staticmethod
    def get_cache_block_size(
            block_size: int,
            model_config: ModelConfig,
            parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = _get_dtype_size(model_config.dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

'''
Cache Engine根据之前调用workers里面profile 的数据（cpu/gpu blocks数）来申请 cache 内存。
然后再给 caching 操作初始化一个 cuda Stream，并且为每个网络层创建了一个CUDA事件，并保存为一个列表。

观察到key block的形状为[num_heads, head_size // x, block_size, x]，
符号的具体含义看上面的注释，这里为什么要// x 还不清楚，后续在cuda实现应该可以找到答案。

接着我们看到Paged attention中对KV Cache的经典操作copy，swap_in/out都是由cuda kernel实现然后通过torch extension模块导出的。

接下来对cache相关的cuda算子进行解析。
'''