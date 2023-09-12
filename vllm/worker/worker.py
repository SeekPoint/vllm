"""A GPU worker class."""
import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.model_executor import get_model, InputMetadata, set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel)
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata, SequenceOutputs
from vllm.worker.cache_engine import CacheEngine
from vllm.utils import get_gpu_memory

'''
0x2. worker
在llm_engine的实现中，和worker相关的就是_init_workers和_run_workers函数，_init_workers 用来初始化worker。
这些worker负责在硬件资源（如GPU）上执行计算任务，一个GPU对应一个worker。
llm_engine通过 _run_workers("<method_name>", *args, get_all_outputs, **kwargs) 来执行给定的方法。
如果 get_all_outputs 参数为 True，那么它会将所有 workers 的返回结果包装成 List 来返回。
否则，它只会返回第一个 worker 的结果，并且 assert 所有 workers 的输出都是一样的。

接下来我们对worker的实现做一个简要的解析，代码位置在 vllm/vllm/worker/worker.py 。
'''

# 这个Worker类定义了一个执行模型在GPU上的工作单元。每个工作单元都与一个单独的GPU相关联。
# 在分布式推理的情况下，每个工作单元会被分配模型的一个部分。
class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    # 这是类的构造函数，用于初始化Worker对象的新实例。它接受五个参数：
    # model_config，parallel_config，scheduler_config，rank和distributed_init_method。
    def __init__(
            self,
            model_config: ModelConfig,
            parallel_config: ParallelConfig,
            scheduler_config: SchedulerConfig,
            rank: Optional[int] = None,
            distributed_init_method: Optional[str] = None,
    ) -> None:
        # 存储传入的配置
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        # 这部分代码设置了与缓存引擎相关的未初始化属性。从注释中可以了解，
        # 这些属性将在之后的self.init_cache_engine()方法中进行初始化。
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.block_size = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

    # 在worker中初始化模型
    def init_model(self):
        # This env var set by Ray causes exceptions with graph building.
        # 这行代码删除了名为NCCL_ASYNC_ERROR_HANDLING的环境变量。
        # 从注释可以看出，这个环境变量会在构建计算图时导致异常。
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # Env vars will be set by Ray.
        # 这部分代码首先检查self.rank是否已经设定。如果没有设定，它会尝试从环境变量中获取RANK的值。
        # 然后，它从环境变量中获取LOCAL_RANK，这通常表示当前节点的编号或ID。
        # 最后，它使用此local_rank为worker设置所使用的GPU设备。
        self.rank = self.rank if self.rank is not None else int(
            os.getenv("RANK", "-1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        # 这部分代码首先确保工作单元的rank有效。如果rank是一个无效值（小于0），
        # 则会引发一个错误。接下来，它将当前CUDA设备设置为之前确定的self.device。
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank.")
        torch.cuda.set_device(self.device)

        # Initialize the distributed environment.
        # 使用给定的并行配置、rank和分布式初始化方法来初始化分布式环境。
        # 这涉及到设置必要的通信和同步机制以支持分布式训练或推理。
        _init_distributed_environment(self.parallel_config, self.rank,
                                      self.distributed_init_method)

        # Initialize the model.
        # 这里，首先设置随机种子以确保可重现性。然后，使用提供的model_config从
        # get_model函数获取模型实例并存储在self.model中。
        set_random_seed(self.model_config.seed)
        self.model = get_model(self.model_config)

    # 该函数主要用于基于配置的GPU和CPU内存约束来估计可以用于缓存块的最大数量。
    @torch.inference_mode()
    # 使用PyTorch的inference_mode来确保在执行此函数时不会跟踪计算图或计算梯度。
    # 函数名为profile_num_available_blocks，带有三个参数：块大小、GPU内存利用率和CPU交换空间。
    # 函数返回一个包含两个整数的元组，分别代表可用的GPU和CPU块数。
    def profile_num_available_blocks(
            self,
            block_size: int,
            gpu_memory_utilization: float,
            cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        # 清空CUDA内存缓存并重置内存使用峰值统计信息。
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.

        # Enable top-k sampling to reflect the accurate memory usage.
        # 获取模型的词汇表大小，然后设置采样参数。
        vocab_size = self.model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        # 从调度器配置中获取max_num_batched_tokens和max_num_seqs。
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        seqs = []
        # 这部分代码为每个序列组生成一个SequenceGroupMetadata对象，并收集这些对象以供之后的模型输入准备。
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
            )
            seqs.append(seq)

        # 使用_prepare_inputs方法准备模型的输入数据。
        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seqs)

        # Execute the model.
        # 这部分代码运行模型来估计其在给定输入上的内存使用情况。
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=[(None, None)] * num_layers,
            input_metadata=input_metadata,
            cache_events=None,
        )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        # 基于之前模型执行的内存使用情况来计算可以分配给缓存的块数。
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        # 使用CacheEngine.get_cache_block_size方法根据给定的块大小、
        # 模型配置和并行配置来计算缓存块的大小。
        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.model_config, self.parallel_config)
        # 这行代码计算了可以在GPU上分配的缓存块数量。首先，它考虑了GPU的总内存和指定的GPU内存利用率
        # （gpu_memory_utilization）。从这个总量中减去了模型操作的峰值内存使用量，
        # 然后将结果除以每个缓存块的大小来得到块的数量。
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        # 再次清空CUDA的内存缓存。
        torch.cuda.empty_cache()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        # 为确保此函数的执行不会影响模型的随机状态，重置随机种子。
        set_random_seed(self.model_config.seed)
        # 函数返回可以在GPU和CPU上分配的块数。
        return num_gpu_blocks, num_cpu_blocks

    # 这个函数的目的是初始化cache enegine
    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        # 这里我们将cache_config参数（应该是一个缓存配置对象）赋值给该类的
        # 成员变量self.cache_config，这样在类的其他部分也可以访问此配置。
        self.cache_config = cache_config
        # 从缓存配置中获取块大小并将其存储在self.block_size中。
        self.block_size = cache_config.block_size
        # 使用提供的缓存配置、模型配置和并行配置创建一个新的CacheEngine实例。
        # CacheEngine 是一个专门负责处理和管理缓存的类。
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache

    # 为模型准备输入数据
    def _prepare_inputs(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        # 初始化四个空列表，用于存储准备的数据。
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        # Add prompt tokens.
        # 初始化一个空列表，用于存储提示的长度。
        prompt_lens: List[int] = []
        # 遍历传入的seq_group_metadata_list列表。
        for seq_group_metadata in seq_group_metadata_list:
            # 如果当前seq_group_metadata不是提示，则跳过当前迭代。
            if not seq_group_metadata.is_prompt:
                continue

            # 获取当前seq_group_metadata的序列ID和采样参数，并将它们添加到seq_groups列表中。
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            # 将prompt tokens添加到input_tokens列表中，
            # 并为每个token添加一个位置到input_positions列表中。
            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(range(len(prompt_tokens)))

            # 如果当前的seq_group_metadata没有块表，为slot_mapping添加0，并跳过当前迭代。
            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([0] * prompt_len)
                continue

            # Compute the slot mapping.
            # 这个代码段计算了如何在内存中布局tokens，它使用了块大小和块表。
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # Add generation tokens.
        # 这里初始化了两个整数用于存储最大的上下文长度和序列块表长度的最大值，
        # 以及两个列表用于存储上下文的长度和每个生成的序列对应的块表。
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        # 对seq_group_metadata_list中的每个seq_group_metadata进行迭代。
        for seq_group_metadata in seq_group_metadata_list:
            # 如果seq_group_metadata是一个提示，则跳过此次迭代，因为我们在这里只处理生成部分。
            if seq_group_metadata.is_prompt:
                continue

            # 获取序列ID和采样参数，并将它们以元组形式添加到seq_groups列表中。
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # 遍历每一个序列ID。
            for seq_id in seq_ids:
                # 对于当前的seq_id，获取其对应的序列数据。
                seq_data = seq_group_metadata.seq_data[seq_id]
                # 从seq_data获取最后一个token ID并将其添加到input_tokens列表中。
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                # 取seq_data的长度，并计算位置（为最后一个token的位置），
                # 然后将位置添加到input_positions列表中。
                context_len = seq_data.get_len()
                position = context_len - 1
                input_positions.append(position)

                # 获取seq_id对应的块表，并将其添加到generation_block_tables列表中。
                block_table = seq_group_metadata.block_tables[seq_id]
                generation_block_tables.append(block_table)

                # 更新max_context_len和max_num_blocks_per_seq的值。
                max_context_len = max(max_context_len, context_len)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq,
                                             len(block_table))
                # 将context_len添加到context_lens列表中。
                context_lens.append(context_len)

                # 计算当前token在内存中的插槽，并将其添加到slot_mapping列表中。
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        # 这里使用一个名为_pad_to_alignment的函数来将input_tokens和
        # input_positions列表的长度填充到8的倍数。
        # 这是一个针对NVIDIA Tensor Cores的优化，可以确保更高的性能。
        input_tokens = _pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = _pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        # 上述代码将input_tokens和input_positions列表转化为PyTorch在CUDA上的LongTensor张量。
        tokens_tensor = torch.cuda.LongTensor(input_tokens)
        positions_tensor = torch.cuda.LongTensor(input_positions)
        # 然后，slot_mapping和context_lens也被转换为在CUDA上的IntTensor张量。
        slot_mapping_tensor = torch.cuda.IntTensor(slot_mapping)
        context_lens_tensor = torch.cuda.IntTensor(context_lens)
        # 对generation_block_tables中的每个block_table进行迭代，
        # 并使用_pad_to_max函数来对其进行填充，确保其长度达到max_num_blocks_per_seq。
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in generation_block_tables
        ]
        # 将填充后的padded_block_tables转换为CUDA上的IntTensor张量。
        block_tables_tensor = torch.cuda.IntTensor(padded_block_tables)

        # 初始化一个空字典seq_data，然后遍历seq_group_metadata_list来收集所有的seq_data。
        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        # 使用收集到的数据来创建InputMetadata对象。
        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,
        )
        return tokens_tensor, positions_tensor, input_metadata

    @torch.inference_mode()
    def execute_model(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            blocks_to_swap_in: Dict[int, int],
            blocks_to_swap_out: Dict[int, int],
            blocks_to_copy: Dict[int, List[int]],
    ) -> Dict[int, SequenceOutputs]:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        if issued_cache_op:
            cache_events = self.cache_events
        else:
            cache_events = None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # Prepare input tensors.
        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seq_group_metadata_list)

        # Execute the model.
        output = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.gpu_cache,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )
        return output


def _init_distributed_environment(
        parallel_config: ParallelConfig,
        rank: int,
        distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)


def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)


def _pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))

'''
我们对worker的几个关键函数进行了解析，总结一下：

        init_model：对模型进行初始化。
        
        profile_num_available_blocks: 这个函数首先运行一次模型来profile内存峰值占用并以次计算没张卡上的blocks个数。
        
        init_cache_engine：初始化 cache engine。
        
        execute_model：执行模型。

llm_engine通过 _run_workers("<method_name>", *args, get_all_outputs, **kwargs) 来和上面的三个函数建立起联系。
从llm_engine.step函数我们基本可以看到scheduler，worker的关系：
            001.webp

在原始的LLM类的genarete函数中，对于每个输入的prompt，都会给 llm engine 生成一个 request并添加到scheduler里。
然后调用 _run_engine 函数，这个函数的逻辑是对于所有未完成的 requests，就调用 llm engine 的 step 函数得到这一步的 outputs，然后 append 到返回的 List 里。
在step函数里，由scheduler获取本次要作为输入的 seq_group_metadata_list ，同时产生一个 scheduler_outputs。
然后 engine 会调用 worker 的 execute_model 来执行对 seq_group_metadata_list 的模型前向计算。
'''