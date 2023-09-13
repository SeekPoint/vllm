import time
import copy
from functools import partial
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.core.scheduler import Scheduler
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.ray_utils import initialize_cluster, ray, RayWorker
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
                                               get_tokenizer)
from vllm.utils import Counter

if ray:
    from ray.air.util.torch_dist import init_torch_dist_process_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5


class LLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The `LLM` class wraps this class for offline batched inference and the
    `AsyncLLMEngine` class wraps this class for online serving.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        distributed_init_method: The initialization method for distributed
            execution. See `torch.distributed.init_process_group` for details.
        stage_devices: The list of devices for each stage. Each stage is a list
            of (rank, node_resource, device) tuples.
        log_stats: Whether to log statistics.
    """
    """这段代码定义了一个名为 LLMEngine 的类，它是一个接收请求并生成文本的语言模型(LLM)引擎。
   
   这个类是vLLM引擎的主要类，它从客户端接收请求，并从LLM生成文本。
   这个类包含了一个分词器，一个语言模型（可能在多个GPU之间切分），
   以及为中间状态（也称为KV缓存）分配的GPU内存空间。
   此类使用了迭代级别的调度和有效的内存管理来最大化服务吞吐量。

    LLM 类将此类封装用于离线批量推理，而 AsyncLLMEngine 类将此类封装用于在线服务

    注意：配置参数源自 EngineArgs 类。有关参数的完整列表，请参见 EngineArgs。

    Args:
        model_config: 与LLM模型相关的配置。
        cache_config: 与KV缓存内存管理相关的配置。
        parallel_config: 与分布式执行相关的配置。
        scheduler_config: 与分布式执行相关的配置。
        distributed_init_method: 分布式执行的初始化方法，参见torch.distributed.init_process_group了解详情。
        stage_devices: 每个stage的设备列表. 每个stage都是一个(rank, node_resource, device)元组.
        log_stats: 是否记录统计数据。
        
    从LLMEngine的定义可以知道，它做了初始化tokenizer，
    创建并行的worker信息以及初始化KV Cache等事情，
    这里的worker是每个GPU对应一个，我们稍后会讲到。    
    """
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        placement_group: Optional["PlacementGroup"],
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"use_dummy_weights={model_config.use_dummy_weights}, "
            f"download_dir={model_config.download_dir!r}, "
            f"use_np_weights={model_config.use_np_weights}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"seed={model_config.seed})")
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        self._verify_args()

        # 设置tokenizer
        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code)
        self.seq_counter = Counter()

        # 对于每个 device（也即每张卡 / 每个 rank）创建一个 Worker。
        # Worker 是运行 model 的单位。一个 Engine 管理所有的 workers。
        # Create the parallel GPU workers.
        if self.parallel_config.worker_use_ray:
            self._init_workers_ray(placement_group)
        else:
            self._init_workers(distributed_init_method)

        # 初始化这个 engine 的 KV cache。
        # Profile the memory usage and initialize the cache.
        self._init_cache()

        # Create the scheduler.
        self.scheduler = Scheduler(scheduler_config, cache_config)

        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []

    # _init_workers 这个方法是 LLMEngine 类的一个私有方法，其主要目的是初始化worker。
    # 这些worker负责在硬件资源（如GPU）上执行计算任务。
    # 这个函数只接受一个参数，即 distributed_init_method，它是一个字符串，用于指定分布式执行的初始化方法。
    def _init_workers(self, distributed_init_method: str):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker

        # 从vllm.worker.worker模块中导入Worker类。这个导入操作被放在了函数内部，
        # 这样做的目的是为了避免在CUDA_VISIBLE_DEVICES被Worker类设定之前就导入
        # 了torch.cuda/xformers，因为那样可能会产生问题。
        from vllm.worker.worker import Worker  # pylint: disable=import-outside-toplevel

        # 断言self.parallel_config.world_size（并行世界的大小）等于1，如果不等于1，
        # 则会抛出错误，提示用户需要使用Ray框架进行并行计算。
        assert self.parallel_config.world_size == 1, (
            "Ray is required if parallel_config.world_size > 1.")

        self.workers: List[Worker] = []
        # 创建一个新的 Worker 对象，并将其添加到 self.workers 列表中。
        # 每个 Worker 对象都需要以下参数：
        # self.model_config，self.parallel_config，self.scheduler_config，
        # 以及工作节点的 rank（在这个例子中，rank 是0，表示这是第一个，也是唯一的工作节点）
        # 和 distributed_init_method。
        worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            0,
            distributed_init_method,
        )

        # 调用_run_workers方法，参数为 "init_model"
        # 和 get_all_outputs=True，对所有的worker进行初始化。
        self.workers.append(worker)
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )

    def _init_workers_ray(self, placement_group: "PlacementGroup"):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker  # pylint: disable=import-outside-toplevel

        self.workers: List[Worker] = []
        for bundle in placement_group.bundle_specs:
            if not bundle.get("GPU", 0):
                continue
            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True),
            )(RayWorker).remote()
            self.workers.append(worker)

        # Initialize torch distributed process group for the workers.
        init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        self._run_workers("init_worker",
                          get_all_outputs=True,
                          worker_init_fn=lambda: Worker(
                              model_config,
                              parallel_config,
                              scheduler_config,
                              None,
                              None,
                          ))
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )

    def _verify_args(self) -> None:
        # 这行代码调用model_config对象的verify_with_parallel_config方法来检查模型配置是否与并行配置兼容。
        # self.model_config和self.parallel_config分别是ModelConfig和ParallelConfig对象，
        # 它们包含了模型和并行计算的相关配置。
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    # _init_cache函数是LLMEngine类的一个私有方法，不接受任何参数，没有返回值。
    # 其目标是测量内存使用并初始化KV（键值）Cache。
    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        # 使用_run_workers方法来获取可以在GPU和CPU上分配的最大块数量。
        # _run_workers函数执行的方法是profile_num_available_blocks，并且提供了如块大小、
        # GPU内存使用率和CPU交换空间等参数，所有这些参数都是从cache_config对象中提取出来的。
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        # 找到所有workers中可用块的最小值，以确保所有的内存操作都可以应用到所有worker。
        # 在这个步骤中，函数分别计算了GPU和CPU的块数量。
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        # 如果GPU的块数量小于等于0，函数将抛出一个值错误。
        # 这是为了确保在初始化引擎时，为缓存块提供足够的可用内存。
        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")

        # 根据计算的块数量，更新cache_config对象的num_gpu_blocks和num_cpu_blocks属性。
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        # 使用_run_workers方法初始化缓存。此步骤中的_run_workers执行的方法
        # 是init_cache_engine，并且提供了cache_config对象作为参数。
        self._run_workers("init_cache_engine", cache_config=self.cache_config)

    # from_engine_args是一个类方法（classmethod），这意味着它可以在没有创建类实例的情况下调用。
    # 此方法需要接受一个EngineArgs类型的参数engine_args，并返回一个LLMEngine类型的对象。
    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        # 利用create_engine_configs方法从engine_args参数中创建引擎配置，该方法返回一个配置的列表，
        # 其中包含了model_config、cache_config、parallel_config和scheduler_config。
        # 这些配置都是初始化LLMEngine对象所需的。
        engine_configs = engine_args.create_engine_configs()

        # 提取并保存parallel_config，这是一个关于分布式执行的配置。
        parallel_config = engine_configs[2]

        # 使用initialize_cluster方法初始化集群，该方法接受parallel_config作为参数并返回两个结果：
        # distributed_init_method（分布式执行的初始化方法）和placement_group。
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config)

        # 利用之前创建的引擎配置、初始化方法、放置组以及日志统计设置来创建一个新的LLMEngine对象。
        # 这里使用了Python的*操作符，这样可以把列表中的每个元素分别作为单独的参数传递给LLMEngine类
        # 的构造方法。log_stats参数用于决定是否需要记录统计数据，如果engine_args.disable_log_stats
        # 为True，则不记录统计数据。
        # Create the LLM engine.
        engine = cls(*engine_configs,
                     distributed_init_method,
                     placement_group,
                     log_stats=not engine_args.disable_log_stats)
        return engine

    # add_request函数是LLMEngine类的一个方法，它接受一个请求并将其加入到scheduler的请求池中。
    # 这个请求在调用engine.step()函数时由调度器进行处理，具体的调度策略由调度器决定。
    def add_request(
        self,
        request_id: str,  # 请求的唯一ID。
        prompt: Optional[str],  # prompt字符串。如果提供了prompt_token_ids，这个参数可以为None。
        sampling_params: SamplingParams,  # 用于文本生成的采样参数。
        # prompt的token ID。如果它为None，则使用分词器将提示转换为token ID。
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,  # 请求的到达时间。如果为None，则使用当前时间。
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current time.
        """
        if arrival_time is None:
            arrival_time = time.time()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        # 每一个序列代表一次独立的文本生成任务。它们的数量由sampling_params.best_of决定。
        # 每个序列都包含了唯一的seq_id，提示和标记ID，以及block_size（块大小）。
        block_size = self.cache_config.block_size
        seqs: List[Sequence] = []
        for _ in range(sampling_params.best_of):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)
            seqs.append(seq)

        # Create the sequence group.
        # 创建序列组（SequenceGroup）。一个序列组包含了一组相关的序列，
        # 它们共享相同的请求ID和采样参数，并且在同一时间到达。
        seq_group = SequenceGroup(request_id, seqs, sampling_params,
                                  arrival_time)

        # Add the sequence group to the scheduler.
        # 将序列组添加到调度器中。这样，当调用engine.step()函数时，
        # 调度器就可以根据它的调度策略处理这些序列组。
        self.scheduler.add_seq_group(seq_group)

    # 此函数接受一个请求ID作为参数，并调用调度器的abort_seq_group方法来终止具有该ID的请求。
    # 简单地说，这个函数的目的是取消一个特定的请求。
    def abort_request(self, request_id: str) -> None:
        """Aborts a request with the given ID.

        Args:
            request_id: The ID of the request to abort.
        """
        self.scheduler.abort_seq_group(request_id)

    # 此函数没有参数，并返回当前的模型配置。模型配置是一个ModelConfig对象，
    # 包含模型和分词器的配置信息，以及其他可能的模型相关的配置选项。
    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    # 此函数也没有参数，它返回未完成的请求的数量。它通过调用调度器的
    # get_num_unfinished_seq_groups方法实现，该方法返回
    # 未完成的序列组的数量，因为每个请求都对应一个序列组。
    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    # 此函数没有参数，并返回一个布尔值，指示是否有未完成的请求。
    # 这是通过调用调度器的has_unfinished_seqs方法实现的。
    # 如果有未完成的序列，该函数返回True，否则返回False。
    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    # 逻辑还是很清晰的，首先调用scheduler.schedule()
    # 得到scheduler_outputs，scheduler_outputs中包含了swap_in、swap_out等映射，
    # 通过worker的execute_model得到逻辑输出，使用scheduler.update对输出进行更新，将使用完的block置换出。
    # 然后对已经完成的seq解码，更新seq状态，返回输出。

    # 这个函数是 LLMEngine 类的一个关键函数，其功能是执行一次解码迭代并返回新生成的结果。
    # 这个函数的执行过程可以分解为以下步骤：
    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """

        # 0x6.ModelExecution
        # 接下来我们对Model
        # Execution做一个介绍，ModelExecution是scheduler.schedule函数之后执行的，对应这几行代码：

        # 首先，调用 self.scheduler.schedule() 进行调度，返回要在下一次迭代中执行的序列，
        # 以及要被换入，换出，复制的 token 块。
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

        # 然后，检查 scheduler_outputs 是否为空。如果为空并且没有被忽略的序列组，
        # 则表示没有需要做的工作，函数返回空列表。如果存在被忽略的序列组，那么我们需要将它们作为请求输出返回。
        if scheduler_outputs.is_empty():
            if not scheduler_outputs.ignored_seq_groups:
                # Nothing to do.
                return []
            # If there are ignored seq groups, we need to return them as the
            # request outputs.
            return [
                RequestOutput.from_seq_group(seq_group)
                for seq_group in scheduler_outputs.ignored_seq_groups
            ]

        # 这里调用了worker的execute_model函数来做模型的推理，在模型推理中比较特殊的一点就是使用了PagedAttention，
        # 以及如果在模型执行前传入的SchedulerOutputs对象中的swap in / out
        # 的blocks，blocks_to_copy等不空的话，我们需要在获取KV Cache之间做一个全局同步让Block相关的操作做完，
        # 这是通过在Transformer的每一层插入一个cuda Event来实现的。
        # 这里我们来分析一下模型执行过程里面最核心的部分也就是PagedAttention。

        # Execute the model.
        # 如果 scheduler_outputs 不为空，那么就会执行模型，将 seq_group_metadata_list、
        # blocks_to_swap_in、blocks_to_swap_out 和 blocks_to_copy 作为参数传给 _run_workers
        # 方法。这一步可能包括将一些状态从内存移到 GPU，执行模型计算，以及将一些状态从 GPU 移回内存。
        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        # Update the scheduler with the model outputs.
        # 之后，使用模型的输出结果来更新调度器。
        seq_groups = self.scheduler.update(output)

        # Decode the sequences.
        # 然后对序列进行解码，并终止满足停止条件的序列。完成的序列组将被释放。
        self._decode_sequences(seq_groups)
        # Stop the sequences that meet the stopping criteria.
        self._stop_sequences(seq_groups)
        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        # 最后，创建输出结果。对于每一个序列组，都将其转换为 RequestOutput 对象
        # 并添加到输出列表中。如果 log_stats 为真，那么还会记录系统状态。
        request_outputs: List[RequestOutput] = []
        for seq_group in seq_groups + scheduler_outputs.ignored_seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

        if self.log_stats:
            # Log the system stats.
            self._log_system_stats(scheduler_outputs.prompt_run,
                                   scheduler_outputs.num_batched_tokens)
        return request_outputs

    # _log_system_stats 函数的主要作用是记录和打印系统状态信息。
    # 这些信息可以帮助理解系统的运行状态，包括吞吐量、请求的运行情况以及GPU和CPU的KV缓存使用情况等。
    # 这是一个监控和调试工具，用于跟踪系统性能和资源使用情况。
    def _log_system_stats(
        self,
        prompt_run: bool,
        num_batched_tokens: int,
    ) -> None:
        now = time.time()
        # Log the number of batched input tokens.
        # 函数首先记录当前时间，并根据prompt_run值，记录批处理输入标记的数量。
        if prompt_run:
            self.num_prompt_tokens.append((now, num_batched_tokens))
        else:
            self.num_generation_tokens.append((now, num_batched_tokens))

        # 接下来，函数检查自上次记录日志以来是否已经过了足够的时间（由 _LOGGING_INTERVAL_SEC 定义）。
        # 如果时间还没有过去足够，函数就会返回，不进行任何操作。
        elapsed_time = now - self.last_logging_time
        if elapsed_time < _LOGGING_INTERVAL_SEC:
            return

        # Discard the old stats.
        # 如果已经过了足够的时间，函数会丢弃过旧的统计信息，
        # 只保留在 _LOGGING_INTERVAL_SEC 时间窗口内的数据。
        self.num_prompt_tokens = [(t, n) for t, n in self.num_prompt_tokens
                                  if now - t < _LOGGING_INTERVAL_SEC]
        self.num_generation_tokens = [(t, n)
                                      for t, n in self.num_generation_tokens
                                      if now - t < _LOGGING_INTERVAL_SEC]

        # 函数然后计算prompt和genaration的平均吞吐量。
        # 这是通过计算在指定时间窗口内处理的标记总数，然后除以时间窗口的长度来实现的。
        if len(self.num_prompt_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_prompt_tokens[:-1])
            window = now - self.num_prompt_tokens[0][0]
            avg_prompt_throughput = total_num_tokens / window
        else:
            avg_prompt_throughput = 0.0
        if len(self.num_generation_tokens) > 1:
            total_num_tokens = sum(n
                                   for _, n in self.num_generation_tokens[:-1])
            window = now - self.num_generation_tokens[0][0]
            avg_generation_throughput = total_num_tokens / window
        else:
            avg_generation_throughput = 0.0

        # 函数接下来计算 GPU 和 CPU 的 KV 缓存使用情况。这是通过查看已使用的和总的缓存块的数量来实现的。
        total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = (
            self.scheduler.block_manager.get_num_free_gpu_blocks())
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

        total_num_cpu_blocks = self.cache_config.num_cpu_blocks
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = (
                self.scheduler.block_manager.get_num_free_cpu_blocks())
            num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0
        # 最后，函数使用 logger.info 打印出所有的统计信息，并更新最后一次记录日志的时间。
        logger.info("Avg prompt throughput: "
                    f"{avg_prompt_throughput:.1f} tokens/s, "
                    "Avg generation throughput: "
                    f"{avg_generation_throughput:.1f} tokens/s, "
                    f"Running: {len(self.scheduler.running)} reqs, "
                    f"Swapped: {len(self.scheduler.swapped)} reqs, "
                    f"Pending: {len(self.scheduler.waiting)} reqs, "
                    f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                    f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%")
        self.last_logging_time = now

    # 它遍历给定的序列组（seq_groups），并对每个处于运行状态的序列进行解码。
    # 通过逐渐解码，它可以有效地处理和更新正在运行的序列，从而支持逐步生成和流式处理的用例。
    def _decode_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        """Decodes the sequence outputs."""
        # 函数首先遍历传入的序列组列表。每个序列组都包含一组相关的序列。
        for seq_group in seq_groups:
            # 对于每个序列组，使用get_seqs方法并传入status=SequenceStatus.RUNNING参数，
            # 从序列组中选择所有处于运行状态的序列。只有运行状态的序列才需要解码。
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # 对于每个运行中的序列，函数调用detokenize_incrementally方法，
                # 将序列的output_tokens逐渐解码为文本。这个解码过程考虑了最后一个
                # 已经解码的标记，以及是否跳过特殊的标记。
                # detokenize_incrementally方法返回新解码的token（new_token）
                # 和新解码的输出文本（new_output_text）。
                new_token, new_output_text = detokenize_incrementally(
                    self.tokenizer,
                    seq.output_tokens,
                    seq.get_last_token_id(),
                    skip_special_tokens=True,
                )
                # 如果有新的token被解码（new_token不是None），则将其追加到序列的output_tokens
                # 列表中，并更新序列的output_text属性为新解码的文本。
                if new_token is not None:
                    seq.output_tokens.append(new_token)
                    seq.output_text = new_output_text

    # 这个函数_stop_sequences的主要目的是停止已完成的序列。完成可以由几种不同的条件定义，
    # 例如序列生成了特定的停止字符串、达到了最大模型长度、达到了最大token数或生成了结束符（EOS）。
    def _stop_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        """Stop the finished sequences."""
        # 函数开始通过遍历提供的序列组列表，检查每个组中的序列。
        for seq_group in seq_groups:
            # 当前序列组中获取采样参数sampling_params，它包含此序列组的特定设置，如停止字符串、最大标记数等。
            sampling_params = seq_group.sampling_params
            # 对于每个序列组，函数筛选出处于运行状态的序列进行处理
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # Check if the sequence has generated a stop string.
                stopped = False
                # 通过遍历sampling_params.stop中的每个停止字符串，
                # 检查当前序列的输出文本是否以其中任何一个停止字符串结束。
                for stop_str in sampling_params.stop:
                    if seq.output_text.endswith(stop_str):
                        # Truncate the output text so that the stop string is
                        # not included in the output.
                        # 如果序列以停止字符串结束，函数将截断该字符串，并将序列的状态设置为FINISHED_STOPPED。
                        seq.output_text = seq.output_text[:-len(stop_str)]
                        # 一旦找到停止字符串，就不再检查其余的停止字符串，并跳到下一个序列。
                        self.scheduler.free_seq(
                            seq, SequenceStatus.FINISHED_STOPPED)
                        stopped = True
                        break
                if stopped:
                    continue

                # Check if the sequence has reached max_model_len.
                # 如果序列的长度大于scheduler_config.max_model_len，则将其状态设置
                # 为FINISHED_LENGTH_CAPPED并继续处理下一个序列。
                if seq.get_len() > self.scheduler_config.max_model_len:
                    self.scheduler.free_seq(
                        seq, SequenceStatus.FINISHED_LENGTH_CAPPED)
                    continue
                # Check if the sequence has reached max_tokens.
                # 如果序列的输出长度等于sampling_params.max_tokens，
                # 则将其状态设置为FINISHED_LENGTH_CAPPED并继续处理下一个序列
                if seq.get_output_len() == sampling_params.max_tokens:
                    self.scheduler.free_seq(
                        seq, SequenceStatus.FINISHED_LENGTH_CAPPED)
                    continue
                # Check if the sequence has generated the EOS token.
                # 如果sampling_params.ignore_eos为False，并且序列的最后一个标记ID等于
                # 分词器的eos_token_id，则将序列的状态设置为FINISHED_STOPPED并继续。
                if not sampling_params.ignore_eos:
                    if seq.get_last_token_id() == self.tokenizer.eos_token_id:
                        self.scheduler.free_seq(
                            seq, SequenceStatus.FINISHED_STOPPED)
                        continue

    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        # 创建一个名为all_outputs的空列表，用于存储每个worker的输出。
        all_outputs = []
        # 通过遍历self.workers中的每个工作线程来运行给定的方法。
        for worker in self.workers:
            # 如果self.parallel_config.worker_use_ray为True，则使用远程执行（Ray框架的一部分）。
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                # 如果为False，则直接在worker上获取和调用方法。
                executor = getattr(worker, method)

            # 用executor函数运行给定的方法，并将结果添加到all_outputs列表中。
            output = executor(*args, **kwargs)
            all_outputs.append(output)

        # 如果使用了Ray进行远程执行，那么需要使用ray.get来获取远程执行的结果。
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)

        # 如果get_all_outputs参数为True，则返回all_outputs列表，其中包括每个worker的输出。
        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        # 如果为False，则确保所有工作线程的输出都相同，并仅返回第一个工作线程的输出。
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output

'''
LLMEngine实现了一个处理序列生成任务的引擎，我们对它的函数进行一个总结：

__init__：通过特定的EngineArgs来初始化LLMEngine，包括初始化tokenizer，
创建并行的worker信息以及初始化KV Cache，创建调度器。

_init_workers：初始化worker。这
些worker负责在硬件资源（如GPU）上执行计算任务，一个GPU对应一个worker。

_init_cache：profile gpu内存使用并初始化KV（键值）Cache。

from_engine_args：此方法需要接受一个EngineArgs类型的参数engine_args，
并返回一个LLMEngine类型的对象。

add_request: 它接受一个请求并将其加入到scheduler的请求池中。
 这个请求在调用engine.step()函数时由scheduler进行处理，具体的调度策略由scheduler决定。
 
abort_request：终止某个请求id的请求。

get_model_config返回当前的模型配置。

get_num_unfinished_requests：返回scheduler中未完成的序列组（SequenceGroup）的数量，
因为每个请求都对应一个序列组。

has_unfinished_requests：scheduler中是否有未完成的序列（Sequence）。

step：step 过程先从 scheduler 获取本次要作为输入的 seq_group_metadata_list ，
同时产生一个 scheduler_outputs 和 ignored_seq_groups。
然后 engine 会调用 workers 的 execute_model。

_log_system_stats：主要作用是记录和打印系统状态信息。
这些信息可以帮助理解系统的运行状态，包括吞吐量、请求的运行情况以及GPU和CPU的KV缓存使用情况等。 
这是一个监控和调试工具，用于跟踪系统性能和资源使用情况。

_decode_sequences：它遍历给定的序列组（seq_groups），并对每个处于运行状态的序列进行解码。

_stop_sequences： 这个函数的主要目的是停止已完成的序列。
完成可以由几种不同的条件定义，
例如序列生成了特定的停止字符串、达到了最大模型长度、达到了最大token数或生成了结束符（EOS）。

_run_workers：在step函数中调用，实际上就是在每个GPU上的worker的模型推理。

从LLMEngine实现的函数来看，vllm关键的几个组件scheduler，worker，cache engine已经出现了。
这三个组件的解析我们会分别单独开一节，这一节还是以走完整个vllm的generate流程为主旨。
'''