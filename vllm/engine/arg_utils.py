import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)

# 这段代码定义了一个名为EngineArgs的Python数据类，该类包含vLLM engine 的所有参数。
@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str # 字符串类型，指定模型的名称。
    tokenizer: Optional[str] = None # 可选的字符串类型，用于指定分词器的名称。
    tokenizer_mode: str = 'auto' # 字符串类型，指定分词器的模式。
    trust_remote_code: bool = False # 布尔类型，是否信任远程代码。
    download_dir: Optional[str] = None # 可选的字符串类型，指定模型和分词器的下载路径。
    use_np_weights: bool = False #  布尔类型，是否使用numpy权重。
    use_dummy_weights: bool = False #  布尔类型，是否使用虚拟权重。
    dtype: str = 'auto' # 字符串类型，指定数据类型。
    seed: int = 0 # 整型，用于初始化随机数生成器的种子。
    worker_use_ray: bool = False # 布尔类型，worker节点是否使用ray库。
    pipeline_parallel_size: int = 1 # 整型，指定流水线并行的大小。
    tensor_parallel_size: int = 1 # 整型，指定张量并行的大小。
    block_size: int = 16 # 整型，指定block_size的大小。
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90 # 浮点型，指定GPU内存使用率。
    max_num_batched_tokens: int = 2560 # 整型，指定最大批量token数。
    max_num_seqs: int = 256 #  整型，一个 iteration 最多处理多少个tokens。
    disable_log_stats: bool = False # 布尔类型，是否禁用日志统计。

    # 它在数据类实例化后立即执行。该函数将tokenizer设置为model（如果tokenizer为None），
    # 并将max_num_seqs设置为max_num_seqs和max_num_batched_tokens之间的最小值。
    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model
        self.max_num_seqs = min(self.max_num_seqs, self.max_num_batched_tokens)

    # 它接受一个argparse.ArgumentParser实例，并添加共享的命令行接口参数。
    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for vLLM engine."""
        # Model arguments
        parser.add_argument(
            '--model',
            type=str,
            default='facebook/opt-125m',
            help='name or path of the huggingface model to use')
        parser.add_argument(
            '--tokenizer',
            type=str,
            default=EngineArgs.tokenizer,
            help='name or path of the huggingface tokenizer to use')
        parser.add_argument('--tokenizer-mode',
                            type=str,
                            default=EngineArgs.tokenizer_mode,
                            choices=['auto', 'slow'],
                            help='tokenizer mode. "auto" will use the fast '
                            'tokenizer if available, and "slow" will '
                            'always use the slow tokenizer.')
        parser.add_argument('--trust-remote-code',
                            action='store_true',
                            help='trust remote code from huggingface')
        parser.add_argument('--download-dir',
                            type=str,
                            default=EngineArgs.download_dir,
                            help='directory to download and load the weights, '
                            'default to the default cache dir of '
                            'huggingface')
        parser.add_argument('--use-np-weights',
                            action='store_true',
                            help='save a numpy copy of model weights for '
                            'faster loading. This can increase the disk '
                            'usage by up to 2x.')
        parser.add_argument('--use-dummy-weights',
                            action='store_true',
                            help='use dummy values for model weights')
        # TODO(woosuk): Support FP32.
        parser.add_argument(
            '--dtype',
            type=str,
            default=EngineArgs.dtype,
            choices=['auto', 'half', 'bfloat16', 'float'],
            help='data type for model weights and activations. '
            'The "auto" option will use FP16 precision '
            'for FP32 and FP16 models, and BF16 precision '
            'for BF16 models.')
        # Parallel arguments
        parser.add_argument('--worker-use-ray',
                            action='store_true',
                            help='use Ray for distributed serving, will be '
                            'automatically set when using more than 1 GPU')
        parser.add_argument('--pipeline-parallel-size',
                            '-pp',
                            type=int,
                            default=EngineArgs.pipeline_parallel_size,
                            help='number of pipeline stages')
        parser.add_argument('--tensor-parallel-size',
                            '-tp',
                            type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='number of tensor parallel replicas')
        # KV cache arguments
        parser.add_argument('--block-size',
                            type=int,
                            default=EngineArgs.block_size,
                            choices=[8, 16, 32],
                            help='token block size')
        # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
        parser.add_argument('--seed',
                            type=int,
                            default=EngineArgs.seed,
                            help='random seed')
        parser.add_argument('--swap-space',
                            type=int,
                            default=EngineArgs.swap_space,
                            help='CPU swap space size (GiB) per GPU')
        parser.add_argument('--gpu-memory-utilization',
                            type=float,
                            default=EngineArgs.gpu_memory_utilization,
                            help='the percentage of GPU memory to be used for'
                            'the model executor')
        parser.add_argument('--max-num-batched-tokens',
                            type=int,
                            default=EngineArgs.max_num_batched_tokens,
                            help='maximum number of batched tokens per '
                            'iteration')
        parser.add_argument('--max-num-seqs',
                            type=int,
                            default=EngineArgs.max_num_seqs,
                            help='maximum number of sequences per iteration')
        parser.add_argument('--disable-log-stats',
                            action='store_true',
                            help='disable logging statistics')
        return parser

    # 根据命令行接口参数创建EngineArgs实例
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    # 根据EngineArgs实例的参数创建一组配置（ModelConfig、CacheConfig、
    # ParallelConfig和SchedulerConfig）并返回它们。
    def create_engine_configs(
        self,
    ) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig]:
        # Initialize the configs.
        # ModelConfig 包括了对 model 和 tokenizer 的定义，dtype 和随机数 seed
        # 以及是否用 pretrained weights 还是 dummy weights 等。
        model_config = ModelConfig(self.model, self.tokenizer,
                                   self.tokenizer_mode, self.trust_remote_code,
                                   self.download_dir, self.use_np_weights,
                                   self.use_dummy_weights, self.dtype,
                                   self.seed)
        # CacheConfig 包括 block_size（每个 block 多大）， gpu_utilization（GPU 利用率，
        # 后面 allocate 的时候占多少 GPU）和 swap_space（swap 的空间大小）。
        # 默认 block_size=16，swap_space=4GiB。
        cache_config = CacheConfig(self.block_size,
                                   self.gpu_memory_utilization,
                                   self.swap_space)
        # ParallelConfig 包括了 tensor_parallel_size 和 pipeline_parallel_size，
        # 即张量并行和流水线并行的 size
        parallel_config = ParallelConfig(self.pipeline_parallel_size,
                                         self.tensor_parallel_size,
                                         self.worker_use_ray)
        # SchdulerConfig 包括了 max_num_batched_tokens（一个 iteration 最多处理多少个
        # tokens），max_num_seqs（一个 iteration 最多能处理多少数量的 sequences）
        # 以及 max_seq_len（最大生成多长的 context length，也就是一个 sequence 的最长长度，
        # 包含 prompt 部分和 generated 部分）。
        scheduler_config = SchedulerConfig(self.max_num_batched_tokens,
                                           self.max_num_seqs,
                                           model_config.get_max_model_len())
        return model_config, cache_config, parallel_config, scheduler_config


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""
    engine_use_ray: bool = False
    disable_log_requests: bool = False

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument('--engine-use-ray',
                            action='store_true',
                            help='use Ray to start the LLM engine in a '
                            'separate process as the server process.')
        parser.add_argument('--disable-log-requests',
                            action='store_true',
                            help='disable logging requests')
        return parser
