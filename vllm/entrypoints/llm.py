from typing import List, Optional, Union

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter


class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        seed: The seed to initialize the random number generator for sampling.
    """
    """这是一个名为LLM（语言模型）的Python类，这个类用于从给定的提示和采样参数生成文本。
    类的主要部分包括tokenizer（用于将输入文本分词）、语言模型（可能分布在多个GPU上执行）
    以及为中间状态分配的GPU内存空间（也被称为KV缓存）。给定一批提示和采样参数，
    该类将使用智能批处理机制和高效的内存管理从模型中生成文本。

    这个类设计用于离线推理。在线服务的话，应使用AsyncLLMEngine类。
    对于参数列表，可以参见EngineArgs。

    Args:
        model: HuggingFace Transformers模型的名称或路径.
        tokenizer: HuggingFace Transformers分词器的名称或路径。默认为None。.
        tokenizer_mode: 分词器模式。"auto"将使用快速分词器（如果可用），
        "slow"将总是使用慢速分词器。默认为"auto"。.
        trust_remote_code: 当下载模型和分词器时，是否信任远程代码
        （例如，来自HuggingFace的代码）。默认为False。
        tensor_parallel_size: 用于分布式执行的GPU数量，使用张量并行性。默认为1。
        dtype: 模型权重和激活的数据类型。目前，我们支持float32、float16和bfloat16。
        如果是auto，我们使用在模型配置文件中指定的torch_dtype属性。
        但是，如果配置中的torch_dtype是float32，我们将使用float16。默认为"auto"。
        seed: 初始化采样的随机数生成器的种子。默认为0。
        
    可以看到LLM类似于对LLMEngine进行了封装，一个LLM对象对应了一个LLMEngine对象    
    """
    def __init__(
            self,
            model: str,
            tokenizer: Optional[str] = None,
            tokenizer_mode: str = "auto",
            trust_remote_code: bool = False,
            tensor_parallel_size: int = 1,
            dtype: str = "auto",
            seed: int = 0,
            **kwargs, # 其它关键字参数。
    ) -> None:
        # 在初始化函数中，首先检查kwargs中是否包含"disable_log_stats"键，
        # 如果没有，则在kwargs中添加该键并设置其值为True。
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        # 使用所有给定的参数（包括通过kwargs传递的任何额外参数）来初始化EngineArgs对象，
        # 然后使用这些参数来初始化LLMEngine对象
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            seed=seed,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        # 初始化一个名为request_counter的Counter对象，用于请求计数。
        self.request_counter = Counter()
        
# 个方法返回存储在类的 llm_engine 属性中的分词器对象。返回类型是 Union[PreTrainedTokenizer,
# PreTrainedTokenizerFast]，表示可以返回这两种类型的任何一个。
def get_tokenizer(
        self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    return self.llm_engine.tokenizer

# 这个方法接收一个参数 tokenizer，类型为 Union[PreTrainedTokenizer,
# PreTrainedTokenizerFast]。这个参数是新的分词器对象，将替换现有的分词器。
def set_tokenizer(
    self,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
) -> None:
    self.llm_engine.tokenizer = tokenizer

# generate 函数是用于根据给定的prompts生成完整文本的核心方法。
    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

    Args:
        prompts: A list of prompts to generate completions for.
        sampling_params: The sampling parameters for text generation. If
            None, we use the default sampling parameters.
        prompt_token_ids: A list of token IDs for the prompts. If None, we
            use the tokenizer to convert the prompts to token IDs.
        use_tqdm: Whether to use tqdm to display the progress bar.

    Returns:
        A list of `RequestOutput` objects containing the generated
        completions in the same order as the input prompts.
    """
    # 这段代码确保至少提供了prompts或prompts的token ID之一。
    if prompts is None and prompt_token_ids is None:
        raise ValueError("Either prompts or prompt_token_ids must be "
                         "provided.")
    # 如果只提供了一个字符串prompts（而不是列表），这段代码将其转换为列表，以便后续处理。
    if isinstance(prompts, str):
        # Convert a single prompt to a list.
        prompts = [prompts]
    # 如果同时提供了prompts和prompts token ID，则此代码确保它们的长度相同。
    if prompts is not None and prompt_token_ids is not None:
        if len(prompts) != len(prompt_token_ids):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")
    # 如果未提供采样参数，此代码将使用默认参数。
    if sampling_params is None:
        # Use default sampling params.
        sampling_params = SamplingParams()

    # Add requests to the engine.
    # 此段代码循环遍历prompts或prompt token IDs，并使用它们调用 _add_request 方法
    # 将请求添加到引擎。根据是否提供了prompts或token ID，适当地处理了参数。
    if prompts is not None:
        num_requests = len(prompts)
    else:
        num_requests = len(prompt_token_ids)
    for i in range(num_requests):
        prompt = prompts[i] if prompts is not None else None
        if prompt_token_ids is None:
            token_ids = None
        else:
            token_ids = prompt_token_ids[i]
        self._add_request(prompt, sampling_params, token_ids)
    # 此代码调用先前定义的 _run_engine 方法来运行引擎，并返回其输出。
    # 这些输出是一个RequestOutput对象的列表，包含生成的完整文本，与输入prompt的顺序相同。
    return self._run_engine(use_tqdm)

# LLM模型为 llm_engine 添加一个请求
def _add_request(
    self,
    prompt: Optional[str],
    sampling_params: SamplingParams,
    prompt_token_ids: Optional[List[int]],
) -> None:
    # 从 self.request_counter 获取下一个值并转换为字符串来创建请求ID。
    request_id = str(next(self.request_counter))
    # 调用 llm_engine 的 add_request 方法，将请求添加到llm_engine中。
    self.llm_engine.add_request(request_id, prompt, sampling_params,
                                prompt_token_ids)

# 这个函数负责运行 self.llm_engine的step函数，并收集已完成的请求的输出。
def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
    # Initialize tqdm.
    # 如果参数 use_tqdm 为真，则代码初始化一个 tqdm 进度条来跟踪处理进度。
    # tqdm 是一个流行的库，用于在命令行中显示循环的进度条。num_requests 是尚未完成的请求的数量。
    if use_tqdm:
        num_requests = self.llm_engine.get_num_unfinished_requests()
        pbar = tqdm(total=num_requests, desc="Processed prompts")
    # Run the engine.
    outputs: List[RequestOutput] = []
    # 主要的循环在引擎有未完成的请求时持续运行。在每个步骤中，通过调用引擎的 step 方法来处理请求。
    # 如果输出表示已完成的请求，则将其添加到 outputs 列表中。如果使用 tqdm，进度条将相应地更新。
    while self.llm_engine.has_unfinished_requests():
        step_outputs = self.llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                if use_tqdm:
                    pbar.update(1)
    # 如果使用了 tqdm，此代码段将关闭进度条。
    if use_tqdm:
        pbar.close()
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    # 输出列表是按完成顺序排列的，这可能与原始请求顺序不同。
    # 这一行代码通过请求ID将它们排序，确保输出按原始顺序排列。
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    return outputs
'''
现在基本走完了vllm根据prompt，特定模型架构和特定采样参数去生成结果的全流程，我们再对这个流程总结一下。

首先，vllm进来之后先实例化一个LLM对象即：llm = LLM(model="facebook/opt-125m")。
然后调用llm.generate函数，这个函数的输入是prompts（List[str]类型），采样参数，然后返回 List[RequestOutput]，
对应outputs = llm.generate(prompts, sampling_params)这行代码。

从llm.generate的实现来看，对于每一个prompt都会生成一个request喂给llm_engine，
然后执行_run_engine（这个函数负责运行 llm_engine.step函数，并收集已完成的请求的输出。）函数结束。
llm_engine.step函数首先从scheduler获取当前的输入seq_group_metadata_list ，同时生成一个 scheduler_outputs，
接下来会调用 workers 的 execute_model来指定模型的前向推理过程，
拿到这个结果之后再进行解码（对应self._decode_sequences(seq_groups)这行）。
最后scheduler再更新已经解码完毕的序列的状态，并释放序列占用的内存。

接下来，分别对vllm关键的几个组件scheduler，worker，cache engine进行解析。
'''