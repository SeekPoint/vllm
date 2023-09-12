"""Sequence and its related classes."""
import copy
import enum
from typing import Dict, List, Optional, Union

from vllm.block import LogicalTokenBlock
from vllm.sampling_params import SamplingParams

'''
0x4. Sequence相关的数据结构解析
回顾一下，在原始的LLM类的genarete函数中，对于每个输入的prompt，都会给 llm engine 生成一个 request并添加到scheduler里。
然后调用 _run_engine 函数，这个函数的逻辑是对于所有未完成的 requests，
就调用 llm engine 的 step 函数得到这一步的 outputs，然后 append 到返回的 List 里。在step函数里，
由scheduler获取本次要作为输入的 seq_group_metadata_list ，同时产生一个 scheduler_outputs。
然后 engine 会调用 worker 的 execute_model 来执行对 seq_group_metadata_list 的模型前向计算。

这里的对于每个输入prompt，都会给llm engine生成一个request对应了llm_engine.add_request函数：

def add_request(
        self,
....

可以看到对于每个request vllm都会生成sampling_params.best_of个输出序列，在使用 beam search 时候这个参数也作为 beam width。
这里有2个数据结构Sequence和SequenceGroup，对于每一个request我们都会构造sampling_params.best_of个Sequence，
然后这些Sequence组成一个SequenceGroup。
在构造 Sequence 的时候会给这个 Sequence 分配相应的 logical_token_blocks（是一个List[LogicalTokenBlock]）。
一个 LogicalTokenBlock对象对应 block_size 个 token ids。
初始化的时候会把 token ids 分段存储在 logical blocks里面，但最后一个logical block可能是存不满的。
SequenceGroup构造好之后会给到Scheduler（self.scheduler.add_seq_group(seq_group)），
然后在llm_engine.step函数中seq_group_metadata_list, 
scheduler_outputs = self.scheduler.schedule()这行代码会把schdule出来的序列包装成SequenceGroupMetadata。
接下来，在执行worker的execute_model函数时会通过_prepare_inputs转成 tokens_tensor，position_tensor 和 InputMetadata，
然后execute_model函数实际上包含模型的前向推理和Sample过程，它的返回值数据集结构是Dict[int, SequenceOutputs]，
也就是 seq id -> 对应的输出，输出包含着 log prob。 
这个输出会被传到 scheduler 的 update 接口，用于更新对应的 running sequences。
最后这个Dict[int, SequenceOutputs]数据结构对象还会经过_decode_sequences并最终被包装成RequestOutput返回出来。
接下来对Sequnence涉及到的数据结构进行代码解析：


对于一个Sequence对象来说，它会被加到一个或者多个LogicalTokenBlock中，
我们这里也简单解析一下LogicalTokenBlock和PhysicalTokenBlock的定义代码。
'''
# 这行代码定义了一个新的枚举类SequenceStatus，它继承自Python的内置enum.Enum类。
class SequenceStatus(enum.Enum):
    """Status of a sequence."""
    # 这些都是序列可能的状态。enum.auto()是一个方便的方法，用于自动分配枚举值。
    WAITING = enum.auto()  # 序列正在等待运行。
    RUNNING = enum.auto()  # 序列正在运行。
    SWAPPED = enum.auto()  # 序列已被交换
    FINISHED_STOPPED = enum.auto()  # 序列已经完成，并且是因为遇到停止标志而停止的。
    FINISHED_LENGTH_CAPPED = enum.auto()  # 序列已经完成，并且是因为达到最大长度限制而停止的。
    FINISHED_ABORTED = enum.auto()  # 序列已经完成，并且是因为某种异常或错误而中止的。
    FINISHED_IGNORED = enum.auto()  # 序列已经完成，但是被忽略了。

    # 这是一个静态方法，接受一个序列状态作为输入，并返回该序列是否已经完成。
    # 它通过检查状态是否在完成的状态列表中来做出决策。
    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
        ]

    # 这也是一个静态方法，它返回一个序列完成的原因。如果序列没有完成，它返回None。
    # 它使用一个简单的条件语句来决定完成的原因。
    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


# SequenceData类代表与序列关联的数据。
class SequenceData:
    """Data associated with a sequence.


    Args:
        prompt_token_ids: The token IDs of the prompt.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """

    # 在初始化方法中，对象接收提示的token IDs（prompt_token_ids）作为参数，并初始化了
    # 两个其他属性：输出的token IDs（output_token_ids）和累积的对数概率（cumulative_logprob）。
    def __init__(
            self,
            prompt_token_ids: List[int],
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.cumulative_logprob = 0.0

    # 此方法接收一个token ID和其对应的对数概率，然后将token添加到output_token_ids列表，并累积对数概率。
    def append_token_id(self, token_id: int, logprob: float) -> None:
        self.output_token_ids.append(token_id)
        self.cumulative_logprob += logprob

    # 这个方法返回整个序列（提示+输出）的长度。
    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    # 此方法仅返回输出的长度。
    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    # 该方法返回整个token ID序列，包括提示和输出。
    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    # 此方法返回整个序列的最后一个token ID。如果没有输出，它会返回提示的最后一个token。
    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    # 此方法提供了SequenceData对象的字符串表示形式，有助于调试和可读性。
    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob})")


# 这个类Sequence存储了一个序列的数据、状态和块信息。
class Sequence:
    """Stores the data, status, and block information of a sequence.

    Args:
        seq_id: The ID of the sequence.
        prompt: The prompt of the sequence.
        prompt_token_ids: The token IDs of the prompt.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
    """

    # 对象初始化时需要序列ID、prompt、prompt_token_ids和块大小作为参数。
    def __init__(
            self,
            seq_id: int,
            prompt: str,
            prompt_token_ids: List[int],
            block_size: int,
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.block_size = block_size

        # 它还为序列设置了其他初始属性，如数据（使用SequenceData类）、
        # 输出的对数概率、输出的token和输出文本。
        self.data = SequenceData(prompt_token_ids)
        self.output_logprobs: List[Dict[int, float]] = []
        self.output_tokens: List[str] = []
        self.output_text = ""

        self.logical_token_blocks: List[LogicalTokenBlock] = []
        # Initialize the logical token blocks with the prompt token ids.
        self._append_tokens_to_blocks(prompt_token_ids)
        self.status = SequenceStatus.WAITING

    # 为logical_token_blocks列表添加新的LogicalTokenBlock。
    def _append_logical_block(self) -> None:
        block = LogicalTokenBlock(
            block_number=len(self.logical_token_blocks),
            block_size=self.block_size,
        )
        self.logical_token_blocks.append(block)

    # 这个方法的目的是将给定的token_ids逐个添加到逻辑token块中。
    def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
        # 设置一个cursor变量为0，它用于迭代token_ids列表。
        cursor = 0
        while cursor < len(token_ids):
            # 首先检查是否已经存在任何逻辑token块。如果没有，
            # 则调用_append_logical_block方法添加一个新块。
            if not self.logical_token_blocks:
                self._append_logical_block()

            # 获取logical_token_blocks的最后一个块。
            last_block = self.logical_token_blocks[-1]
            # 如果这个块已经满了（没有空位放置新tokens），那么就添加一个新块。
            if last_block.is_full():
                self._append_logical_block()
                last_block = self.logical_token_blocks[-1]

            # 计算上一个块中的空槽数量。
            num_empty_slots = last_block.get_num_empty_slots()
            # 添加尽可能多的token_ids到最后一个块中，不超过块的容量。
            last_block.append_tokens(token_ids[cursor:cursor +
                                                      num_empty_slots])
            # 更新cursor，使其指向下一个尚未处理的token。
            cursor += num_empty_slots

    # 将一个token ID及其对应的对数概率添加到序列中：
    def append_token_id(
            self,
            token_id: int,
            logprobs: Dict[int, float],
    ) -> None:
        # 断言确保给定的token_id存在于logprobs字典中。
        assert token_id in logprobs
        # 断言确保给定的token_id存在于logprobs字典中。
        self._append_tokens_to_blocks([token_id])
        # 将logprobs添加到output_logprobs列表。
        self.output_logprobs.append(logprobs)
        # 更新data对象，添加token ID和其对数概率。
        self.data.append_token_id(token_id, logprobs[token_id])

    def get_len(self) -> int:  # 返回整个序列的长度。
        return self.data.get_len()

    def get_output_len(self) -> int:  # 返回序列输出的长度。
        return self.data.get_output_len()

    def get_token_ids(self) -> List[int]:  # 返回整个token ID序列。
        return self.data.get_token_ids()

    def get_last_token_id(self) -> int:  # 返回整个序列的最后一个token ID。
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> List[int]:  # 返回序列输出的token IDs。
        return self.data.output_token_ids

    def get_cumulative_logprob(self) -> float:  # 返回输出的累积对数概率。
        return self.data.cumulative_logprob

    def is_finished(self) -> bool:  # 检查该序列是否已完成。
        return SequenceStatus.is_finished(self.status)

    # 这个方法用于创建一个当前序列的复制，并将其设置到另一个序列对象中。
    # 它会深度复制当前序列的逻辑token块、输出对数概率和数据。
    def fork(self, child_seq: "Sequence") -> None:
        child_seq.logical_token_blocks = copy.deepcopy(
            self.logical_token_blocks)
        child_seq.output_logprobs = copy.deepcopy(self.output_logprobs)
        child_seq.data = copy.deepcopy(self.data)

    # 提供了Sequence对象的字符串表示形式，用于调试和可读性。
    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"status={self.status.name}, "
                f"num_blocks={len(self.logical_token_blocks)})")


# 这是一个名为SequenceGroup的类，表示从相同提示生成的一组序列。
class SequenceGroup:
    """A group of sequences that are generated from the same prompt.
    SequenceGroup表示从同一提示生成的序列集合。它有以下属性：

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
	request_id: 请求的ID。
        seqs: 序列列表。
        sampling_params: 用于生成输出的采样参数。
        arrival_time: 请求的到达时间。
    """

    # 初始化方法设置了上述描述的属性。当创建一个SequenceGroup对象时，这些属性必须由用户提供。
    def __init__(
            self,
            request_id: str,
            seqs: List[Sequence],
            sampling_params: SamplingParams,
            arrival_time: float,
    ) -> None:
        self.request_id = request_id
        self.seqs = seqs
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time

    # 根据给定的状态返回序列列表。
    def get_seqs(
            self,
            status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        # 如果没有提供状态（默认为None），则返回所有序列。
        if status is None:
            return self.seqs
        # 否则，返回具有给定状态的序列。
        else:
            return [seq for seq in self.seqs if seq.status == status]

    # 返回具有给定状态的序列数量。如果没有提供状态，则返回所有序列的数量。
    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        return len(self.get_seqs(status))

    # 在序列组中查找具有给定ID的序列。
    def find(self, seq_id: int) -> Sequence:
        for seq in self.seqs:
            if seq.seq_id == seq_id:
                return seq
        raise ValueError(f"Sequence {seq_id} not found.")

    # 检查序列组中的所有序列是否都已完成。
    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.seqs)

    # 返回SequenceGroup对象的字符串表示形式，通常用于调试和可读性。
    # 这将提供关于请求ID、采样参数和序列数量的简要信息。
    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs)})")


# 这是一个名为SequenceGroupMetadata的类，它表示序列组的元数据。
# 这个类的主要目的是为了保存与特定SequenceGroup关联的元数据，可以用来创建InputMetadata。
class SequenceGroupMetadata:
    """Metadata for a sequence group. Used to create `InputMetadata`.


    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. (Seq id -> list of physical block
            numbers)
    """

    # 初始化方法会接收上述属性作为参数，并将它们设置为类的属性。
    # 这意味着当你创建一个SequenceGroupMetadata对象时，必须提供这些参数。
    def __init__(
            self,
            request_id: str,
            is_prompt: bool,
            seq_data: Dict[int, SequenceData],
            sampling_params: SamplingParams,
            block_tables: Dict[int, List[int]],
    ) -> None:
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables


# 这是一个名为SequenceOutputs的类，代表与一个序列关联的模型输出。
class SequenceOutputs:
    """The model output associated with a sequence.

    Args:
        seq_id: The ID of the sequence.
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """

    def __init__(
            self,
            seq_id: int,  # 表示序列的ID。
            # 表示父序列的ID。在进行如束搜索(beam search)这样的算法时，
            # 可能会"分叉"或"分裂"序列，此时，新序列将具有一个"父序列"。
            parent_seq_id: int,
            output_token: int,
            logprobs: Dict[int, float],
    ) -> None:
        self.seq_id = seq_id
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token  # 输出的token ID。
        # 这是一个字典，表示输出token的对数概率。键是token的ID，
        # 值是给定先前所有token后，下一个token是当前token的对数概率。
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f"SequenceOutputs(seq_id={self.seq_id}, "
                f"parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}), "
                f"logprobs={self.logprobs}")

    # 这是一个特殊的方法，允许对象进行等值比较。它定义了当两个SequenceOutputs对象
    # 是否应该被认为是"相等"的条件。
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutputs):
            return NotImplemented
        return (self.seq_id == other.seq_id
                and self.parent_seq_id == other.parent_seq_id
                and self.output_token == other.output_token
                and self.logprobs == other.logprobs)
