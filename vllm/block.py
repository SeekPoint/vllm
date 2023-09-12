"""Token blocks."""
from typing import List

from vllm.utils import Device

# 这定义了一个私有常数_BLANK_TOKEN_ID，其值为-1。此常数用于在逻辑块中表示空位置。
_BLANK_TOKEN_ID = -1


# 这个类表示了一个逻辑块，其设计用于在KV Cache中代表相应的物理块的状态。
class LogicalTokenBlock:
    """A block that stores a contiguous chunk of tokens from left to right.

    Logical blocks are used to represent the states of the corresponding
    physical blocks in the KV cache.
    """

    def __init__(
        self,
        block_number: int, # 表示此块的编号。
        block_size: int, # 块的大小，即块可以容纳的最大token数量。
    ) -> None:
        self.block_number = block_number
        self.block_size = block_size

        # 存储在这个块中的token的ID。初始化为_BLANK_TOKEN_ID的列表，长度为block_size。
        self.token_ids = [_BLANK_TOKEN_ID] * block_size
        # 块中当前token的数量。
        self.num_tokens = 0

    # 判断块是否为空。如果num_tokens为0，则块为空。
    def is_empty(self) -> bool:
        return self.num_tokens == 0

    # 获取块中的空闲位置数量。这通过块的总大小和当前的token数量计算得出。
    def get_num_empty_slots(self) -> int:
        return self.block_size - self.num_tokens

    # 判断块是否已满。如果num_tokens等于块的大小，则块已满。
    def is_full(self) -> bool:
        return self.num_tokens == self.block_size

    # 将token ID列表追加到块中。函数首先检查新添加的token数量是否超出了块的空闲空间，
    # 然后将新的token追加到块中，并更新num_tokens。
    def append_tokens(self, token_ids: List[int]) -> None:
        assert len(token_ids) <= self.get_num_empty_slots()
        curr_idx = self.num_tokens
        self.token_ids[curr_idx:curr_idx + len(token_ids)] = token_ids
        self.num_tokens += len(token_ids)

    # 获取存储在块中的token的ID列表。
    def get_token_ids(self) -> List[int]:
        return self.token_ids[:self.num_tokens]

    # 获取块中的最后一个token的ID。如果块是空的，则会引发断言错误。
    def get_last_token_id(self) -> int:
        assert self.num_tokens > 0
        return self.token_ids[self.num_tokens - 1]

# 这段代码定义了一个名为PhysicalTokenBlock的类。这个类代表KV Cache中一个块的状态。
class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
    ) -> None:
        self.device = device # 表示此块所在的设备，如CPU、GPU等。具体类型为Device，
        self.block_number = block_number # 这是块的编号，用于识别或索引它。
        self.block_size = block_size # 块的大小。这可能代表块可以容纳的数据量。

        self.ref_count = 0 # 引用计数，通常用于跟踪有多少其他对象或操作正在引用或使用此物理块

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'ref_count={self.ref_count})')

'''
这一节介绍的Sequence相关的数据结构和Block相关的数据结构本质上都是在为Scheduler服务，来一起完成Paged Attention的内存管理功能。

下一节我们对Scheduler也叫vllm的调度策略进行解析，它也包含了对在这里定义的Block的管理细节。
'''
