"""A block manager that manages token blocks."""
from typing import Dict, List, Optional, Set, Tuple

from vllm.block import PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device


# 这段代码定义了一个 BlockAllocator 类，它管理设备上的PhysicalTokenBlock的分配和释放
class BlockAllocator:
    """这个类维护了一个空闲块列表，并在请求时分配一个块。当块被释放时，它的引用计数会减少。
    当引用计数变为零时，该块会被重新添加到空闲列表中
    Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    # 这是类的初始化方法，当创建 BlockAllocator 的实例时，它会被调用。
    # 方法接受三个参数：设备 (device)、块大小 (block_size) 和块数 (num_blocks)。
    # 这三个参数分别保存在类的属性中。
    def __init__(
            self,
            device: Device,
            block_size: int,
            num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        # 这段代码初始化了空闲块列表。它为每个块创建了一个 PhysicalTokenBlock 实例，
        # 并添加到 free_blocks 列表中。
        self.free_blocks: List[PhysicalTokenBlock] = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size)
            self.free_blocks.append(block)

    # 这个方法用于分配一个空闲的PhysicalTokenBlock。如果没有可用的空闲块，
    # 它会引发一个值错误。分配的块的引用计数被设置为1。
    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    # 这个方法释放一个先前分配的块。如果块的引用计数已经是0，它会引发一个值错误，
    # 因为这意味着块已经被释放。块的引用计数减少，如果引用计数变为零，块被重新添加到空闲列表中。
    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    # 这个方法返回当前可用的空闲块的数量。
    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]


# BlockTable 是一个类型别名，表示的是从逻辑块号映射到物理块的列表。
class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    # 这是类的初始化方法，它接受四个参数：块大小 (block_size)、GPU上的块数 (num_gpu_blocks)、
    # CPU上的块数 (num_cpu_blocks) 和一个水印值 (watermark)。
    # 水印是一个比例，用于确定当GPU块数量低于这个水印值时可能需要采取的措施。
    def __init__(
            self,
            block_size: int,
            num_gpu_blocks: int,
            num_cpu_blocks: int,
            watermark: float = 0.01,
    ) -> None:
        # 这段代码初始化了类的属性，并确保提供的水位值是非负的。
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.watermark = watermark
        assert watermark >= 0.0

        # 首先，基于给定的水位值和GPU上的块数，计算出一个整数值的水位块数。
        # 然后，为GPU和CPU初始化块分配器。
        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size,
                                            num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size,
                                            num_cpu_blocks)
        # Mapping: seq_id -> BlockTable.
        # 这里初始化了一个块映射表，它是一个字典，其键是序列ID (seq_id)，
        # 值是上面定义的 BlockTable 类型。这个字典将帮助我们跟踪每个序列ID与哪些物理块相关联。
        self.block_tables: Dict[int, BlockTable] = {}

    # can_allocate 函数接受一个参数 seq_group（类型为 SequenceGroup）并返回一个布尔值。
    def can_allocate(self, seq_group: SequenceGroup) -> bool:
        # 此注释标识了一个潜在的问题或假设。该注释提到在当前实现中，我们假设在 seq_group
        # 中的所有序列共享相同的提示（prompt）。但是，这个假设可能对于被preempted的序列并不成立。
        # 由于这是一个 FIXME 注释，它意味着这个问题在未来需要被修复或重新审视。
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        # 这里，代码首先从 seq_group 中获取序列列表并选择第一个序列
        # （假设在该组中的所有序列都有相同数量的块）。然后，计算该序列所需的逻辑块数量。
        seq = seq_group.get_seqs()[0]
        num_required_blocks = len(seq.logical_token_blocks)
        # 该行代码调用 gpu_allocator（即GPU块分配器）的 get_num_free_blocks 方法，
        # 来得到当前可用的空闲GPU块数量。
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        # Use watermark to avoid frequent cache eviction.
        # 此行代码检查当前空闲的GPU块数量是否足以满足序列的需求，
        # 并且剩余的块数量是否大于或等于水位块数量（watermark_blocks）。
        # 使用水位是为了避免频繁地清除缓存，确保始终有一定数量的块保持空闲。
        # 如果满足这些条件，函数返回 True，表示可以为 seq_group 分配所需的块；否则，返回 False。
        return (num_free_gpu_blocks - num_required_blocks >=
                self.watermark_blocks)

    # 这段代码定义了BlockSpaceManager类中的allocate成员函数，
    # 该函数用于为给定的seq_group分配PhysicalTokenBlock
    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        # 此行从seq_group中选择了第一个序列。由于之前的注释提到所有的序列都有相同的提示，
        # 因此我们可以通过查看第一个序列来确定需要的块数量。
        seq = seq_group.get_seqs()[0]

        # Allocate new physical token blocks that will store the prompt tokens.
        # 首先，初始化一个空的block_table，用于存储分配给序列的PhysicalTokenBlock。
        # 然后，根据第一个序列的logical_token_blocks的数量进行循环，为每个logical_token_blocks分配一个PhysicalTokenBlock。
        # PhysicalTokenBlock的引用计数被设置为序列组中的序列数量。这可能意味着每个
        # PhysicalTokenBlock可以被多个序列共享。每分配一个PhysicalTokenBlock，
        # 就将其添加到block_table中。
        block_table: BlockTable = []
        for _ in range(len(seq.logical_token_blocks)):
            block = self.gpu_allocator.allocate()
            # Set the reference counts of the token blocks.
            block.ref_count = seq_group.num_seqs()
            block_table.append(block)

        # Assign the block table for each sequence.
        # 对于seq_group中的每个序列，将block_table复制并分配给该序列。
        # 这意味着，尽管每个序列都有自己的块表副本，但它们都引用同一组PhysicalTokenBlock。
        for seq in seq_group.get_seqs():
            self.block_tables[seq.seq_id] = block_table.copy()

    # 此函数是BlockSpaceManager类的一个成员函数，名为can_append_slot。
    # 它决定是否可以为给定的seq_group追加一个新的token块（slot）。
    # 函数的逻辑是基于一个简单的启发式方法：如果有足够的自由块（free blocks）
    # 来满足序列组中每个序列的需求，那么就可以追加。
    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        # 此行从gpu_allocator获取GPU上当前的空闲块数量，并将其存储在num_free_gpu_blocks变量中。
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        # 使用num_seqs方法，函数从seq_group获取当前状态为RUNNING的序列数量。
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        # 函数返回一个布尔值，这个布尔值是基于一个简单的判断：
        # 如果当前RUNNING状态的序列数量num_seqs小于或等于GPU上的空闲块数量num_free_gpu_blocks，
        # 则返回True（表示可以追加新的slot），否则返回False。
        return num_seqs <= num_free_gpu_blocks

    # 这段代码定义了BlockSpaceManager类中的append_slot函数，
    # 该函数的主要目标是为一个新的token分配一个物理slot。
    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        # 从输入的序列seq中提取逻辑块，并从块表中获取与该序列ID关联的块表。
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        # 如果块表的长度小于逻辑块的数量，这意味着序列有一个新的逻辑块。
        # 因此，分配一个新的物理块并将其添加到块表中。
        if len(block_table) < len(logical_blocks):
            # The sequence has a new logical block.
            # Allocate a new physical block.
            block = self.gpu_allocator.allocate()
            block_table.append(block)
            return None

        # We want to append the token to the last physical block.
        # 获取块表中的最后一个块，并确保它在GPU设备上。
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        # 如果最后一个块的引用计数为1，这意味着它不与其他序列共享，因此可以直接追加token。
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            # 如果最后一个块与其他序列共享，我们将使用“Copy on Write”策略：
            # 为新的token分配一个新的块，并复制旧块中的tokens。
            # 然后，释放旧块，并在块表中更新块的引用。
            # 函数返回旧块和新块的块编号，以便外部调用者知道哪些块已更改。
            new_block = self.gpu_allocator.allocate()
            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return last_block.block_number, new_block.block_number

    # 这段代码定义了BlockSpaceManager类中的fork方法，该方法的目的是为子序列创建一个块表，
    # 该表是基于其父序列的块表的副本。此函数确保每个物理块的引用计数在分叉时正确地增加，
    # 这是因为两个序列（父序列和子序列）现在共享相同的物理块。
    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        # 从块表映射中，根据父序列的ID获取其块表。
        src_block_table = self.block_tables[parent_seq.seq_id]
        # 为子序列创建一个新的块表，这个块表是父序列块表的一个副本。这
        # 意味着父序列和子序列现在都引用相同的物理块，但是它们有各自独立的逻辑块。
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        # 由于子序列现在也引用相同的物理块，所以需要增加每个物理块的引用计数。
        # 这确保了物理块不会在它仍然被使用时被意外释放。
        for block in src_block_table:
            block.ref_count += 1

    # 这段代码定义了BlockSpaceManager类中的一个私有方法_get_physical_blocks，
    # 它的目的是获取一个SequenceGroup内部所有序列所使用的所有物理块，而没有重复。
    # 这个方法接受一个SequenceGroup对象作为参数，并返回一个PhysicalTokenBlock对象的列表。
    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        # 这个注释提到一个关键的假设：物理块只能在同一组内的序列之间共享。
        # 这意味着不同组之间的序列不会共享相同的物理块。
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        # 这里使用了一个集合（set）来存储物理块，因为集合不允许有重复的元素。
        # 这样可以确保，即使多个序列引用同一个物理块，该块也只会在集合中出现一次。
        blocks: Set[PhysicalTokenBlock] = set()
        # 首先，对序列组内的每个序列进行遍历。
        for seq in seq_group.get_seqs():
            # 使用seq.is_finished()来检查该序列是否已完成。如果已完成，那么我们跳过这个序列，继续下一个。
            if seq.is_finished():
                continue
            # 如果该序列没有完成，我们从self.block_tables中获取与该序列ID关联的块表。
            block_table = self.block_tables[seq.seq_id]
            # 遍历这个块表中的每个物理块，并将其添加到blocks集合中。
            # 由于使用了集合，重复的块不会被多次添加。
            for block in block_table:
                blocks.add(block)
        return list(blocks)

    # 这段代码定义了BlockSpaceManager类中的一个方法can_swap_in，
    # 这个方法的目的是确定给定的SequenceGroup是否可以被交换到GPU中。
    # 该方法接受一个SequenceGroup对象作为参数，并返回一个布尔值，
    # 表示该SequenceGroup是否可以被交换到GPU中。
    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        # 使用之前定义的_get_physical_blocks方法来获取seq_group中所有序列所使用的物理块的列表。
        blocks = self._get_physical_blocks(seq_group)
        # 计算seq_group中状态为SWAPPED的序列的数量。
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        # 使用gpu_allocator来获取当前可用于GPU的空闲块的数量。
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        # 这里的逻辑有两个关键的假设：
        # 1. 每一个被交换进来的序列都会在交换后立即分配至少一个空闲块。
        # 2. 这种分配逻辑应该与can_append_slot()中的逻辑匹配。
        # 因此，为了确定我们是否有足够的块来满足这个要求，我们需要将当前物理块的数量
        # 与SWAPPED状态的序列的数量相加
        num_required_blocks = len(blocks) + num_swapped_seqs
        # 最终的决策是基于空闲块的数量是否超过所需块的数量加上一个阈值self.watermark_blocks。
        # 如果是，则返回True，表示该SequenceGroup可以被交换到GPU中；否则，返回False。
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    # 这个函数是BlockSpaceManager类的swap_in方法，其作用是将一个序列组(seq_group)从CPU交换到GPU，
    # 并为此过程中涉及的每个CPU块与GPU块创建映射关系。
    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # CPU block -> GPU block.
        # 这里初始化一个字典来记录从CPU块到GPU块的映射关系。
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        # 遍历SequenceGroup中的每一个序列。如果这个序列已经完成，我们就跳过它。
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            # 这里我们为当前的序列初始化一个新的GPU块表。block_table是原来与这个序列关联的CPU块列表。
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            # 对于每个CPU块：
            for cpu_block in block_table:
                # 我们首先检查它是否已经有一个关联的GPU块(通过mapping字典)。
                # 如果有，我们简单地增加这个GPU块的引用计数。
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    # 如果没有，则为该CPU块分配一个新的GPU块，并在mapping字典中记录这种关联。
                    gpu_block = self.gpu_allocator.allocate()
                    mapping[cpu_block] = gpu_block
                # 把新的GPU块添加到new_block_table列表中。
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                # 释放原来的CPU块，因为我们现在在GPU上有了一个拷贝。
                self.cpu_allocator.free(cpu_block)
            # 我们使用新的GPU块列表更新当前序列的块映射。
            self.block_tables[seq.seq_id] = new_block_table

        # 在这个方法的最后，我们创建并返回一个新的映射，这个映射使用块号
        # （而不是块对象）来表示从CPU块到GPU块的关联。这样的表示方式可能更加方便和简洁。
        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    # 这个方法是BlockSpaceManager类中的一个成员函数，其名称为can_swap_out。
    # 其作用是判断是否可以将指定的seq_group（序列组）从GPU交换出到CPU。
    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        # 首先，使用内部方法_get_physical_blocks获取序列组seq_group当前在GPU上的所有物理块。
        blocks = self._get_physical_blocks(seq_group)
        # 这里检查序列组中的物理块数量是否小于或等于CPU上的空闲块数量。
        # 这确保我们有足够的空间在CPU上容纳所有要交换的块。
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    # 这个swap_out方法是BlockSpaceManager类的成员函数。其核心功能是从GPU将特定的seq_group
    # （序列组）交换出到CPU。在此过程中，方法还返回一个字典，
    # 说明GPU上的块号与交换到CPU上的块号之间的映射关系。
    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block.
        # 这是一个映射字典，其键是在GPU上的块，值是在CPU上的块。
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        # 为了交换整个序列组，我们遍历其中的每一个序列。
        for seq in seq_group.get_seqs():
            # 如果当前序列已经完成，则跳过该序列，继续处理下一个。
            if seq.is_finished():
                continue
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            # 我们现在开始处理每一个在GPU上的块。
            for gpu_block in block_table:
                # 如果当前GPU块已经有一个映射到CPU的块，我们增加该CPU块的引用计数。
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    # 如果当前GPU块不在映射中，我们在CPU上为其分配一个新块，
                    # 并将此映射关系添加到mapping字典中。
                    cpu_block = self.cpu_allocator.allocate()
                    mapping[gpu_block] = cpu_block
                # 我们将新分配或已映射的CPU块添加到new_block_table中。
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                # 当一个GPU块被成功交换到CPU上后，我们释放该GPU块，使其可以被其他数据使用。
                self.gpu_allocator.free(gpu_block)
            # 我们用新的new_block_table更新当前序列的块映射。
            self.block_tables[seq.seq_id] = new_block_table

        # 最后，我们创建一个块号的映射字典，其键是GPU块号，值是CPU块号。
        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping

    # 这是一个私有方法（由下划线_开始，通常表示这个方法是内部方法，不应该在类的外部直接调用），
    # 名为_free_block_table，它的主要任务是释放提供的块表（block_table）中的块。
    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in block_table:
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    # 它的主要任务是释放与指定seq（一个Sequence对象）相关的资源
    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    # 这个方法的目的是重置BlockSpaceManager的状态，释放所有与之相关的块，并清空block_tables字典。
    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    # 这个方法的目的是根据给定的seq（一个Sequence对象），返回与该序列关联的块表中所有块的块编号。
    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    # 这个方法返回当前可用的GPU块的数量。
    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    # 这个方法返回当前可用的CPU块的数量。
    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()

'''
vllm/vllm/core/目录下的policy.py和block_manager.py这两个文件为Scheduler定义了队列优先出队的规则即时间优先，
换句话说就是先到达的请求先出队被推理。

然后BlockSpaceManager定义了Paged Attention中对一个SequenceGroup来说物理块是怎么申请的，
以及逻辑块和物理块是怎么映射的，最后还实现了swap_in，swap_out 等在CPU/GPU移动物理块的函数。
'''