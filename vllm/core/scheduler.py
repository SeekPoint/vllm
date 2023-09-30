import enum
import time
from typing import Dict, List, Optional, Tuple

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.block_manager import BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceOutputs,
                           SequenceStatus)

logger = init_logger(__name__)


# 这定义了一个新的枚举类PreemptionMode，它继承自Python的内置enum.Enum类。
# PreemptionMode枚举为预占模式提供了一个明确的类型化表示，有两种可选的模式：SWAP和RECOMPUTE。
class PreemptionMode(enum.Enum):
    # 这是一段解释预占模式的文档字符串。它描述了两种不同的预占模式：
    # Swapping: 当序列被抢占时，将其块交换到CPU内存中，并在恢复序列时将它们再次交换回来。
    # Recomputation: 当序列被抢占时，丢弃其块，并在恢复序列时重新计算它们，将序列视为新的提示。
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


# 这段代码定义了一个名为SchedulerOutputs的类。该类似乎旨在为某种调度操作提供输出或结果。
class SchedulerOutputs:

    def __init__(
            self,
            scheduled_seq_groups: List[SequenceGroup],  # 被调度的序列组的列表。
            prompt_run: bool,  # 一个布尔值，可能表示是否根据给定的提示执行了某种运行。
            num_batched_tokens: int,  # 批处理的token数。
            blocks_to_swap_in: Dict[int, int],
            blocks_to_swap_out: Dict[int, int],
            blocks_to_copy: Dict[int, List[int]],
            ignored_seq_groups: List[SequenceGroup],  # 被忽略的序列组的列表。
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)


class Scheduler:

    def __init__(
            self,
            # scheduler_config: 调度器的配置，类型为 SchedulerConfig。
            scheduler_config: SchedulerConfig,
            # cache_config: 缓存的配置，类型为 CacheConfig。
            cache_config: CacheConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        # Instantiate the scheduling policy.
        # 使用 PolicyFactory 的 get_policy 方法为调度策略分配一个实例。
        # 这里固定选择了 "fcfs"（可能表示"先来先服务"）策略。
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        # 创建一个 BlockSpaceManager 的实例，该实例管理数据块的空间。
        # 它使用 cache_config 中的配置数据，包括块大小、GPU块数和CPU块数。
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
        )

        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []

    # 这个函数是 Scheduler 类的成员函数，用于将新的 SequenceGroup 添加到等待队列中。
    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    # 该函数是 Scheduler 类的成员函数，用于根据提供的 request_id 中止一个 SequenceGroup。
    def abort_seq_group(self, request_id: str) -> None:
        # 这是一个外层循环，它遍历三个队列：等待、运行和交换。
        # 这意味着它会检查所有的 SequenceGroup，无论它们处于哪种状态。
        for state_queue in [self.waiting, self.running, self.swapped]:
            # 这是一个内部循环，遍历当前状态队列中的每一个 SequenceGroup
            for seq_group in state_queue:
                if seq_group.request_id == request_id:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.seqs:
                        if seq.is_finished():
                            continue
                        self.free_seq(seq, SequenceStatus.FINISHED_ABORTED)
                    return

    # 如果三个队列中的任何一个非空，那么这意味着仍有未完成的序列，函数返回True，否则返回False。
    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped

    # 该方法返回三个队列（waiting, running, swapped）中的SequenceGroup的总数。
    # 它通过取每个队列的长度并将它们加在一起来做到这一点。
    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    #重点看_schedule函数，其主要功能是从waiting队列中找到符合token限制的请求，添加到running队列中，然后在scheduled中添加对应的seq_group。
    # 如果swapped队列不为空，则从running队列中找到最低优先级的seq,然后添加到preempted标记。
    # swapped队列类似，只不过只有当swapped_out队列为空时才开始调度，最后将调度好的seq放入到running队列中进行后续调度。

    # 这个函数是Scheduler类中的一个复杂的私有方法，它尝试安排SequenceGroup实例的执行，
    # 可能需要进行资源的分配、替换和拷贝。函数的主要目的是返回一个SchedulerOutputs对象，
    # 它包含了执行的相关信息。
    def _schedule(self) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        # 初始化几个字典，用于跟踪需要在模型执行前需要换入，换出，复制的块
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        # 获取当前时间，这可能会被用来决定哪些任务应该被优先调度。
        now = time.time()

        # Join waiting sequences if possible.
        # 检查是否有序列组被交换到CPU。如果没有，尝试合并等待中的序列。
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            num_batched_tokens = 0
            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            # 当等待队列不为空时，获取队列中的第一个序列组。
            while self.waiting:
                seq_group = self.waiting[0]

                # 获取当前序列组中的第一个序列的长度（tokens数量）。
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                # 计算允许的最大prompt长度。
                prompt_limit = min(
                    self.scheduler_config.max_model_len,
                    self.scheduler_config.max_num_batched_tokens)
                # 如果当前序列超过了上述限制，发出警告并将该序列组标记为被忽略。
                if num_prompt_tokens > prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    break

                # If the sequence group cannot be allocated, stop.
                # 检查是否有足够的块空间来为该序列组分配资源。
                if not self.block_manager.can_allocate(seq_group):
                    break

                # 这里检查已经批处理的token数量（num_batched_tokens）加上当前序列组
                # （seq_group）的token数量（num_prompt_tokens）是否超过了配置中的
                # 最大token限制。如果超过了，循环就会中断。
                if (num_batched_tokens + num_prompt_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # 这里获取等待状态下的序列数。
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.num_seqs(
                    status=SequenceStatus.WAITING)
                # 这里计算了当前正在运行状态的所有序列组中的序列数量。
                num_curr_seqs = sum(
                    seq_group.num_seqs(status=SequenceStatus.RUNNING)
                    for seq_group in self.running)
                # 检查当前正在运行的序列数量和新的序列数量的总和是否超过配置中的最大序列限制。
                # 如果超过了，循环就会中断。
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                # 从等待队列的前端移除并获取一个序列组。
                seq_group = self.waiting.pop(0)
                # 为从等待队列中获取的序列组分配资源。
                self._allocate(seq_group)
                # 将这个序列组添加到正在运行的队列中。
                self.running.append(seq_group)
                # 更新已批处理的token数量，加上当前序列组的token数量。
                num_batched_tokens += num_prompt_tokens
                # 将这个序列组添加到已调度的队列中。
                scheduled.append(seq_group)

            # 这里检查scheduled列表是否不为空。scheduled列表保存了在当前调度周期中被成功调度的序列组。
            if scheduled:
                # 这行开始创建一个SchedulerOutputs对象，并将以下参数传递给它：
                # 将被成功调度的序列组列表传递给scheduled_seq_groups参数。
                # 这是一个标识符，说明序列组是基于输入提示运行的。
                # 当前已批处理的token数量。
                # 需要从CPU内存中换入的块的映射。
                # 需要换出到CPU内存的块的映射。
                # 需要在GPU内存中复制的块的映射。
                # 由于某种原因（如输入提示太长）而被忽略的序列组列表。
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=num_batched_tokens,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs

        # 这段代码关注在没有足够的空闲插槽可用以保持所有序列组处于RUNNING状态时的抢占策略。
        # 它包括了哪些序列组应该被抢占，以及如何为当前运行的序列组分配新的token插槽。
        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        # 这是一个注释，解释了接下来的代码部分。当没有足够的插槽来保持所有序列组处于RUNNING状态时，
        # 就会发生抢占。决定哪个序列组被抢占是由策略决定
        # 这行代码使用策略对象对当前正在运行的序列组列表按优先级进行排序。
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        # 这两行代码初始化两个新的列表：running（将要运行的序列组）和preempted（被抢占的序列组）。
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        # 这是一个循环，处理每一个当前正在运行的序列组。每次迭代中，它从self.running列表中取出一个序列组。
        while self.running:
            seq_group = self.running.pop(0)
            # 检查当前序列组是否可以增加新的token插槽。如果不能，进入下面的循环。
            while not self.block_manager.can_append_slot(seq_group):
                # 如果self.running列表仍有序列组，则取出最后一个（优先级最低的）序列组进行抢占。
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # 否则，抢占当前的seq_group序列组。
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            # 如果seq_group能够增加新的token插槽，则调用_append_slot方法
            # 为其增加新的插槽，并将其添加到running列表中。
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        # 在循环结束后，更新self.running为running列表。
        # 这意味着self.running现在包含了所有已成功分配了新插槽的序列组。
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        # 这段代码涉及尝试将处于SWAPPED状态的序列组切换回（swap in）为运行状态，如果可能的话。
        # 首先，使用策略对象按优先级对swapped中的序列组进行排序。
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        # 开始一个循环，只要swapped列表不为空，并且没有块要被换出，就继续循环。
        while self.swapped and not blocks_to_swap_out:
            # 获取swapped列表中的第一个序列组。
            seq_group = self.swapped[0]
            # If the sequence group has been preempted in this step, stop.
            # 检查这个序列组是否在这个步骤中被抢占。如果是，就终止循环。
            if seq_group in preempted:
                break
            # If the sequence group cannot be swapped in, stop.
            # 检查是否可以将这个序列组从SWAPPED状态切换回来。如果不可以，就终止循环。
            if not self.block_manager.can_swap_in(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            # 这部分代码确保运行状态的序列总数不超过最大序列数。
            # 它首先计算SWAPPED状态和RUNNING状态的序列数，并检查它们的总和是否超过允许的最大值。
            # 如果超过，就终止循环。
            num_new_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            num_curr_seqs = sum(
                seq_group.num_seqs(status=SequenceStatus.RUNNING)
                for seq_group in self.running)
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break

            # 从swapped列表中移除并获取第一个序列组。
            seq_group = self.swapped.pop(0)
            # 将这个序列组从SWAPPED状态切换回来。
            self._swap_in(seq_group, blocks_to_swap_in)
            # 为这个序列组添加新的插槽。
            self._append_slot(seq_group, blocks_to_copy)
            # 将这个序列组添加到running列表中，意味着现在它正在运行。
            self.running.append(seq_group)

        # 最后，计算RUNNING状态的所有序列的总数。
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        # 包装成SchedulerOutputs对象返回
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs

    # 这段代码定义了一个名为schedule的方法。这个方法的目的是根据调度器的内部状态
    # 生成一系列SequenceGroupMetadata对象，并将这些对象与调度的输出结果一起返回。
    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        # 首先调用_schedule()方法，对序列组进行调度，并将其结果存储在scheduler_outputs变量中。
        # 此方法可能会更改调度器的内部状态，如self.running、self.swapped和self.waiting。
        scheduler_outputs = self._schedule()

        # Create input data structures.
        # 初始化一个列表seq_group_metadata_list来存储计划好的SequenceGroupMetadata。
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        # 开始遍历已计划好的所有序列组。
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            # 为每个序列组初始化两个字典：seq_data（用于存储序列数据）和block_tables（用于存储块表）。
            seq_data: Dict[int, List[SequenceData]] = {}
            block_tables: Dict[int, List[int]] = {}
            # 遍历序列组中所有处于RUNNING状态的序列。
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                # 将当前序列的数据存储在seq_data字典中。
                seq_data[seq_id] = seq.data
                # 使用block_manager为当前序列获取块表，并将其存储在block_tables字典中。
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            # 为当前的序列组创建一个新的SequenceGroupMetadata对象，它包含了该组的所有元数据。
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
            )
            # 将新创建的SequenceGroupMetadata对象添加到列表seq_group_metadata_list中。
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    # update函数则是根据当前的找到已完成的block，将其空间释放，
    # 如果碰到beam search中的子seq，则将对应的空间释放，并将父seq fork为子seq。
    # 这段代码定义了一个名为update的函数，用于更新序列组的状态并处理新的序列输出。
    def update(
            self,
            seq_outputs: Dict[int, SequenceOutputs],
    ) -> List[SequenceGroup]:
        # 这是一个空列表，稍后将用来存储正在运行且其输出在seq_outputs中的序列组
        scheduled: List[SequenceGroup] = []
        # 这部分代码首先迭代self.running中的所有正在运行的序列组。
        # 对于每一个序列组，它检查该序列组中正在运行的序列是否其输出在seq_outputs中。
        # 如果是，则将该序列组添加到scheduled列表中。
        for seq_group in self.running:
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                if seq.seq_id in seq_outputs:
                    scheduled.append(seq_group)
                    break

        # Update the scheduled sequences and free blocks.
        for seq_group in scheduled:
            # Process beam search results before processing the new tokens.
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                output = seq_outputs[seq.seq_id]
                # 对于每一个正在运行的序列，它首先检查该序列是否是父序列的一个fork
                # （这是束搜索的一个特性）。如果是，它释放当前序列，并对父序列进行fork。
                if seq.seq_id != output.parent_seq_id:
                    # The sequence is a fork of the parent sequence (beam
                    # search). Free the current sequence.
                    self.block_manager.free(seq)
                    # Fork the parent sequence.
                    parent_seq = seq_group.find(output.parent_seq_id)
                    parent_seq.fork(seq)
                    self.block_manager.fork(parent_seq, seq)

            # Process the new tokens.
            # 对于每一个正在运行的序列，它将新的token追加到该序列中。
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # Append a new token to the sequence.
                output = seq_outputs[seq.seq_id]
                seq.append_token_id(output.output_token, output.logprobs)
        return scheduled

    # free_seq 是一个方法，用于释放与一个给定序列关联的资源，并更新该序列的状态。
    def free_seq(self, seq: Sequence, finish_status: SequenceStatus) -> None:
        seq.status = finish_status
        self.block_manager.free(seq)

    # free_finished_seq_groups方法则负责从self.running列表中移除已完成的序列组。
    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]

    # 这段代码定义了一个名为_allocate的方法。这个方法的主要目的是为一个指定的SequenceGroup分配资源，
    # 并将其中的所有序列的状态设置为RUNNING。
    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING

    # 这段代码定义了一个名为_append_slot的方法。它的主要功能是为SequenceGroup
    # 中正在运行的序列追加一个资源或内存块，并同时更新一个叫做blocks_to_copy的字典。
    def _append_slot(
            self,
            seq_group: SequenceGroup,
            blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            # 对每一个正在运行的序列`seq`，调用`self.block_manager`
            # 的`append_slot`方法尝试为其追加一个资源或内存块。
            ret = self.block_manager.append_slot(seq)
            # 返回值`ret`可能是一个包含两个整数的元组，或者为`None`。
            if ret is not None:
                # 如果`ret`不是`None`，则将其解包为两个整数`src_block`和`dst_block`。
                src_block, dst_block = ret
                # 检查`src_block`是否已经在`blocks_to_copy`字典中：
                if src_block in blocks_to_copy:
                    # 如果是，将`dst_block`追加到对应的列表中。
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    # 如果不是，创建一个新的条目，其中`src_block`是键，
                    # 值是一个包含`dst_block`的新列表。
                    blocks_to_copy[src_block] = [dst_block]

    # 这段代码定义了一个名为_preempt的私有方法，这个方法负责预占或
    # 中断SequenceGroup的执行，要么通过重新计算，要么通过交换内存。
    def _preempt(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_out: Dict[int, int],
            preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not supported. In such a case,
        # we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        # 如果调用时没有明确指定预占模式，那么这部分代码会根据SequenceGroup中
        # 运行状态的序列数量来决定使用哪种模式。
        if preemption_mode is None:
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            # 如果有一个正在运行的序列，则默认使用RECOMPUTE模式。
            if len(seqs) == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        # 如果是RECOMPUTE，调用_preempt_by_recompute方法。
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        # 如果是SWAP，调用_preempt_by_swap方法并传入blocks_to_swap_out。
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            assert False, "Invalid preemption mode."

    def _preempt_by_recompute(
            self,
            seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)

    def _preempt_by_swap(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_out: Dict[int, int],
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        for seq in seqs:
            seq.status = SequenceStatus.SWAPPED
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED


'''
上面完整解析了scheduler的代码，我们这里来总结一下流程。
首先在llm_engine.add_request函数中把根据输入prompt构造好的SequenceGroup添加到
 scheduler 的 waiting 队列里（self.scheduler.add_seq_group(seq_group)）。
 
scheduler的初始化函数中定义了三个代表状态的List[SequenceGroup] waiting, running, 
swapped 以及初始化了一个block_manager来管理 logical blocks 和 physical blocks 之间的映射关系。
然后在scheduler的代码实现中，schedule函数尤其关键。

首先，当waiting不空时，它会把一些waiting中的SequenceGroup加到running中，
但需要满足一些条件比如block_manager里面是否还有足够的空间可以塞得下这个SequenceGroup(对应 if not self.block_manager.can_allocate(seq_group): break)，
当前序列的prompt长度是否超出了prompt_limit，
已经批处理的token数量（num_batched_tokens）加上当前序列组（seq_group）的token数量（num_prompt_tokens）是否超过了配置中的最大token限制，
如果超过了，循环就会中断。还有一些限制可以看上面的代码。

然后，scheduler会遍历running里面的每个SequenceGroup，然后检查block_manager是否够塞得下。 
如果不行，则它会驱逐 running 队列中优先级最低的SequenceGroup，
如果空间够的话，则会对这个SequenceGroup allocate 相应的 physical blocks，
然后将其放入 update 后的 running 列表中。
经过这个过程，scheduler 更新了 running 列表，并把部分任务驱逐掉。

接下来，scheduler会过一遍swapped里面的每个SequenceGroup，
尝试 swap in 那些能够 swap 的 SequenceGroup，并把它们放到新的 running 列表中。

scheduler做完上述过程之后，
最后会把相应的信息（swap in/out 的 blocks，blocks_to_copy）包装成SchedulerOutputs对象供后面worker进行Model Execution（也包含序列本身相关的信息叫 seq_group_metadata_list）。
'''