from typing import List

from vllm.sequence import SequenceGroup

'''
0x5. Scheduler
有了上面的铺垫，接下来我们可以对Scheduler进行解析了，对应源码在vllm/vllm/core/scheduler.py。
在解析Scheduler的代码实现之前，我们解析一下vllm/vllm/core/目录下的policy.py和block_manager.py，
它们分别负责实现Scheduler的队列中的优先法则以及KV Cache块的管理过程，并且实现都非常简单和简短。

首先来看 vllm/vllm/core/policy.py
'''
# 这是一个抽象的策略类。
class Policy:
    # 计算一个 SequenceGroup 的优先级。子类需要重写这个方法来提供具体的优先级计算逻辑。
    def get_priority(
            self,
            now: float,
            seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    # 根据优先级对一组 SequenceGroup 进行排序。这是一个通用方法，使用了前面的 get_priority() 方法。
    def sort_by_priority(
            self,
            now: float,
            seq_groups: List[SequenceGroup],
    ) -> List[SequenceGroup]:
        return sorted(
            seq_groups,
            key=lambda seq_group: self.get_priority(now, seq_group),
            reverse=True,
        )


# 这是 Policy 的一个具体子类，它实现了先到先得（First-Come-First-Serve, FCFS）的调度策略。
# 它重写了 get_priority() 方法，以便为每个 SequenceGroup 分配一个与其到达时间相关的优先级。
# 此处，优先级是当前时间减去序列组的到达时间，这意味着越早到达的 SequenceGroup 优先级越高。
class FCFS(Policy):

    def get_priority(
            self,
            now: float,
            seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.arrival_time


# 这是一个工厂类，用于创建和返回特定的策略对象。它使用一个名为 _POLICY_REGISTRY
# 的字典来注册策略类，其中键是策略的名称（如 'fcfs'），值是相应的策略类。
class PolicyFactory:
    _POLICY_REGISTRY = {
        'fcfs': FCFS,
    }

    # get_policy(): 是一个类方法，它接受策略名称作为参数，
    # 查找 _POLICY_REGISTRY 字典并返回对应的策略对象实例。
    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)