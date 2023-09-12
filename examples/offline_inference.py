# 0x0. 前言
# 本文在对VLLM进行解析时只关注单卡情况，忽略基于ray做分布式推理的所有代码。
#
# 0x1. 运行流程梳理
# 先从使用VLLM调用opt-125M模型进行推理的脚本看起：
# 可以看到这里创建了一个LLM对象，然后调用了LLM对象的generate函数。
# 这就是vllm的入口点，接下来我们对LLM这个类的generaet过程进行解析。

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
