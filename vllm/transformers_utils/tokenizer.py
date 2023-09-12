from typing import List, Tuple, Union

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from vllm.logger import init_logger

logger = init_logger(__name__)

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if "llama" in tokenizer_name.lower() and kwargs.get("use_fast", True):
        logger.info(
            "For some LLaMA-based models, initializing the fast tokenizer may "
            "take a long time. To eliminate the initialization time, consider "
            f"using '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs)
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA-based "
            f"model, use '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer.")
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if (not trust_remote_code and
            ("does not exist or is not currently imported." in str(e)
             or "requires you to execute the tokenizer file" in str(e))):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead.")
    return tokenizer

# 这个函数 detokenize_incrementally 负责将新token与之前的输出toke一起进行逐渐解码。
# tokenizer: 这是用于解码token的分词器对象.
# prev_output_tokens: 以前已解码的输出token列表。
# new_token_id: 要解码的新token ID。
# skip_special_tokens: 一个布尔值，如果为 True，特殊token将被跳过。
def detokenize_incrementally(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        prev_output_tokens: List[str],
        new_token_id: int,
        skip_special_tokens: bool,
) -> Tuple[str, str]:
    """Detokenizes the new token in conjuction with the previous output tokens.

    NOTE: This function does not update prev_output_tokens.

    # 返回一个元组，其中包括新token作为字符串和新的输出文本作为字符串

    Returns:
        new_token: The new token as a string.
        output_text: The new output text as a string.
    """
    # 如果 skip_special_tokens 为 True 并且新token ID是特殊令牌，
    # 则直接返回 None 和 prev_output_tokens。
    if skip_special_tokens and (new_token_id in tokenizer.all_special_ids):
        return None, prev_output_tokens
    # 使用 tokenizer.convert_ids_to_tokens 方法将新的token ID转换为token字符串。
    new_token = tokenizer.convert_ids_to_tokens(
        new_token_id, skip_special_tokens=skip_special_tokens)
    # 将新token添加到先前的输出token列表中。
    output_tokens = prev_output_tokens + [new_token]

    # Convert the tokens to a string.
    # Optimization: If the tokenizer does not have `added_tokens_encoder`,
    # then we can directly use `convert_tokens_to_string`.
    if not getattr(tokenizer, "added_tokens_encoder", {}):
        # 如果分词器没有 added_tokens_encoder 属性，则可以直接使用
        # convert_tokens_to_string 方法将输出token转换为字符串。
        output_text = tokenizer.convert_tokens_to_string(output_tokens)
        return new_token, output_text

    # 否则，需要更复杂的逻辑来处理添加的token和特殊token，这涉及对输出token进行迭代并逐个解析。
    # Adapted from
    # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # NOTE(woosuk): The following code is slow because it runs a for loop over
    # the output_tokens. In Python, running a for loop over a list can be slow
    # even when the loop body is very simple.
    # 存储解析的子文本片段的列表。
    sub_texts = []
    # 存储当前正在处理的token的子文本的列表。
    current_sub_text = []
    # 对 output_tokens 中的每个token执行迭代。
    for token in output_tokens:
        # 如果 skip_special_tokens 为 True 并且token是特殊token，则跳过当前迭代。
        if skip_special_tokens and token in tokenizer.all_special_tokens:
            continue
        # 如果token在 tokenizer.added_tokens_encoder 中，则：
        if token in tokenizer.added_tokens_encoder:
            if current_sub_text:
                # 如果 current_sub_text 不为空，则使用 convert_tokens_to_string
                # 将其转换为字符串，并将结果添加到 sub_texts 中。
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                # 清空 current_sub_text。
                current_sub_text = []
            # 将token本身添加到 sub_texts 中。
            sub_texts.append(token)
        else:
            # 如果令牌不是特殊令牌也不是添加的令牌，则将其添加到 current_sub_text 中。
            current_sub_text.append(token)
    if current_sub_text:
        # 如果遍历完成后 current_sub_text 不为空，则转换并添加到 sub_texts 中。
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    # 使用空格将 sub_texts 中的所有子文本连接在一起，形成最终的输出文本。
    output_text = " ".join(sub_texts)
    return new_token, output_text