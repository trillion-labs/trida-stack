import numpy as np

IMAGE_TOKEN = "<|image|>"
PROMPT_PREFIX = "[gMASK]<sop><|user|>\n<|begin_of_image|>"


def build_ocr_prompt_text(
    image_tokens: int,
    task_prompt: str = "Text Recognition:",
) -> str:
    if image_tokens <= 0:
        raise ValueError("image_tokens must be positive")
    return PROMPT_PREFIX + IMAGE_TOKEN * image_tokens + _prompt_suffix(task_prompt)


def _prompt_suffix(task_prompt: str) -> str:
    return "<|end_of_image|>" + task_prompt + "<|assistant|>\n"


def _encode_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    # transformers returns a list of ids; the Rust tokenizers.Tokenizer returns Encoding.
    return list(getattr(encoded, "ids", encoded))


def _image_token_id(tokenizer) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        convert = getattr(tokenizer, "token_to_id", None)
    if convert is None:
        raise TypeError("Tokenizer exposes no token-to-id lookup")
    token_id = convert(IMAGE_TOKEN)
    if token_id is None:
        raise ValueError(f"Tokenizer has no {IMAGE_TOKEN} token")
    return int(token_id)


_AFFIX_CACHE: dict[tuple[int, str], tuple[list[int], list[int], int]] = {}


def _prompt_affixes(tokenizer, task_prompt: str) -> tuple[list[int], list[int], int]:
    """Token ids for the parts of the prompt that do not depend on the image size.

    Both affixes end and begin on a special token, so splicing the image placeholders
    between them yields the same ids as tokenizing the whole prompt string. The cached
    values save re-tokenizing a prompt that is thousands of characters long — the
    placeholder run alone is nine characters per visual token — on every request.
    """
    key = (id(tokenizer), task_prompt)
    cached = _AFFIX_CACHE.get(key)
    if cached is None:
        cached = (
            _encode_ids(tokenizer, PROMPT_PREFIX),
            _encode_ids(tokenizer, _prompt_suffix(task_prompt)),
            _image_token_id(tokenizer),
        )
        _AFFIX_CACHE[key] = cached
    return cached


def build_ocr_prompt_token_ids(
    tokenizer,
    image_tokens: int,
    task_prompt: str = "Text Recognition:",
) -> tuple[list[int], int]:
    if image_tokens <= 0:
        raise ValueError("image_tokens must be positive")
    prefix, suffix, image_token_id = _prompt_affixes(tokenizer, task_prompt)
    token_ids = prefix + [image_token_id] * image_tokens + suffix
    if token_ids.count(image_token_id) != image_tokens:
        raise ValueError("Tokenizer did not preserve the requested image placeholders")
    return token_ids, image_token_id


def build_ocr_prompt_tensors(
    tokenizer,
    image_tokens: int,
    task_prompt: str = "Text Recognition:",
) -> tuple[np.ndarray, np.ndarray]:
    token_ids, image_token_id = build_ocr_prompt_token_ids(
        tokenizer,
        image_tokens,
        task_prompt,
    )
    input_ids = np.asarray([token_ids], dtype=np.int64)
    modality_types = (input_ids == image_token_id).astype(np.int64)
    return input_ids, modality_types
