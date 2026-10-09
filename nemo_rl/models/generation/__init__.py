# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from typing import cast

from transformers import GenerationConfig as HFGenerationConfig
from transformers import PreTrainedTokenizerBase

from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.vllm import VllmConfig

TokenizerType = PreTrainedTokenizerBase


def configure_generation_config(
    config: GenerationConfig,
    tokenizer: TokenizerType,
    is_eval: bool = False,
    has_refit_draft_weights: bool = False,
) -> GenerationConfig:
    """Apply specific configurations to generation config."""
    # tokenizer setting
    if "_pad_token_id" in config:
        warnings.warn(
            "'_pad_token_id' found in generation config and will be overridden with tokenizer.pad_token_id. "
            "Note: '_pad_token_id' is intended for internal use and has no effect when set in user-provided configs.",
            UserWarning,
        )
    config["_pad_token_id"] = tokenizer.pad_token_id
    if config["stop_token_ids"] is None:
        config["stop_token_ids"] = [tokenizer.eos_token_id]

    # vllm setting
    if config["backend"] == "vllm":
        config = cast(VllmConfig, config)
        # set load_format
        config["vllm_cfg"]["load_format"] = "auto" if is_eval else "dummy"
        speculative_config = config.get("vllm_kwargs", {}).get("speculative_config")
        if speculative_config:
            # Speculative decoding needs real startup weights unless the draft
            # weights will be pushed into vLLM during the initial refit.
            if not is_eval and not has_refit_draft_weights:
                warnings.warn(
                    "Speculative decoding is enabled without draft refit sync. "
                    "Setting vllm_cfg['load_format'] to 'auto' so the drafter does "
                    "not start from dummy weights."
                )
                config["vllm_cfg"]["load_format"] = "auto"

        # Respect the skip_tokenizer_init setting from the config. VLMs for example, require this to be False.
        if "skip_tokenizer_init" not in config["vllm_cfg"]:
            # set skip_tokenizer_init
            if (
                is_eval
                or config["stop_strings"] is not None
                or config["vllm_cfg"].get("expose_http_server", None)
            ):
                config["vllm_cfg"]["skip_tokenizer_init"] = False
            else:
                config["vllm_cfg"]["skip_tokenizer_init"] = True

        if config["vllm_cfg"].get("expose_http_server", None):
            _check_eos_ends_assistant_turn(tokenizer)

    return config


def _check_eos_ends_assistant_turn(tokenizer: TokenizerType) -> None:
    """[CUSTOM] Fail fast when the tokenizer EOS is not the token that ends a chat turn.

    _replace_prefix_tokens (vllm_worker_async.py) finds turn boundaries by counting
    tokenizer.eos_token_id in the rendered chat template, and vLLM stops generation on
    generation_config.json's eos_token_id.
    """
    eos = tokenizer.eos_token_id
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ],
        tokenize=False,
    )
    if eos not in tokenizer.encode(rendered, add_special_tokens=False):
        raise ValueError(
            f"Tokenizer EOS {tokenizer.eos_token!r} ({eos}) never appears in a rendered "
            f"assistant turn, so multi-call rollouts cannot be spliced. Stale tokenizer "
            f"files in {tokenizer.name_or_path}? Rendered: {rendered!r}"
        )

    try:
        gen_eos = HFGenerationConfig.from_pretrained(tokenizer.name_or_path).eos_token_id
    except OSError:
        # No generation_config.json: vLLM stops on the tokenizer EOS alone.
        return
    gen_eos_ids = {gen_eos} if isinstance(gen_eos, int) else set(gen_eos or [])
    if gen_eos_ids and gen_eos_ids != {eos}:
        raise ValueError(
            f"generation_config.json eos_token_id {sorted(gen_eos_ids)} != tokenizer EOS "
            f"{tokenizer.eos_token!r} ({eos}) in {tokenizer.name_or_path}; vLLM would end "
            f"turns on tokens the prefix splicing does not recognize. Stale assets?"
        )
