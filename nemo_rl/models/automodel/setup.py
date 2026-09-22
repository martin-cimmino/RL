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

"""Setup utilities for automodel-based training in NeMo RL."""

import importlib
import inspect
import os
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.utils.checkpoint
from hydra.utils import get_class
from nemo_automodel import NeMoAutoModelForSequenceClassification
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.config.loader import _resolve_target
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.mesh_utils import create_device_mesh
from nemo_automodel.components.distributed.tensor_utils import get_cpu_state_dict
from nemo_automodel.components.moe.config import MoEParallelizerConfig
from nemo_automodel.shared.utils import dtype_from_str
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.data.chat_templates import COMMON_CHAT_TEMPLATES
from nemo_rl.models.automodel.config import (
    DistributedContext,
    ModelAndOptimizerState,
    RuntimeConfig,
)
from nemo_rl.models.policy import PolicyConfig, TokenizerConfig
from nemo_rl.models.policy.utils import configure_dynamo_cache, resolve_model_class

STRING_TO_DTYPE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _maybe_set_force_hf(automodel_kwargs: dict, model_config) -> None:
    """Validate and maybe auto-set force_hf based on adapter compatibility.

    Custom model implementations (e.g. Qwen2, Llama) use state_dict_adapters to
    convert between native and HF weight formats. NeMo RL's weight syncing requires
    the adapter to implement `convert_single_tensor_to_hf`. Some adapters (like
    CombinedProjectionStateDictAdapter) don't implement this yet.

    This function checks the adapter BEFORE model loading to avoid wasting time:
    - force_hf=True: no check needed, HF model won't have an adapter.
    - force_hf not set + adapter incompatible: auto-set force_hf=True with a warning.
    - force_hf=False + adapter incompatible: raise an error telling the user to set
      force_hf=True or file an issue to NeMo-Automodel.

    See: https://github.com/NVIDIA-NeMo/RL/issues/2072
    """
    # If force_hf is already True, no check needed — HF model won't use an adapter.
    if automodel_kwargs.get("force_hf") is True:
        return

    # If the architecture doesn't have a custom model implementation,
    # HF model will be used regardless — no adapter involved.
    architectures = getattr(model_config, "architectures", None) or []
    if (
        not architectures
        or architectures[0] not in ModelRegistry.model_arch_name_to_cls
    ):
        return

    arch = architectures[0]
    model_cls = ModelRegistry.model_arch_name_to_cls[arch]

    # Check the adapter class from the model's sibling state_dict_adapter module.
    # Each custom model (e.g. nemo_automodel.components.models.qwen2.model) has
    # a corresponding state_dict_adapter.py in the same package.
    adapter_ok = True
    try:
        package_path = model_cls.__module__.rsplit(".", 1)[0]
        adapter_module = importlib.import_module(f"{package_path}.state_dict_adapter")
        for name, obj in inspect.getmembers(adapter_module, inspect.isclass):
            if name.endswith("StateDictAdapter"):
                method = getattr(obj, "convert_single_tensor_to_hf", None)
                if method is None or getattr(method, "__isabstractmethod__", False):
                    adapter_ok = False
                break
    except (ImportError, AttributeError):
        # Can't find or import the adapter module. This means the model either
        # doesn't use an adapter or has a non-standard layout — skip the check.
        return

    if adapter_ok:
        return

    force_hf_explicitly_false = (
        "force_hf" in automodel_kwargs and automodel_kwargs["force_hf"] is False
    )
    if force_hf_explicitly_false:
        raise RuntimeError(
            f"force_hf=False but the custom model for {arch} uses an adapter that "
            f"does not implement 'convert_single_tensor_to_hf', which is required "
            f"for weight syncing. Please set "
            f"`policy.dtensor_cfg.automodel_kwargs.force_hf=true` or file an issue "
            f"at https://github.com/NVIDIA-NeMo/Automodel to add support."
        )

    # force_hf not set — auto-enable it.
    print(
        f"WARNING: Custom model for {arch} uses an adapter that does not implement "
        f"'convert_single_tensor_to_hf' (required for weight syncing). "
        f"Auto-setting force_hf=True. To silence this warning, explicitly set "
        f"`policy.dtensor_cfg.automodel_kwargs.force_hf=true` in your config."
    )
    automodel_kwargs["force_hf"] = True


def _maybe_patch_domyn_edge_activation_checkpointing(
    model, activation_checkpointing: bool
) -> None:
    """Monkeypatches each decoder layer's actual runtime class to checkpoint.

    Automodel's generic activation-checkpointing (applied inside
    from_pretrained -> fsdp2_strategy_parallelize) only wraps submodules
    named `mlp`/`self_attn`/`input_layernorm`/`post_attention_layernorm`
    (Llama-style naming). DomynEdge's real attribute names are `attention`/
    `feedforward`/`pre_attention_norm`/`post_attention_norm`/
    `pre_feedforward_norm`/`post_feedforward_norm` — silent no-op for this
    model.

    A property-based attribute-aliasing approach (making `layer.self_attn`
    resolve to `layer.attention`) does NOT work here. Patches `forward`
    directly instead, wrapping the attention and feedforward sub-blocks
    in `torch.utils.checkpoint.checkpoint`.

    Must run AFTER `from_pretrained`, on the model's ACTUAL runtime layer
    classes -- not before, on the plain `DomynEdgeDecoderLayer` imported
    from the modeling module.
    """
    if not activation_checkpointing:
        return
    architectures = getattr(getattr(model, "config", None), "architectures", None) or []
    if not architectures or architectures[0] != "DomynEdgeForCausalLM":
        return

    def _checkpointed_forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_embeddings=None,
        **kwargs,
    ):
        def attn_block(hidden_states):
            return self.post_attention_norm(
                self.attention(
                    hidden_states=self.pre_attention_norm(hidden_states),
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            )

        hidden_states = hidden_states + torch.utils.checkpoint.checkpoint(
            attn_block, hidden_states, use_reentrant=False
        )

        def ffn_block(hidden_states):
            return self.post_feedforward_norm(
                self.feedforward(self.pre_feedforward_norm(hidden_states))
            )

        hidden_states = hidden_states + torch.utils.checkpoint.checkpoint(
            ffn_block, hidden_states, use_reentrant=False
        )
        return hidden_states

    layers = model.model.layers
    if isinstance(layers, torch.nn.ModuleDict):
        layers = layers.values()
    for layer in layers:
        layer_cls = type(layer)
        if getattr(layer_cls, "_activation_checkpointing_patched", False):
            continue
        layer_cls.forward = _checkpointed_forward
        layer_cls._activation_checkpointing_patched = True


def _maybe_patch_domyn_edge_rope_autocast(model) -> None:
    """Forces DomynEdge's RoPE frequency computation to run outside autocast.

    DomynEdgeRotaryEmbedding.forward computes

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

    The `.float()` calls do NOT survive autocast: torch.autocast overrides the input dtypes
    of matmul, so under a bf16 autocast region this product is evaluated in bf16. bf16 has
    8 mantissa bits, so position ids above 255 are not exactly representable -- the ulp
    doubles at every power of two, reaching 64 at position 16384 and 128 near 32767. The
    RoPE angles are therefore quantized progressively harder the deeper into a sequence you
    go, and the model's attention degrades with position.

    Upstream transformers guards against exactly this; see modeling_llama.py, which wraps
    the same three lines in `maybe_autocast(device_type=..., enabled=False)  # Force
    float32`. DomynEdge's trust_remote_code modeling file simply omits that guard, so the
    bug only appears where the model is run under autocast -- which is the training path
    (fp32 master weights + bf16 autocast) and not vLLM (pure bf16).
    """
    architectures = getattr(getattr(model, "config", None), "architectures", None) or []
    if not architectures or architectures[0] != "DomynEdgeForCausalLM":
        return

    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rotary is None:
        return

    rotary_cls = type(rotary)
    if getattr(rotary_cls, "_domyn_edge_rope_autocast_patched", False):
        return

    original_forward = rotary_cls.forward

    @torch.no_grad()
    def _forward_no_autocast(self, x, position_ids):
        device_type = x.device.type
        if not isinstance(device_type, str) or device_type == "mps":
            device_type = "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            return original_forward(self, x, position_ids)

    rotary_cls.forward = _forward_no_autocast
    rotary_cls._domyn_edge_rope_autocast_patched = True


def _maybe_patch_domyn_edge_varlen_attn(model_config) -> None:
    """Monkeypatches torch.nn.attention.varlen.varlen_attn for DomynEdge.

    DomynEdge's own trust_remote_code modeling file calls varlen_attn with
    kwargs (scale, window_size, enable_gqa) that match flash-attn's API, not
    our pinned torch==2.10.0's actual varlen_attn signature (no scale/
    window_size/enable_gqa at all) — a checkpoint/torch-version mismatch, not
    something to fix in this repo. Routes those calls to
    flash_attn.flash_attn_varlen_func instead, restoring the causal=True the
    checkpoint code never passes. No-op for every other model.
    """
    architectures = getattr(model_config, "architectures", None) or []
    if not architectures or architectures[0] != "DomynEdgeForCausalLM":
        return

    import torch.nn.attention.varlen

    original_varlen_attn = torch.nn.attention.varlen.varlen_attn

    def _varlen_attn_compat(
        query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, *args, **kwargs
    ):
        if "scale" in kwargs or "window_size" in kwargs or "enable_gqa" in kwargs:
            import flash_attn

            scale = kwargs.pop("scale", None)
            window_size = kwargs.pop("window_size", (-1, -1))
            kwargs.pop(
                "enable_gqa", None
            )  # flash-attn auto-detects GQA from q/k head-count mismatch
            return flash_attn.flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seq_q,
                cu_seq_k,
                max_q,
                max_k,
                softmax_scale=scale,
                window_size=window_size,
                causal=True,
                **kwargs,
            )
        return original_varlen_attn(
            query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, *args, **kwargs
        )

    torch.nn.attention.varlen.varlen_attn = _varlen_attn_compat


class _SkipLMHeadSignal(Exception):
    """Raised from a lm_head forward-pre-hook to abort a forward pass early.

    Carries the pre-lm_head hidden states out of the model call.
    """

    def __init__(self, hidden_states):
        super().__init__()
        self.hidden_states = hidden_states


def _maybe_patch_domyn_edge_skip_lm_head_support(model) -> None:
    """Registers a forward-pre-hook on lm_head to support skip_lm_head.

    Used by the linear-CE-fusion training-loss path (nemo_rl.models.automodel
    .train.model_forward's skip_lm_head=True), which needs pre-lm_head hidden
    states to apply lm_head+cross-entropy together via a fused kernel instead
    of materializing the full [seq_len, vocab_size] logits tensor.

    Two other mechanisms were tried and both failed under the real
    setup_model_and_optimizer/from_pretrained + FSDP2 + TP path:
    1. Calling `model.model(...)` directly
    2. Monkeypatching `type(model).forward`

    A forward-pre-hook on `lm_head` sidesteps all of that.
    """
    architectures = getattr(getattr(model, "config", None), "architectures", None) or []
    if not architectures or architectures[0] != "DomynEdgeForCausalLM":
        return

    lm_head = model.lm_head
    if getattr(lm_head, "_skip_lm_head_hook_registered", False):
        return

    enabled = {"value": False}
    model._nemo_rl_skip_lm_head_state = enabled

    def _capture_hidden_states_hook(module, args):
        if enabled["value"]:
            raise _SkipLMHeadSignal(args[0])
        return None

    lm_head.register_forward_pre_hook(_capture_hidden_states_hook)
    lm_head._skip_lm_head_hook_registered = True


def _maybe_patch_domyn_edge_attention_head_sharding(model_config) -> None:
    """Monkeypatches Automodel's post-TP head-count update for DomynEdge.

    After applying a tensor-parallel plan, Automodel auto-updates each
    attention layer's local head count (_update_attention_head_counts_for_tp)
    so DomynEdgeAttention.forward()'s .view() calls use the per-rank head
    count instead of the global one — without it, .view() is called with the
    *global* head count against a tensor TP already sharded to 1/tp_size the
    heads, e.g. observed: `shape '[1, 16384, 32, 256]' is invalid for input
    of size 33554432` (33554432 == 1*16384*8*256, i.e. the real, TP=4-local
    head count of 8, not the global 32 the .view() call was still using).

    Two independent gaps, both from the same root cause — DomynEdge's real
    attention module is named "attention", not "self_attn" (see
    modeling_domynedge.py):

    1. Whether the update runs at all is gated by _attention_is_head_sharded,
       which only recognizes "self_attn.*_proj" keys, so it always returns
       False for DomynEdge's custom_parallel_plan.
    2. Even when forced to run, _update_attention_head_counts_for_tp's own
       body only touches `attn.num_heads`/`attn.num_key_value_heads` on
       layers matching `hasattr(layer, "self_attn")` — that pattern matches
       standard HF attention classes that cache their head count as instance
       attributes read at forward time. DomynEdge's DomynEdgeAttention has no
       such attributes; it reads `self.config.num_attention_heads` /
       `num_key_value_heads` directly off the (shared) config object every
       forward call. So even with the gate fixed, the original function's
       update logic is a complete no-op for this architecture — it needs the
       shared config object's attributes updated instead, which is what
       every layer's `self.config` (same object reference, not a copy)
       actually reads.

    No-op for every other model.
    """
    architectures = getattr(model_config, "architectures", None) or []
    if not architectures or architectures[0] != "DomynEdgeForCausalLM":
        return

    import nemo_automodel.components.distributed.parallelizer as _parallelizer

    # Idempotency guard: without this, calling this function twice in the
    # same process (e.g. two models/setup calls sharing one process, or a
    # test harness building multiple models) captures the FIRST call's
    # compat wrapper as "original" on the second call, nesting the
    # tp_size-division in _update_head_counts_compat below twice
    if getattr(_parallelizer, "_domyn_edge_head_sharding_patched", False):
        return
    _parallelizer._domyn_edge_head_sharding_patched = True

    original_is_head_sharded = _parallelizer._attention_is_head_sharded

    def _is_head_sharded_compat(model_parallel_plan):
        aliased = dict(model_parallel_plan)
        for key, style in model_parallel_plan.items():
            if key.endswith(
                ("attention.q_proj", "attention.k_proj", "attention.v_proj")
            ):
                aliased[key.replace("attention.", "self_attn.", 1)] = style
        return original_is_head_sharded(aliased)

    _parallelizer._attention_is_head_sharded = _is_head_sharded_compat

    original_update_head_counts = _parallelizer._update_attention_head_counts_for_tp

    def _update_head_counts_compat(model, tp_size):
        config = getattr(model, "config", None)
        model_architectures = getattr(config, "architectures", None) or []
        if (
            tp_size > 1
            and config is not None
            and model_architectures
            and model_architectures[0] == "DomynEdgeForCausalLM"
        ):
            config.num_attention_heads = config.num_attention_heads // tp_size
            if getattr(config, "num_key_value_heads", None) is not None:
                config.num_key_value_heads = config.num_key_value_heads // tp_size
        return original_update_head_counts(model, tp_size)

    _parallelizer._update_attention_head_counts_for_tp = _update_head_counts_compat


def get_domyn_edge_tp_plan(sequence_parallel: bool = False) -> dict:
    """Tensor-parallel plan for DomynEdge.

    `model.embed_tokens`/`lm_head` are deliberately left OUT of the TP plan
    proper (never given a Colwise/RowwiseParallel style, i.e. their weights
    stay vocab-unsharded) rather than TP-sharded along the vocab dimension.
    DomynEdge's vocab_size (115276) isn't evenly divisible by tp_size

    `sequence_parallel=True` adds Automodel's own generic SP overrides
    (mirrors `_get_parallel_plan`'s `base_model_sp_plan` for stock
    Llama-style models), adapted to DomynEdge.
    """
    from functools import partial

    from torch.distributed.tensor import DTensor, distribute_module
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        PrepareModuleOutput,
        RowwiseParallel,
        SequenceParallel,
    )
    from torch.distributed.tensor.placement_types import Replicate, Shard

    class _SequenceParallelNormWithFullOutput(SequenceParallel):
        """Custom SequenceParallel for DomynEdge.

        Like SequenceParallel, but the module's output is redistributed
        to a genuine full (Replicate) tensor via an actual all-gather,
        instead of SequenceParallel's own to_local().
        """

        def _apply(self, module, device_mesh):
            def _prepare_full_output_fn(mod, outputs, device_mesh):
                return (
                    outputs.full_tensor() if isinstance(outputs, DTensor) else outputs
                )

            return distribute_module(
                module,
                device_mesh,
                self._replicate_module_fn,
                partial(self._prepare_input_fn, self.sequence_sharding),
                _prepare_full_output_fn,
            )

    plan: dict = {
        "model.layers.*.attention.q_proj": ColwiseParallel(),
        "model.layers.*.attention.k_proj": ColwiseParallel(),
        "model.layers.*.attention.v_proj": ColwiseParallel(),
        "model.layers.*.attention.o_proj": RowwiseParallel(),
        "model.layers.*.feedforward.gate_proj": ColwiseParallel(),
        "model.layers.*.feedforward.up_proj": ColwiseParallel(),
        "model.layers.*.feedforward.down_proj": RowwiseParallel(),
    }
    if sequence_parallel:
        plan.update(
            {
                "model.embed_tokens": PrepareModuleOutput(
                    output_layouts=Replicate(),
                    desired_output_layouts=Shard(1),
                    use_local_output=False,
                ),
                "model.layers.*.pre_attention_norm": SequenceParallel(),
                "model.layers.*.attention.o_proj": RowwiseParallel(
                    output_layouts=Shard(1), use_local_output=False
                ),
                "model.layers.*.post_attention_norm": SequenceParallel(),
                "model.layers.*.pre_feedforward_norm": SequenceParallel(),
                "model.layers.*.feedforward.down_proj": RowwiseParallel(
                    output_layouts=Shard(1), use_local_output=False
                ),
                "model.layers.*.post_feedforward_norm": SequenceParallel(),
                "model.norm": _SequenceParallelNormWithFullOutput(),
            }
        )
    return plan


def get_tokenizer(
    tokenizer_config: TokenizerConfig, get_processor: bool = False
) -> Union[PreTrainedTokenizerBase, AutoProcessor]:
    """Get tokenizer using NeMoAutoTokenizer for automodel workers.

    Uses NeMoAutoTokenizer which provides custom tokenizer dispatch per model type
    and falls back to NeMoAutoTokenizerWithBosEosEnforced for default handling.

    Args:
        tokenizer_config: A dictionary containing tokenizer configuration.
            Required keys:
                - name: The name or path of the pretrained tokenizer
            Optional keys:
                - chat_template: The chat template to use. Can be:
                    - None: Uses a passthrough template that just returns message content
                    - "default": Uses the tokenizer's default template
                    - A file path ending in ".jinja": Loads template from file
                    - A custom jinja2 template string
                    If not specified, the tokenizer's default template will be used.
                - chat_template_kwargs: Arguments passed to tokenizer.apply_chat_template()
        get_processor: Whether to return a processor (via AutoProcessor) instead of a tokenizer.

    Returns:
        The configured tokenizer or processor instance.
    """
    processor = None

    if get_processor:
        processor = AutoProcessor.from_pretrained(
            tokenizer_config["name"], trust_remote_code=True, use_fast=True
        )
        tokenizer = processor.tokenizer
    else:
        tokenizer = NeMoAutoTokenizer.from_pretrained(
            tokenizer_config["name"], trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "chat_template" in tokenizer_config:
        if tokenizer_config["chat_template"] is None:
            print("Using passthrough chat template")
            tokenizer.chat_template = COMMON_CHAT_TEMPLATES.passthrough_prompt_response
        elif tokenizer_config["chat_template"].lower() == "default":
            print("Using tokenizer's default chat template")
        elif tokenizer_config["chat_template"].endswith(".jinja"):
            template_path = tokenizer_config["chat_template"]
            print(f"Loading chat template from file: {template_path}")
            with open(template_path, "r") as f:
                tokenizer.chat_template = f.read()
        else:
            print("Using custom chat template")
            tokenizer.chat_template = tokenizer_config["chat_template"]
    else:
        print("No chat template provided, using tokenizer's default")

    if (
        "chat_template_kwargs" in tokenizer_config
        and tokenizer_config["chat_template_kwargs"] is not None
    ):
        assert isinstance(tokenizer_config["chat_template_kwargs"], dict), (
            "chat_template_kwargs should be a dictionary"
        )
        tokenizer.apply_chat_template = partial(
            tokenizer.apply_chat_template, **tokenizer_config["chat_template_kwargs"]
        )

    if processor is not None:
        processor.pad_token = tokenizer.pad_token
        processor.eos_token = tokenizer.eos_token
        processor.bos_token = tokenizer.bos_token
        processor.pad_token_id = tokenizer.pad_token_id
        processor.eos_token_id = tokenizer.eos_token_id
        processor.bos_token_id = tokenizer.bos_token_id
        processor.name_or_path = tokenizer.name_or_path

    return tokenizer if processor is None else processor


def validate_and_prepare_config(
    config: PolicyConfig,
    processor: Optional[AutoProcessor],
    rank: int,
) -> RuntimeConfig:
    """Validate configuration and prepare runtime settings.

    This function validates the policy configuration, sets environment variables,
    determines model configuration, and returns runtime settings as a named tuple.

    Args:
        config: Policy configuration dictionary
        processor: Optional processor for multimodal models
        rank: Current process rank

    Returns:
        RuntimeConfig named tuple containing validated configuration values

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If incompatible settings are detected
    """
    # Set basic configuration
    is_vlm = processor is not None
    is_generation_colocated = None
    sampling_params = None
    if "generation" in config and config["generation"] is not None:
        generation_cfg = config["generation"]
        # set generation colocated
        is_generation_colocated = generation_cfg["colocated"]["enabled"]
        # set sampling params
        sampling_params = TrainingSamplingParams(
            top_k=generation_cfg["top_k"],
            top_p=generation_cfg["top_p"],
            temperature=generation_cfg["temperature"],
        )

    # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
    # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
    if not is_generation_colocated:
        os.environ["NCCL_CUMEM_ENABLE"] = "1"

    # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
    # with different order of node_bundles
    configure_dynamo_cache()

    # Parse precision
    precision = config["precision"]
    if precision not in STRING_TO_DTYPE:
        raise ValueError(f"Unknown precision: {precision}")
    dtype = STRING_TO_DTYPE[precision]

    # Get other configuration values
    cpu_offload = config["dtensor_cfg"]["cpu_offload"]
    offload_optimizer_for_logprob = config.get("offload_optimizer_for_logprob", False)
    max_grad_norm = config["max_grad_norm"]
    enable_seq_packing = config["sequence_packing"]["enabled"]
    model_name = config["model_name"]

    # Validate sequence packing
    if enable_seq_packing:
        if is_vlm:
            raise ValueError(
                "Sequence packing is not supported for VLM models. "
                "Please set policy.sequence_packing.enabled = False to train VLM models."
            )
        print(f"[Rank {rank}] Sequence packing is enabled for model {model_name}")
        print(f"[Rank {rank}] Using FlashAttention2 for sequence packing")

    # Get HF config overrides
    hf_config_overrides = config.get("hf_config_overrides", {}) or {}

    rope_scaling = hf_config_overrides.get("rope_scaling") or {}
    assert rope_scaling.get("rope_type") != "yarn", (
        "YaRN RoPE scaling is not supported with the automodel (DTensor) backend. "
        "Please use the Megatron backend (policy.megatron_cfg.enabled=True) for YaRN."
    )

    # NeMoAutoModelForCausalLM uses flash_attention_2 by default
    # so we need to set it to None if sequence packing is disabled
    # See https://github.com/NVIDIA-NeMo/Automodel/blob/7e748be260651349307862426c0c168cebdeeec3/nemo_automodel/components/_transformers/auto_model.py#L180
    cp_size_cfg = config["dtensor_cfg"]["context_parallel_size"]
    attn_impl = (
        "flash_attention_2"
        if (enable_seq_packing and cp_size_cfg == 1)
        else ("sdpa" if cp_size_cfg > 1 else None)
    )

    # Load model config
    model_config = AutoConfig.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Always load in float32 for master weights
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if enable_seq_packing else None,
        **hf_config_overrides,
    )

    # Check if model supports flash attention args
    allow_flash_attn_args = True
    if (
        model_config.architectures[0] == "DeciLMForCausalLM"
        and model_config.model_type == "nemotron-nas"
    ):
        allow_flash_attn_args = False

    # Determine if reward model
    is_reward_model = (
        "reward_model_cfg" in config and config["reward_model_cfg"]["enabled"]
    )

    if is_reward_model:
        # Validate reward model configuration
        if enable_seq_packing:
            raise NotImplementedError(
                "Sequence packing is not supported for reward models"
            )

        rm_type = config["reward_model_cfg"]["reward_model_type"]
        if rm_type == "bradley_terry":
            model_class = NeMoAutoModelForSequenceClassification
            if model_config.num_labels != 1:
                print(
                    "model_config.num_labels is not 1. Setting it to 1 since this value is used as the out_features "
                    "for the linear head of Bradley-Terry reward models."
                )
                model_config.num_labels = 1
        else:
            raise ValueError(f"Unknown reward model type: {rm_type}")
    else:
        model_class = resolve_model_class(model_config.model_type)

    # Get parallelization sizes
    tp_size = config["dtensor_cfg"].get("tensor_parallel_size", 1)
    cp_size = config["dtensor_cfg"].get("context_parallel_size", 1)
    sequence_parallel_enabled = config["dtensor_cfg"]["sequence_parallel"]

    # Validate parallelization configuration
    if cp_size > 1 and enable_seq_packing:
        raise ValueError(
            "Context parallel is not supported for sequence packing. "
            "Refer to https://github.com/NVIDIA/NeMo-RL/blob/main/docs/model-quirks.md#context-parallel-with-fsdp2 for more details."
        )

    if sequence_parallel_enabled and tp_size == 1:
        print(
            "[WARNING]: sequence_parallel=True, but tp_size=1 which has no effect. "
            "Enable tp_size > 1 to use sequence parallelism."
        )

    return RuntimeConfig(
        model_class=model_class,
        model_config=model_config,
        hf_config_overrides=hf_config_overrides,
        allow_flash_attn_args=allow_flash_attn_args,
        attn_impl=attn_impl,
        dtype=dtype,
        enable_seq_packing=enable_seq_packing,
        max_grad_norm=max_grad_norm,
        cpu_offload=cpu_offload,
        offload_optimizer_for_logprob=offload_optimizer_for_logprob,
        is_generation_colocated=is_generation_colocated,
        sampling_params=sampling_params,
        is_reward_model=is_reward_model,
    )


def setup_reference_model_state(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Set up reference model state dict by creating a CPU copy of the model's state dict.

    This creates a reference copy of the model weights on CPU with pinned memory
    for efficient CPU-GPU transfers. The reference model is typically used to
    compute reference log probabilities during RL training.

    Args:
        model: The model to create a reference copy from

    Returns:
        Dictionary mapping parameter names to CPU tensors with pinned memory

    Example:
        >>> model = setup_model(...)
        >>> reference_model_state_dict = setup_reference_model_state(model)
    """
    return get_cpu_state_dict(model.state_dict().items(), pin_memory=True)


def setup_distributed(
    config: PolicyConfig,
    runtime_config: RuntimeConfig,
) -> DistributedContext:
    """Set up distributed training environment and create device meshes.

    Initializes torch.distributed process group and creates FSDP2Config,
    MoEParallelizerConfig, and device meshes for distributed training.

    Args:
        config: Policy configuration dictionary
        runtime_config: RuntimeConfig named tuple from validate_and_prepare_config

    Returns:
        DistributedContext containing device meshes and distributed configuration
    """
    # Initialize process group
    backend = "nccl" if not runtime_config.cpu_offload else "cuda:nccl,cpu:gloo"
    torch.distributed.init_process_group(backend=backend)
    world_size = torch.distributed.get_world_size()

    # Extract configuration values
    dtype = runtime_config.dtype
    cpu_offload = runtime_config.cpu_offload

    # Extract parallelization config
    tp_size = config["dtensor_cfg"].get("tensor_parallel_size", 1)
    cp_size = config["dtensor_cfg"].get("context_parallel_size", 1)
    ep_size = config["dtensor_cfg"].get("expert_parallel_size", 1)
    sequence_parallel_enabled = config["dtensor_cfg"]["sequence_parallel"]

    # Build tp_plan from custom_parallel_plan config if set, else None (auto-select)
    tp_plan = config["dtensor_cfg"].get("custom_parallel_plan", None)
    # TODO: improve this
    # Automodel's own resolution of a custom_parallel_plan import-path string
    # (nemo_automodel...parallelizer.py's `_get_parallel_plan`) always calls
    # it as a zero-arg function (`plan_obj()`) -- there's no mechanism to
    # forward `sequence_parallel` through that generic path. Special-case
    # our own function here and pre-resolve to a literal dict (the other
    # branch `_get_parallel_plan` already supports) so SP actually reaches
    # it instead of always silently getting the `sequence_parallel=False`
    # default.
    if (
        tp_plan
        == f"{get_domyn_edge_tp_plan.__module__}.{get_domyn_edge_tp_plan.__qualname__}"
    ):
        tp_plan = get_domyn_edge_tp_plan(sequence_parallel=sequence_parallel_enabled)

    # dtype each FSDP2-wrapped module's *output* gets cast to, on top of
    # param_dtype's compute dtype -- YAML-configurable: None is the
    # recommended default, no forced cast. The places that actually need fp32
    # (log_softmax/loss computation) already upcast explicitly themselves.
    output_dtype_cfg = config["dtensor_cfg"].get("output_dtype", None)
    if isinstance(output_dtype_cfg, str):
        output_dtype_cfg = dtype_from_str(output_dtype_cfg, default=None)

    # Create FSDP2Config
    fsdp2_config = FSDP2Config(
        sequence_parallel=sequence_parallel_enabled,
        tp_plan=tp_plan,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            output_dtype=output_dtype_cfg,
        ),
        offload_policy=CPUOffloadPolicy(pin_memory=False) if cpu_offload else None,
        activation_checkpointing=config["dtensor_cfg"]["activation_checkpointing"],
        defer_fsdp_grad_sync=config["dtensor_cfg"].get("defer_fsdp_grad_sync", True),
        backend="nccl",
    )

    # Create MoEParallelizerConfig from nested moe_parallelizer options
    moe_parallelizer_cfg = config["dtensor_cfg"].get("moe_parallelizer", {})
    moe_config = MoEParallelizerConfig(**moe_parallelizer_cfg)

    # Handle world_size=1 + cpu_offload
    if world_size == 1 and cpu_offload:
        raise NotImplementedError(
            "CPUOffload doesn't work on single GPU for AutoModel. "
            "If you need this feature, please file an issue on https://github.com/NVIDIA-NeMo/Automodel."
        )

    # Create device meshes (dp_size is derived from world_size / (tp * cp * ep))
    device_mesh, moe_mesh = create_device_mesh(
        fsdp2_config,
        tp_size=tp_size,
        pp_size=1,
        cp_size=cp_size,
        ep_size=ep_size,
        world_size=world_size,
    )

    # Derive sizes from mesh
    resolved_dp_size = device_mesh["dp"].size()
    resolved_tp_size = device_mesh["tp"].size()
    resolved_cp_size = device_mesh["cp"].size()

    return DistributedContext(
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
        fsdp2_config=fsdp2_config,
        moe_config=moe_config,
        dp_size=resolved_dp_size,
        tp_size=resolved_tp_size,
        cp_size=resolved_cp_size,
    )


def setup_model_and_optimizer(
    config: PolicyConfig,
    tokenizer: AutoTokenizer,
    runtime_config: RuntimeConfig,
    distributed_context: DistributedContext,
    checkpoint_manager: Any,
    is_vlm: bool = False,
    init_optimizer: bool = True,
    weights_path: Optional[str] = None,
    optimizer_path: Optional[str] = None,
) -> ModelAndOptimizerState:
    """Set up model, parallelization, and optimizer.

    Creates the model via from_pretrained() which handles meta device init,
    parallelization (FSDP2/TP/CP/EP), LoRA, and base weight loading internally.

    Args:
        config: Policy configuration dictionary
        tokenizer: Tokenizer for the model
        runtime_config: RuntimeConfig named tuple from validate_and_prepare_config
        distributed_context: DistributedContext from setup_distributed
        checkpoint_manager: Checkpoint manager for loading/saving weights
        is_vlm: Whether this is a vision-language model
        init_optimizer: Whether to initialize optimizer
        weights_path: Optional path to checkpoint weights to load
        optimizer_path: Optional path to optimizer state to load

    Returns:
        ModelAndOptimizerState containing model, optimizer, scheduler, and metadata
    """
    # Extract configuration values
    model_config = runtime_config.model_config
    model_class = runtime_config.model_class
    attn_impl = runtime_config.attn_impl
    cpu_offload = runtime_config.cpu_offload
    is_reward_model = runtime_config.is_reward_model

    # Extract distributed configuration from context
    rank = torch.distributed.get_rank()
    device_mesh = distributed_context.device_mesh
    moe_mesh = distributed_context.moe_mesh
    fsdp2_config = distributed_context.fsdp2_config
    moe_config = distributed_context.moe_config
    tp_size = distributed_context.tp_size
    cp_size = distributed_context.cp_size
    sequence_parallel_enabled = fsdp2_config.sequence_parallel
    ep_size = config["dtensor_cfg"].get("expert_parallel_size", 1)

    model_name = config["model_name"]

    # Validate CP configuration with model type before from_pretrained
    if cp_size > 1:
        if model_config.model_type == "gemma3":
            raise AssertionError(
                "Context parallel is not supported for Gemma3ForCausalLM. "
                "Torch context parallel has many limitations. "
                "Please refer to https://github.com/NVIDIA/NeMo-RL/blob/main/docs/model-quirks.md#context-parallel-with-fsdp2 for more details."
            )

        if tp_size > 1 and sequence_parallel_enabled:
            raise AssertionError(
                "It's a known issue that context parallel can't be used together with sequence parallel in DTensor worker. "
                "Please either set cp_size = 1 or disable sequence parallel. "
                "See https://github.com/NVIDIA-NeMo/RL/issues/659 for more details."
            )

        if is_vlm:
            raise AssertionError(
                "Context parallel is yet not supported for VLM models. Please set cp_size = 1 to train VLM models."
            )

        if model_config.model_type == "qwen3_5":
            raise AssertionError(
                "Context parallel is not supported for Qwen3.5 dense models (only torch attention backend is available). "
                "Please set cp_size = 1. For Qwen3.5 MoE models, CP is supported with the TE backend."
            )

        if model_config.model_type == "qwen3_5_moe":
            try:
                import fla  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Qwen3.5 MoE requires flash-linear-attention for context parallel. "
                    "Please install it in your Automodel venv: pip install flash-linear-attention"
                )

    # LoRA configuration
    lora_cfg = config["dtensor_cfg"].get("lora_cfg", None)
    peft_config = None
    lora_enabled = lora_cfg is not None and lora_cfg["enabled"]
    if lora_enabled:
        if tp_size > 1:
            assert not lora_cfg["use_triton"], (
                "Triton is not supported when tensor_parallel_size > 1"
            )
        # Always use float32 since FSDP requires all parameters to be in the same dtype
        cfg_dict_with_dtype = {**lora_cfg, "lora_dtype": "torch.float32"}
        peft_config = PeftConfig.from_dict(cfg_dict_with_dtype)

    print(f"[Rank {rank}] Initializing model via from_pretrained...")

    # Prepare automodel kwargs
    automodel_kwargs = config["dtensor_cfg"].get("automodel_kwargs", {})
    if automodel_kwargs.get("backend", None) is not None:
        backend_class = _resolve_target(
            automodel_kwargs.get("backend", None)["_target_"]
        )
        backend_kwargs = automodel_kwargs.get("backend")
        backend_kwargs.pop("_target_")
        backend = backend_class(**backend_kwargs)
        automodel_kwargs["backend"] = backend

    if "use_liger_kernel" not in automodel_kwargs:
        automodel_kwargs["use_liger_kernel"] = False

    # Determine SDPA method for activation checkpointing and CP
    from torch.nn.attention import SDPBackend

    if cp_size > 1:
        # Match Automodel's `get_train_context` in `cp_utils.py` where only
        # flash and efficient backends are supported
        sdpa_method = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
        ]
    elif config["dtensor_cfg"]["activation_checkpointing"]:
        # For activation checkpointing, we must disable the cudnn SDPA backend because
        # it may not be selected during recomputation.
        # In that case, we will get the following error:
        # "Recomputed values have different metadata than during forward pass."
        sdpa_method = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    else:
        sdpa_method = None

    # For activation checkpointing, we also must globally disable the cudnn SDPA backend
    # to ensure that cudnn does not get selected during recomputation.
    if config["dtensor_cfg"]["activation_checkpointing"]:
        from torch.backends import cuda

        cuda.enable_cudnn_sdp(False)

    # Build from_pretrained kwargs from hf_config_overrides so from_pretrained
    # applies them when loading the config internally (avoids passing config=
    # which causes duplicate 'config' arg for custom model implementations).
    hf_config_overrides = runtime_config.hf_config_overrides or {}
    from_pretrained_kwargs: dict[str, Any] = {
        **hf_config_overrides,
    }
    # Reward model num_labels override
    if is_reward_model and model_config.num_labels == 1:
        from_pretrained_kwargs["num_labels"] = 1

    # Auto-set force_hf if the custom model's adapter doesn't support per-tensor
    # HF conversion (required for weight syncing).
    _maybe_set_force_hf(automodel_kwargs, model_config)

    # Must run before from_pretrained, which applies the TP plan (and the
    # head-count patch this gates) internally.
    _maybe_patch_domyn_edge_attention_head_sharding(model_config)

    # Create model via from_pretrained - handles meta device init, parallelization,
    # LoRA, and base weight loading internally
    model = model_class.from_pretrained(
        model_name,
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
        distributed_config=fsdp2_config,
        moe_config=moe_config if ep_size > 1 else None,
        activation_checkpointing=config["dtensor_cfg"]["activation_checkpointing"],
        peft_config=peft_config,
        attn_implementation=attn_impl,
        torch_dtype=str(model_config.torch_dtype),
        trust_remote_code=True,
        sdpa_method=sdpa_method,
        **from_pretrained_kwargs,
        **automodel_kwargs,
    )

    # Width of the vocab the training loss covered
    trained_vocab_size = config.get("trained_vocab_size")
    if trained_vocab_size is not None:
        model.config.trained_vocab_size = trained_vocab_size

    _maybe_patch_domyn_edge_varlen_attn(model_config)
    _maybe_patch_domyn_edge_rope_autocast(model)
    _maybe_patch_domyn_edge_skip_lm_head_support(model)
    _maybe_patch_domyn_edge_activation_checkpointing(
        model, config["dtensor_cfg"]["activation_checkpointing"]
    )

    print(model)

    # Compute model metadata after from_pretrained
    model_state_dict_keys = list(model.state_dict().keys())
    is_moe_model = any(["expert" in key for key in model_state_dict_keys])
    is_hf_model = (
        model_config.architectures[0] not in ModelRegistry.model_arch_name_to_cls
    )
    # Autocast is disabled for custom MoE models (non-HF) to avoid numerical issues
    autocast_enabled = not (is_moe_model and not is_hf_model)

    # Set pad token ID if needed. Some model configs (e.g. Gemma3 in transformers v5)
    # don't have pad_token_id as a direct attribute.
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Handle tied word embeddings (safety net after from_pretrained)
    is_tied_lm_head = hasattr(model, "lm_head") and getattr(
        getattr(model, "config", {}), "tie_word_embeddings", False
    )
    if is_tied_lm_head:
        model.tie_weights()

    # Freeze visual encoder when not doing VLM training.
    # Without this, the optimizer creates state entries for visual params that never
    # receive gradients, causing a key mismatch when resuming from checkpoint.
    # Note: visual encoder is nested under model.model (e.g. model.model.visual for
    # Qwen3_5MoeForConditionalGeneration), not directly on model.
    visual_module = getattr(getattr(model, "model", None), "visual", None) or getattr(
        model, "visual", None
    )
    if not is_vlm and visual_module is not None:
        for param in visual_module.parameters():
            param.requires_grad_(False)
        if rank == 0:
            print("Froze visual encoder parameters for text-only training")

    # CPU offload if needed
    if cpu_offload:
        # Move buffers to CPU for FSDP modules
        for v in model.buffers():
            v.data = v.data.to("cpu")
        model = model.to("cpu")

    # Initialize optimizer
    optimizer = None
    if init_optimizer:
        optimizer_cls = get_class(config["optimizer"]["name"])
        optimizer = optimizer_cls(model.parameters(), **config["optimizer"]["kwargs"])

    # Initialize scheduler
    scheduler = None
    if "scheduler" in config and optimizer is not None:
        if isinstance(config["scheduler"], dict):
            scheduler_cls = get_class(config["scheduler"]["name"])
            scheduler = scheduler_cls(optimizer, **config["scheduler"]["kwargs"])
        else:
            schedulers = []
            for scheduler_cfg in config["scheduler"]:
                if "name" in scheduler_cfg:
                    schedulers.append(
                        get_class(scheduler_cfg["name"])(
                            optimizer, **scheduler_cfg["kwargs"]
                        )
                    )
                else:
                    assert "milestones" in scheduler_cfg, (
                        "unknown scheduler config: ",
                        scheduler_cfg,
                    )
                    milestones: list[int] = scheduler_cfg["milestones"]

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers, milestones
            )
    elif optimizer is not None:
        # Default to passthrough LR schedule
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1
        )

    # Load NeMo RL checkpoint if provided
    if weights_path:
        checkpoint_manager.load_checkpoint(
            model=model,
            weights_path=weights_path,
            optimizer=optimizer,
            optimizer_path=optimizer_path,
            scheduler=scheduler,
        )
    else:
        print(
            "No weights path provided. Loaded base HF weights via from_pretrained (default policy init)"
        )

    return ModelAndOptimizerState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        is_hf_model=is_hf_model,
        is_moe_model=is_moe_model,
        is_reward_model=is_reward_model,
        model_class=type(model),
        model_config=model.config,
        peft_config=peft_config,
        autocast_enabled=autocast_enabled,
    )
