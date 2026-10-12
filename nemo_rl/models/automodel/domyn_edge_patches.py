# [CUSTOM] This entire file is DomynEdge-specific Automodel integration code, vendored from
# the bare-metal branch (see also nemo_rl/models/generation/vllm/domyn_edge{,_config}.py for
# the vLLM-side counterpart). Kept in its own module -- rather than inline in setup.py -- so
# the diff against upstream automodel/setup.py stays limited to the small set of call sites
# that invoke these functions, instead of ~400 lines of insertions scattered through it. Every
# public function here is a no-op for any model whose config.architectures[0] is not
# "DomynEdgeForCausalLM", so importing/calling them unconditionally from setup.py and train.py
# is safe for all other models.
"""DomynEdge-specific monkeypatches and tensor-parallel plan for Automodel training.

See docs/infrastructure/engineering-notes (bare-metal branch) and the
project_domynedge_* memory entries for the full history of each fix below.
"""

import torch
import torch.utils.checkpoint


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
    of matmul, so under a bf16 autocast region this product is evaluated in bf16. The RoPE angles are therefore quantized progressively harder the deeper into a sequence you go, and the model's attention degrades with position.

    Upstream transformers guards against exactly this; see modeling_llama.py, which wraps the same three lines in
    `maybe_autocast(device_type=..., enabled=False)  # Force float32`. DomynEdge's trust_remote_code modeling file simply omits that guard, so the
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
    our pinned torch's actual varlen_attn signature (no scale/window_size/
    enable_gqa at all) — a checkpoint/torch-version mismatch, not something to
    fix in this repo. Routes those calls to flash_attn.flash_attn_varlen_func
    instead, restoring the causal=True the checkpoint code never passes. No-op
    for every other model.
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


def _maybe_patch_domyn_edge_packed_attention(model) -> None:
    """Makes DomynEdge's attention honour flash_attn_kwargs, enabling sequence packing.

    Without this, the packed row is attended as ONE contiguous sequence and tokens
    from one rollout attend causally to a different rollout's tokens.
    """
    architectures = getattr(getattr(model, "config", None), "architectures", None) or []
    if not architectures or architectures[0] != "DomynEdgeForCausalLM":
        return

    layers = model.model.layers
    if isinstance(layers, torch.nn.ModuleDict):
        layers = list(layers.values())
    layers = list(layers)
    if not layers:
        return
    cls = type(layers[0].attention)
    if getattr(cls, "_packed_attention_patched", False):
        return

    original_forward = cls.forward

    def _packed_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        flash_attn_kwargs = kwargs.pop("flash_attn_kwargs", None)
        if flash_attn_kwargs is None:
            return original_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )

        if hasattr(flash_attn_kwargs, "cu_seqlens_q"):
            cu_source = flash_attn_kwargs.cu_seqlens_q
        else:
            cu_source = flash_attn_kwargs["cu_seqlens_q"]

        import torch.nn.attention.varlen as _varlen

        current = _varlen.varlen_attn

        def _segmented_varlen_attn(
            query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, *args, **kw
        ):
            total = query.shape[0]
            cu = cu_source.to(device=query.device, dtype=torch.int32)
            covered = int(cu[-1].item())
            assert covered <= total, (
                f"cu_seqlens cover {covered} tokens but the attention input has "
                f"only {total}. The packed row and flash_attn_kwargs disagree."
            )
            if covered < total:
                cu = torch.cat(
                    [
                        cu,
                        torch.tensor([total], device=cu.device, dtype=torch.int32),
                    ]
                )
            max_seqlen = int((cu[1:] - cu[:-1]).max().item())
            return current(
                query, key, value, cu, cu, max_seqlen, max_seqlen, *args, **kw
            )

        _varlen.varlen_attn = _segmented_varlen_attn
        try:
            # attention_mask is passed through untouched.
            return original_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )
        finally:
            _varlen.varlen_attn = current

    cls.forward = _packed_forward
    cls._packed_attention_patched = True


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
