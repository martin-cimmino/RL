# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# bare-metal: vendored from the `domyn-edge-leonardo` branch of
# https://github.com/igeniusai/vllm-domyn-edge (a fork of vllm-project/vllm),
# which never got merged into the branch our vllm wheel is actually built
# from. Registered at runtime via vllm.ModelRegistry / _CONFIG_REGISTRY (see
# register_domyn_edge_plugin() in vllm_worker.py) instead of patching the
# installed vllm package, so this survives normal venv rebuilds. See the
# project_domynedge_vllm_plugin memory for the full story.
"""Inference-only DomynEdge model compatible with HuggingFace weights."""

import collections

import torch

import vllm
import vllm.compilation.decorators
import vllm.config
import vllm.model_executor.layers.attention
import vllm.model_executor.layers.layernorm
import vllm.model_executor.layers.linear
import vllm.model_executor.layers.rotary_embedding
import vllm.model_executor.models.interfaces
import vllm.model_executor.models.qwen2
import vllm.v1.attention.backend


class DomynEdgeAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        head_dim: int,
        max_position: int = 4096 * 32,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: vllm.config.CacheConfig | None = None,
        quant_config: vllm.model_executor.layers.quantization.QuantizationConfig | None = None,
        window_size: int = 0,
        prefix: str = "",
    ) -> None:

        super().__init__()
        self.hidden_size = hidden_size
        tp_size = vllm.distributed.get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0

        self.qkv_proj = vllm.model_executor.layers.linear.QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = vllm.model_executor.layers.linear.RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = (
            vllm.model_executor.layers.rotary_embedding.get_rope(
                self.head_dim,
                max_position=max_position,
                rope_parameters=rope_parameters,
                dtype=torch.float,
            )
            if rope_parameters is not None
            else None
        )

        self.attn = vllm.model_executor.layers.attention.Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=vllm.v1.attention.backend.AttentionType.DECODER,
            # The HF reference (modeling_domynedge.py) calls varlen_attn with
            # window_size=(w, 0): query i attends to keys [i-w, i], i.e. w+1
            # tokens. vLLM's per_layer_sliding_window is the total window size
            # (it maps to flash-attn's (N-1, 0)), so add 1 to match the
            # reference. window_size <= 0 means full (non-sliding) attention.
            per_layer_sliding_window=window_size + 1 if window_size > 0 else None,
        )
        self.q_norm = vllm.model_executor.layers.layernorm.RMSNorm(self.head_dim, eps=rms_norm_eps, has_weight=False)
        self.k_norm = vllm.model_executor.layers.layernorm.RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q.float(), k.float())
            q = q.type_as(hidden_states)
            k = k.type_as(hidden_states)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class DomynEdgeDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        config,
        cache_config: vllm.config.CacheConfig | None = None,
        quant_config: vllm.model_executor.layers.quantization.QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        match config.position_layers[vllm.model_executor.models.utils.extract_layer_index(prefix)]:
            case "rope":
                rope_parameters = config.rope_parameters
            case "nope":
                rope_parameters = None
            case _:
                raise ValueError(
                    f"Unsupported position encoding type for layer {vllm.model_executor.models.utils.extract_layer_index(prefix)}: {config.position_layers[vllm.model_executor.models.utils.extract_layer_index(prefix)]}"
                )

        self.attention = DomynEdgeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=False,
            head_dim=config.head_sizes[vllm.model_executor.models.utils.extract_layer_index(prefix)],
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=rope_parameters,
            window_size=config.window_sizes[vllm.model_executor.models.utils.extract_layer_index(prefix)][0],
            prefix=f"{prefix}.attention",
        )
        self.feedforward = vllm.model_executor.models.qwen2.Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.pre_attention_norm = vllm.model_executor.layers.layernorm.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_norm = vllm.model_executor.layers.layernorm.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_norm = vllm.model_executor.layers.layernorm.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_norm = vllm.model_executor.layers.layernorm.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_attention_norm(hidden_states)
        else:
            hidden_states, residual = self.pre_attention_norm(hidden_states, residual)
        hidden_states = self.post_attention_norm(
            self.attention(
                positions=positions,
                hidden_states=hidden_states,
            )
        )

        # Fully Connected
        hidden_states, residual = self.pre_feedforward_norm(hidden_states, residual)
        hidden_states = self.post_feedforward_norm(self.feedforward(hidden_states))
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": DomynEdgeDecoderLayer,
}


@vllm.compilation.decorators.support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class DomynEdgeModel(vllm.model_executor.models.qwen2.Qwen2Model):
    def __init__(self, *, vllm_config: vllm.config.VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix, decoder_layer_type=DomynEdgeDecoderLayer)


class DomynEdgeForCausalLM(
    torch.nn.Module,
    vllm.model_executor.models.interfaces.SupportsLoRA,
    vllm.model_executor.models.interfaces.SupportsPP,
    vllm.model_executor.models.interfaces.SupportsEagle,
    vllm.model_executor.models.interfaces.SupportsEagle3,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: vllm.config.VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = DomynEdgeModel(
            vllm_config=vllm_config, prefix=vllm.model_executor.models.utils.maybe_prefix(prefix, "model")
        )

        if vllm.distributed.get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=vllm.model_executor.models.utils.maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = vllm.model_executor.models.utils.PPMissingLayer()

        self.logits_processor = vllm.model_executor.layers.logits_processor.LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: vllm.sequence.IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | vllm.sequence.IntermediateTensors:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: collections.abc.Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = vllm.model_executor.models.utils.AutoWeightsLoader(self)
        return loader.load_weights(weights)
