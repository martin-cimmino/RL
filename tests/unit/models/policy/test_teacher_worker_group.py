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

# [CUSTOM] Tests for the DTensor non-colocated teacher backend (see
# nemo_rl/models/policy/teacher_worker_group.py) -- a fork-specific extension,
# upstream teacher workers are Megatron-only.


def test_teacher_resource_config_defaults():
    from nemo_rl.algorithms.opd import TeacherResourceConfig

    res = TeacherResourceConfig(tensor_model_parallel_size=4)
    assert res.tensor_model_parallel_size == 4
    assert res.pipeline_model_parallel_size == 1
    assert res.gpus_per_node == 8
    assert res.precision == "bf16"


def test_create_teacher_configs_homogeneous():
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    configs = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {"math": "/ckpt/math", "code": "/ckpt/code"},
            "non_colocated_teachers": {
                "default_teacher_cfg": {"tensor_model_parallel_size": 4}
            },
        }
    )
    assert len(configs) == 2
    assert all(c.tensor_model_parallel_size == 4 for c in configs)


def test_create_teacher_configs_heterogeneous_override():
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    configs = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {"math": "/ckpt/math", "code": "/ckpt/code"},
            "non_colocated_teachers": {
                "default_teacher_cfg": {"tensor_model_parallel_size": 4},
                "teacher_overrides": {"code": {"tensor_model_parallel_size": 8}},
            },
        }
    )
    code_cfg = [c for c in configs if c.alias == "code"][0]
    assert code_cfg.tensor_model_parallel_size == 8


def test_create_teacher_configs_deduplicates():
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    configs = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {
                "math": "/shared",
                "code": "/shared",
                "rlhf": "/rlhf",
            },
            "deduplicate_shared_teacher_checkpoints": True,
            "non_colocated_teachers": {
                "default_teacher_cfg": {"tensor_model_parallel_size": 2}
            },
        }
    )
    assert len(configs) == 2


def _dtensor_policy_cfg():
    return {
        "model_name": "/student",
        "precision": "bfloat16",
        "offload_optimizer_for_logprob": True,
        "tokenizer": {"name": "/student", "use_processor": True},
        "dtensor_cfg": {
            "enabled": True,
            "_v2": True,
            "cpu_offload": True,
            "activation_checkpointing": True,
            "tensor_parallel_size": 4,
            "context_parallel_size": 2,
            "lora_cfg": {"enabled": True},
        },
        "megatron_cfg": {"enabled": False},
        "draft": {"enabled": True},
        "quant_cfg": None,
    }


def _bare_teacher_group(alias="math"):
    from nemo_rl.models.policy.teacher_worker_group import TeacherWorkerGroup

    twg = object.__new__(TeacherWorkerGroup)  # skip Ray worker construction
    twg.alias = alias
    twg.model_name = f"/ckpt/{alias}"
    return twg


def test_dtensor_teacher_cfg_follows_student_backend():
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    (tcfg,) = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {"math": "/ckpt/math"},
            "non_colocated_teachers": {
                "default_teacher_cfg": {
                    "tensor_model_parallel_size": 2,
                    "precision": "bf16",
                    "dtensor_cfg_overrides": {"cpu_offload": True},
                }
            },
        }
    )
    policy = _dtensor_policy_cfg()
    cfg, tp, pp, cp = _bare_teacher_group()._build_dtensor_cfg(tcfg, policy)

    assert (tp, pp, cp) == (2, 1, 1)
    assert cfg["model_name"] == "/ckpt/math"
    assert cfg["tokenizer"]["name"] == "/student"  # student's tokenizer is kept
    assert cfg["tokenizer"]["use_processor"] is False
    dt = cfg["dtensor_cfg"]
    assert dt["tensor_parallel_size"] == 2 and dt["context_parallel_size"] == 1
    assert dt["cpu_offload"] is True  # explicit override wins over the default False
    assert dt["activation_checkpointing"] is False
    assert dt["lora_cfg"]["enabled"] is False
    assert cfg["precision"] == "bfloat16"
    assert cfg["offload_optimizer_for_logprob"] is False
    assert cfg["draft"]["enabled"] is False
    # the student's own config must not be mutated
    assert policy["dtensor_cfg"]["tensor_parallel_size"] == 4
    assert policy["dtensor_cfg"]["lora_cfg"]["enabled"] is True


def test_dtensor_teacher_rejects_pipeline_parallel_and_v1():
    import pytest

    from nemo_rl.models.policy.teacher_worker_group import TeacherConfig

    base = dict(
        alias="math",
        model_name="/ckpt/math",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        num_nodes=1,
        gpus_per_node=4,
        precision="bf16",
        micro_batch_size=1,
        megatron_cfg_overrides={},
    )
    twg = _bare_teacher_group()
    with pytest.raises(ValueError, match="pipeline"):
        twg._build_dtensor_cfg(
            TeacherConfig(**{**base, "pipeline_model_parallel_size": 2}),
            _dtensor_policy_cfg(),
        )
    v1 = _dtensor_policy_cfg()
    v1["dtensor_cfg"]["_v2"] = False
    with pytest.raises(ValueError, match="_v2"):
        twg._build_dtensor_cfg(TeacherConfig(**base), v1)


def test_teacher_get_logprobs_dynamic_batching_shards_and_restores_order():
    """With dynamic_batching on (inherited from the student cfg), DTensor teacher workers read
    data.micro_batch_indices; get_logprobs must shard with dynamic_batching_args and undo the
    length sort on the way back (regression: 'NoneType' object is not subscriptable on 32n run)."""
    import torch

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    class _Axes:
        def get_axis_size(self, name):
            return 2

    class _FakeWorkers:
        def __init__(self):
            self.shards = None

        def run_all_workers_sharded_data(self, method, data, **kwargs):
            self.shards = data
            return data

        def get_all_worker_results(self, shards):
            # each "worker" checks its micro-batch indices and returns per-row logprobs = row id
            out = []
            for s in shards:
                assert s.micro_batch_indices is not None
                assert s.get_microbatch_iterator_dynamic_shapes_len() >= 1
                out.append(BatchedDataDict(logprobs=s["row_id"].float().unsqueeze(-1).repeat(1, s["input_ids"].shape[1])))
            return out

    twg = _bare_teacher_group()
    twg.sharding_annotations = _Axes()
    twg._micro_batch_size = 1
    twg.use_dynamic_batches = True
    twg.use_sequence_packing = False
    twg.cfg = {"dynamic_batching": {"logprob_mb_tokens": 64}}
    twg.dynamic_batching_args = {
        "input_key": "input_ids",
        "input_lengths_key": "input_lengths",
        "sequence_length_round": 8,
        "max_tokens_per_microbatch": 64,
    }
    twg.worker_group = _FakeWorkers()

    lengths = torch.tensor([5, 30, 12, 7, 25, 3, 18, 9])
    data = BatchedDataDict(
        input_ids=torch.zeros(8, 32, dtype=torch.long),
        input_lengths=lengths,
        row_id=torch.arange(8),
    )
    res = twg.get_logprobs(data)
    assert len(twg.worker_group.shards) == 2
    # original row order restored: row i carries logprob value i
    assert torch.equal(res["reference_logprobs"][:, 0], torch.arange(8).float())
