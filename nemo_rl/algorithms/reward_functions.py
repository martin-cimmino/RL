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
from typing import (
    NotRequired,
    TypedDict,
    TypeVar,
)

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class RewardShapingConfig(TypedDict):
    """Configuration for reward function processing.

    This configuration enables custom reward shaping, currently supporting DAPO-style
    penalties for responses that exceed the maximum response length threshold.
    """

    enabled: bool

    # The length of the buffer to penalize responses that exceed the maximum response length threshold.
    # Responses of length greater than overlong_buffer_length + max_response_length will
    # receive the maximum penalty.
    overlong_buffer_length: NotRequired[int]

    # The penalty for responses that exceed the maximum response length threshold.
    overlong_buffer_penalty: NotRequired[float]

    # The maximum response length threshold. Responses exceeding this length will be penalized.
    max_response_length: NotRequired[int]

    # Stop properly penalty: scale factor for rewards of truncated responses (0-1).
    # When set to 0, truncated responses get zero reward.
    # When set to 1, no penalty is applied (default behavior).
    stop_properly_penalty_coef: NotRequired[float | None]


class DegenerationPenaltyConfig(TypedDict):
    """[CUSTOM] Penalty for responses that degenerate into word salad.

    Word salad shows up as a burst of tokens the generator itself found very
    unlikely: a response is flagged when some window of `window_tokens`
    generated tokens holds at least `min_low_logprob_tokens` tokens whose
    generation logprob is below `logprob_threshold`. Flagged responses get
    reward min(reward, penalty_reward). Repetition loops (near-zero surprisal)
    are not detected; they run to the length cap and get the overlong penalty.
    """

    enabled: bool
    # Generation logprob below which a sampled token counts as unlikely (e.g. -10).
    logprob_threshold: float
    # Sliding-window size in generated tokens (e.g. 2000).
    window_tokens: int
    # Unlikely tokens within one window that flag the response (e.g. 50; healthy
    # code-teacher responses peak at ~15 per 2000 tokens at the 99th percentile).
    min_low_logprob_tokens: int
    # Reward a flagged response is capped at (e.g. -1.0).
    penalty_reward: float


def apply_reward_shaping(
    batch: BatchedDataDict, cfg: RewardShapingConfig
) -> BatchedDataDict:
    """Process rewards by applying penalties for responses exceeding max_response_length. Currently, this function only supports DAPO reward shaping as illustrated in the DAPO paper : https://arxiv.org/pdf/2503.14476.

    Nonetheless, it can be potentially extended to support any custom reward logic.
    """
    rewards = batch["total_reward"]
    if not cfg["enabled"]:
        return batch

    # Apply stop properly penalty if configured
    stop_properly_penalty_coef = cfg.get("stop_properly_penalty_coef", None)
    if stop_properly_penalty_coef is not None:
        assert 0 <= stop_properly_penalty_coef <= 1, (
            f"stop_properly_penalty_coef must be in [0, 1], got {stop_properly_penalty_coef}"
        )
        # Warn user that DAPO overlong parameters are ignored when stop_properly_penalty_coef is set
        ignored_params = []
        if cfg.get("overlong_buffer_length") is not None:
            ignored_params.append("overlong_buffer_length")
        if cfg.get("overlong_buffer_penalty") is not None:
            ignored_params.append("overlong_buffer_penalty")
        if cfg.get("max_response_length") is not None:
            ignored_params.append("max_response_length")
        if ignored_params:
            print(
                f"[WARN] stop_properly_penalty_coef is set, so the following DAPO overlong "
                f"parameters are ignored: {', '.join(ignored_params)}. "
                f"Set stop_properly_penalty_coef=null to use DAPO overlong reward shaping instead.",
                flush=True,
            )
        truncated = batch.get("truncated")
        assert truncated is not None, "truncated field not found in batch"
        if isinstance(truncated, list):
            truncated = torch.tensor(truncated, dtype=torch.bool, device=rewards.device)
        else:
            truncated = truncated.to(device=rewards.device)

        num_truncated = truncated.sum().item()
        if num_truncated > 0:
            original_rewards = rewards.clone()
            # For truncated samples, scale the reward by stop_properly_penalty_coef
            rewards = torch.where(
                truncated, rewards * stop_properly_penalty_coef, rewards
            )
            batch["total_reward"] = rewards
            print(
                f"[INFO] stop properly penalty applied: {num_truncated}/{len(truncated)} samples truncated, "
                f"coef={stop_properly_penalty_coef}, "
                f"original_reward_mean={original_rewards[truncated].mean().item():.4f}, "
                f"shaped_reward_mean={rewards[truncated].mean().item():.4f}",
                flush=True,
            )
        else:
            print(
                "[INFO] stop properly penalty: no truncated samples (truncation_rate=0)",
                flush=True,
            )

        return batch

    # DAPO reward shaping requires overlong_buffer_length, overlong_buffer_penalty, and max_response_length to be set.
    if (
        cfg.get("overlong_buffer_length") is None
        or cfg.get("overlong_buffer_penalty") is None
        or cfg.get("max_response_length") is None
    ):
        raise ValueError(
            "Reward function is enabled but only DAPO reward shaping is currently supported. Please ensure overlong_buffer_length, overlong_buffer_penalty, and max_response_length are properly configured."
        )

    # Get the overlong_buffer_length, overlong_buffer_penalty and max_response_length
    overlong_buffer_length = cfg["overlong_buffer_length"]
    overlong_buffer_penalty = cfg["overlong_buffer_penalty"]
    max_response_length = cfg["max_response_length"]
    assert overlong_buffer_penalty >= 0, f"{overlong_buffer_penalty=} must be >=0"
    # Calculate the expected response length
    expected_response_length = max_response_length - overlong_buffer_length

    assert len(batch["message_log"]) == len(rewards), (
        "The number of messages in the batch must match the number of rewards"
    )

    # [CUSTOM-PORTED 2026-11-02] Hand-ported from upstream commit e12d18ba8
    # "fix(training): exclude prompt history from reward lengths and loss masks"
    # (NVIDIA-NeMo/RL PR #4487, merged 2026-10-18) -- not yet in our fork's base
    # (c8d167a73, pre-v0.6.0). Hand-ported rather than cherry-picked: the full PR
    # also touches ppo.py/sync_rollout_actor.py/decompose_message_log, none of
    # which exist on our branch (we don't use upstream's PPO path or its data-plane
    # transfer queue), so a straight cherry-pick produced spurious modify/delete
    # conflicts on files we never had. Only this function's fix applies to us.
    #
    # Original bug: measured only the FIRST "assistant"-role message in
    # message_log and used its length alone -- correct for upstream's assumed
    # single-turn layout (message_log == [prompt, response]), silently wrong for
    # our multi-turn swe_agents/NeMo-Gym rollouts (break on turn 1, ignoring every
    # later turn, including the one that actually runs long/degenerates).
    # Fix: sum token_ids across every assistant message that carries
    # "generation_logprobs" (i.e. was actually generated this rollout, not part
    # of prompt history) -- confirmed this key is already set by rollouts.py at
    # generation time and only backfilled with zeros later in grpo.py (after this
    # function runs), so the presence check correctly separates real generations
    # from prompt-history turns in our pipeline too, not just upstream's.
    #
    # Safe to delete this patch (and this comment) once this fork's RL base moves
    # past e12d18ba8 -- diff this function against upstream at that point first to
    # confirm it already matches before removing.
    updated_rewards = torch.zeros_like(rewards)
    for i, message_log in enumerate(batch["message_log"]):
        assistant_lengths = [
            message["token_ids"].shape[0]
            for message in message_log
            if message["role"] == "assistant" and "generation_logprobs" in message
        ]
        assert assistant_lengths, (
            "Assistant response not found during reward shaping"
        )
        message_response_length = sum(assistant_lengths)

        # Calculate the exceed length and the corresponding reward penalty
        exceed_length = message_response_length - expected_response_length
        overlong_reward = min(
            -exceed_length / overlong_buffer_length * overlong_buffer_penalty, 0
        )
        updated_rewards[i] = rewards[i] + overlong_reward

    # Update the rewards in the batch
    batch["total_reward"] = updated_rewards

    return batch


def max_low_logprob_tokens_in_window(
    generation_logprobs: torch.Tensor, logprob_threshold: float, window_tokens: int
) -> int:
    """Largest number of tokens below `logprob_threshold` in any window of `window_tokens` tokens."""
    low = (generation_logprobs.float() < logprob_threshold).to(torch.int64)
    if low.numel() <= window_tokens:
        return int(low.sum())
    csum = torch.cat([low.new_zeros(1), low.cumsum(0)])
    return int((csum[window_tokens:] - csum[:-window_tokens]).max())


def apply_degeneration_penalty(
    batch: BatchedDataDict, cfg: DegenerationPenaltyConfig
) -> BatchedDataDict:
    """[CUSTOM] Cap the reward of word-salad responses at cfg["penalty_reward"].

    Needs the generators' per-token logprobs ("generation_logprobs") on the
    assistant messages, which both the vLLM and the NeMo-Gym rollouts provide.
    See DegenerationPenaltyConfig for the detection rule.
    """
    if not cfg["enabled"]:
        return batch
    rewards = batch["total_reward"]
    assert len(batch["message_log"]) == len(rewards), (
        "The number of messages in the batch must match the number of rewards"
    )
    flagged = torch.zeros_like(rewards, dtype=torch.bool)
    for i, message_log in enumerate(batch["message_log"]):
        logprobs = [
            m["generation_logprobs"]
            for m in message_log
            if m["role"] == "assistant" and "generation_logprobs" in m
        ]
        assert logprobs, "Assistant generation_logprobs not found during degeneration check"
        worst = max_low_logprob_tokens_in_window(
            torch.cat([torch.as_tensor(lp).flatten() for lp in logprobs]),
            cfg["logprob_threshold"],
            cfg["window_tokens"],
        )
        flagged[i] = worst >= cfg["min_low_logprob_tokens"]
    if flagged.any():
        penalty = torch.full_like(rewards, cfg["penalty_reward"])
        batch["total_reward"] = torch.where(
            flagged, torch.minimum(rewards, penalty), rewards
        )
        print(
            f"[INFO] degeneration penalty: {int(flagged.sum())}/{len(rewards)} responses flagged, "
            f"original rewards {[round(float(r), 3) for r in rewards[flagged]]}",
            flush=True,
        )
    return batch
