# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# [CUSTOM] Adds dynamic sampling (drop zero-advantage prompt groups and let upstream's own
# gap-filling retry relaunch a replacement, DAPO-style) to AsyncTrajectoryCollector -- upstream
# NeMo-RL does not support this on the async GRPO path. See `_process_batch`'s `filterable`
# computation, `_run_prompt_group_worker`'s discard check, and `get_target_stats` for the
# mechanism. Confirmed via `git grep` against both the v0.7.0 tag and current origin/main:
# this feature doesn't exist upstream at all, old or new -- not a temporary gap to wait out.

from __future__ import annotations

import collections
import concurrent.futures
import math
import threading as _threading
import time
from collections import defaultdict
from typing import Any, Optional

import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, scale_rewards
from nemo_rl.algorithms.opd import resolve_reference_aliases, teacher_seq_pad_multiple
from nemo_rl.algorithms.reward_functions import (
    apply_degeneration_penalty,
    apply_reward_shaping,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface

# [CUSTOM-GAPFILL-CURRENT] Backstop re-check period of a paused collection loop; wakes normally
# come from set_weight_version (refit) and _maybe_release_target (a reservation released).
_GENERATION_LIMIT_RECHECK_S = 60.0

TokenizerType = PreTrainedTokenizerBase


@ray.remote  # pragma: no cover
class AsyncTrajectoryCollector:
    """Collects trajectories asynchronously and adds them to replay buffer."""

    def __init__(
        self,
        policy_generation: GenerationInterface,
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        master_config: MasterConfig,
        replay_buffer: Any,
        start_step: int = 0,
        teacher_worker_groups: Optional[dict[str, Any]] = None,
        alias_to_group_alias: Optional[dict[str, str]] = None,
        on_policy_distillation_cfg: Optional[dict[str, Any]] = None,
    ):
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.master_config = master_config
        self.replay_buffer = replay_buffer
        self.teacher_worker_groups = teacher_worker_groups or {}
        self.alias_to_group_alias = alias_to_group_alias or {}
        self.on_policy_distillation_cfg = on_policy_distillation_cfg or {}
        self._has_distillation_teachers = bool(self.teacher_worker_groups)
        self._teacher_seq_pad_multiple = teacher_seq_pad_multiple(
            self.teacher_worker_groups,
            self.master_config.policy["make_sequence_length_divisible_by"],
        )
        # Per-teacher locks to serialize get_logprobs calls. Concurrent calls
        # to the same teacher cause NCCL collective desync across workers
        # (different workers may receive requests in different order → SeqNum
        # mismatch → 600s timeout → crash). Different teachers can still run
        # in parallel since they use separate NCCL groups on separate nodes.
        self._teacher_locks: dict[str, _threading.Lock] = {
            k: _threading.Lock() for k in self.teacher_worker_groups
        }
        self.running = False

        self._pg_lock: _threading.Lock = _threading.Lock()

        # Event for manual pause/resume control
        self._manual_pause_cleared = _threading.Event()
        self._manual_pause_cleared.set()

        self._refit_pause_cleared = _threading.Event()
        self._refit_pause_cleared.set()  # Start in cleared state

        self.current_weight_version: int = start_step
        self.initial_weight_version: int = start_step

        # Track when generation limits cause collection to pause
        self._last_limit_warning_version = None
        self._last_pause_log_version = None

        # Event to signal when generation limits are cleared (more efficient than polling)
        self._generation_limit_cleared = _threading.Event()
        self._generation_limit_cleared.set()  # Start in cleared state

        # Track threads
        self._inflight_threads: set[_threading.Thread] = set()
        self._threads_lock: _threading.Lock = _threading.Lock()

        # [CUSTOM] Over-launch factor per target: a round for a target launches
        # ceil(needed * groups_multiplier) groups and keeps the first `needed`
        # ones to pass dynamic sampling (see _claim_target); 1 keeps the
        # upstream behavior of launching exactly the groups still needed.
        async_cfg = self.master_config.grpo["async_grpo"]
        self._groups_multiplier = float(async_cfg.get("groups_multiplier", 1.0))
        if self._groups_multiplier < 1.0:
            raise ValueError(
                f"async_grpo.groups_multiplier must be >= 1, got {self._groups_multiplier}"
            )
        # [CUSTOM] With flexible sampling the step a group trains on does not
        # matter, so a group finishing after its target is full is reassigned
        # to the next target that still needs groups instead of being dropped.
        self._reassign_surplus = bool(async_cfg.get("flexible_sampling", False))
        self._max_age = int(async_cfg["max_trajectory_age_steps"])

        # Limit in-flight generator requests to
        # num_prompts_per_step * max_trajectory_age_steps * groups_multiplier.
        # This value limits the parallelism of the generation requests.
        # [CUSTOM] async_grpo.max_concurrent_generations overrides the default.
        max_concurrent_generations = async_cfg.get("max_concurrent_generations")
        if max_concurrent_generations is not None:
            max_inflight = int(max_concurrent_generations)
        else:
            max_inflight = (
                math.ceil(
                    int(self.master_config.grpo["num_prompts_per_step"])
                    * self._max_age
                    * self._groups_multiplier
                )
                or 1
            )
        self._inflight_sema = _threading.Semaphore(max_inflight)

        # Simple lock to prevent race conditions when checking/spawning workers
        self._generation_check_lock: _threading.Lock = _threading.Lock()
        # Track which target weights are currently being generated (globally)
        self._generating_targets: set[int] = set()
        # Track spawned, buffered, and completed prompt-group workers per target.
        self._spawned_per_target: dict[int, int] = {}
        self._buffered_per_target: dict[int, int] = {}
        self._completed_per_target: dict[int, int] = {}
        self._spawning_targets: set[int] = set()
        self._counter_lock: _threading.Lock = _threading.Lock()

        # [CUSTOM] Dynamic sampling (DAPO-style): drop prompt groups whose rewards are all
        # equal (zero advantage, no training signal) and relaunch a replacement for the same
        # target, relying on _get_next_target_for_generation's existing gap-filling (via
        # ReplayBuffer.get_trajectories_needed) to pick the slack back up automatically. After
        # dynamic_sampling_max_gen_batches * num_prompts_per_step launches for one target,
        # groups are accepted unfiltered so training cannot stall. Not implemented upstream at
        # all (checked v0.7.0 and current upstream main as of 2026-11-02) -- not a stopgap.
        grpo_cfg = self.master_config.grpo
        self._use_dynamic_sampling = bool(grpo_cfg.get("use_dynamic_sampling", False))
        num_prompts_per_step = int(grpo_cfg["num_prompts_per_step"])
        self._max_filtered_launches_per_target = (
            num_prompts_per_step * int(grpo_cfg["dynamic_sampling_max_gen_batches"])
            if self._use_dynamic_sampling
            else 0
        )
        # These persist across _maybe_release_target's per-round counter resets (unlike
        # _spawned_per_target/_buffered_per_target/_completed_per_target) because the trainer
        # reads them later, via get_target_stats, after sampling the target from the replay
        # buffer -- pruned in set_weight_version once the trainer has moved past a target.
        self._dynsample_spawned_per_target: dict[int, int] = {}
        self._discarded_per_target: dict[int, int] = {}
        self._unfiltered_reward_sum: dict[int, float] = {}
        self._unfiltered_reward_count: dict[int, int] = {}
        # [CUSTOM] Groups that passed dynamic sampling but found their target
        # full (surplus), and surplus groups reassigned to a later target. Same
        # lifetime as the dynamic-sampling counters above.
        self._surplus_per_target: dict[int, int] = {}
        self._reassigned_per_target: dict[int, int] = {}
        # [CUSTOM] Groups claimed for a target but not yet in the replay buffer
        # (see _claim_target): the buffer's own count plus these is what a
        # target already has.
        self._claim_lock: _threading.Lock = _threading.Lock()
        self._claimed_per_target: dict[int, int] = {}

        # [CUSTOM] Data epochs. The collector reads the data max_num_epochs
        # times (upstream reads it once and then stops collecting); epochs fully
        # read are counted under _pending_lock together with the dataloader
        # read so a checkpoint saves a consistent pair (see get_checkpoint_state
        # / restore_collection_state).
        self.current_epoch: int = 0
        self._epochs_completed = 0
        self._max_data_epochs: Optional[int] = (
            int(grpo_cfg["max_num_epochs"])
            if grpo_cfg.get("max_num_epochs") is not None
            else None
        )
        self._data_exhausted = False
        # True while requeued prompts are taken but their workers not started yet.
        self._relaunching = False

        # [CUSTOM] Skip solved prompts: a prompt whose whole group reaches
        # solved_reward_threshold (all-1) is recorded by dataset idx and not
        # generated again in later epochs, while all-0 prompts stay in. Epochs
        # then shrink, so max_num_epochs counts passes over the data: the run
        # ends when the remaining groups cannot fill a step (see is_finished).
        self._skip_solved = bool(async_cfg.get("skip_solved_prompts", False))
        self._solved_threshold = float(async_cfg.get("solved_reward_threshold", 1.0))
        self._solved_prompts: dict[int, int] = {}  # dataset idx -> epoch solved
        self._num_skipped_solved = 0

        # [CUSTOM] Prompt bookkeeping, so no prompt taken from the dataloader is
        # lost: every prompt is pending (keyed by a uid) until it is trained on
        # or discarded by dynamic sampling. Prompts a round does not launch,
        # and surplus, expired or failed groups, go back to the retry queue;
        # checkpoints save the pending prompts so a resume regenerates those
        # that are not in the restored replay buffer.
        self._pending_lock = _threading.Lock()
        self._pending_prompts: dict[int, BatchedDataDict[DatumSpec]] = {}
        self._retry_uids: collections.deque[int] = collections.deque()
        self._next_prompt_uid = 0
        self._num_requeued_prompts = 0
        # [CUSTOM] A prompt whose generation fails, or whose group expires, more
        # than max_prompt_retries times is dropped: a prompt that always fails
        # would otherwise take up launches forever and keep is_finished() False.
        # Surplus and unlaunched prompts do not count; counts restart on resume.
        self._max_prompt_retries = int(async_cfg.get("max_prompt_retries", 3))
        self._prompt_failures: dict[int, int] = {}
        self._num_dropped_prompts = 0

    def _calculate_target_weights(self, generation_weight_version: int) -> list[int]:
        """Calculate target weight versions for given generation weight version.

        The list of versions returned enumerate the possible version a generation
        server can target. These versions are looped over to see what training
        step they can target. If all target versions are exhausted, this generation
        server will remain idle until the next weight update.

        Example:
        generation_weight_version = 10
        max_trajectory_age_steps = 4

        Returns:
            [11, 12, 13, 14]  # Meaning this generation server can create trajectories for training step 11, 12, 13, 14
        """
        # Read async config strictly from grpo.async_grpo
        async_cfg = self.master_config.grpo.get("async_grpo", {})
        max_trajectory_age = async_cfg["max_trajectory_age_steps"]
        if generation_weight_version == self.initial_weight_version:
            return [
                i
                for i in range(
                    self.initial_weight_version,
                    self.initial_weight_version + max_trajectory_age + 1,
                )
            ]

        return [generation_weight_version + i for i in range(1, max_trajectory_age + 1)]

    def _find_target_needing_generation(
        self, generation_weight_version: int
    ) -> tuple[Optional[int], int]:
        """Return (target, trajectories_needed) for the first target that needs generation.

        Must be called with ``self._generation_check_lock`` held. Does not reserve.

        [CUSTOM-GAPFILL-CURRENT 2026-11-08] Besides the future targets of
        _calculate_target_weights, the CURRENT version is a gap-fill-only candidate: a prompt
        group for target v whose worker fails after the refit to v (e.g. a NeMo-Gym 500)
        leaves v short of a full batch, the trainer waits for v, and the upstream window
        [v+1 .. v+age] never refills it -- job 63861875 hung at step 7 with 511/512 groups.
        Gap-fill only (needed < num_prompts): a fully missing current target means the
        trainer already consumed it (sample() removes the groups and advances the watermark
        atomically), so it must never trigger a fresh full generation.
        """
        target_weights = self._calculate_target_weights(generation_weight_version)
        candidates = [(target, False) for target in target_weights]
        if generation_weight_version not in target_weights:
            candidates.insert(0, (generation_weight_version, True))
        num_prompts = int(self.master_config.grpo["num_prompts_per_step"])
        max_age_steps = int(
            self.master_config.grpo["async_grpo"]["max_trajectory_age_steps"]
        )
        last_consumed_target = ray.get(
            self.replay_buffer.get_last_target_weight_already_generated.remote()
        )
        for target_weight, gap_fill_only in candidates:
            if target_weight <= last_consumed_target:
                continue
            if target_weight in self._generating_targets:
                continue
            trajectories_needed = ray.get(
                self.replay_buffer.get_trajectories_needed.remote(
                    target_weight, num_prompts, max_age_steps
                )
            )
            if trajectories_needed <= 0:
                continue
            if gap_fill_only and trajectories_needed >= num_prompts:
                continue
            return target_weight, trajectories_needed
        return None, 0

    def _get_next_target_for_generation(
        self, generation_weight_version: int
    ) -> Optional[int]:
        """Get the next target weight that needs generation (if any)."""
        num_prompts = int(self.master_config.grpo["num_prompts_per_step"])
        with self._generation_check_lock:
            target_weight, trajectories_needed = self._find_target_needing_generation(
                generation_weight_version
            )
            if target_weight is None:
                return None
            self._generating_targets.add(target_weight)
            if trajectories_needed < num_prompts:
                print(
                    f"🎯 Reserved target weight {target_weight} for gap-filling "
                    f"(need {trajectories_needed}/{num_prompts} more trajectories)"
                )
            else:
                print(f"🎯 Reserved target weight {target_weight} for generation")
            return target_weight

    def set_weight_version(self, version: int) -> None:
        self.current_weight_version = version

        # [CUSTOM] Prune dynamic-sampling counters for targets the trainer has moved past --
        # these aren't cleared by _maybe_release_target (see get_target_stats), so without
        # this they'd grow unbounded over a long run.
        with self._counter_lock:
            for consumed_target in [
                t for t in self._dynsample_spawned_per_target if t < version
            ]:
                self._dynsample_spawned_per_target.pop(consumed_target, None)
                self._discarded_per_target.pop(consumed_target, None)
                self._unfiltered_reward_sum.pop(consumed_target, None)
                self._unfiltered_reward_count.pop(consumed_target, None)
                self._surplus_per_target.pop(consumed_target, None)
                self._reassigned_per_target.pop(consumed_target, None)
        with self._claim_lock:
            for consumed_target in [t for t in self._claimed_per_target if t < version]:
                self._claimed_per_target.pop(consumed_target, None)

        # Resume collection if it was paused due to generation limits
        was_paused = not self._generation_limit_cleared.is_set()
        if was_paused:
            self._generation_limit_cleared.set()  # Signal that collection can resume
            print(f"🔄 Updated weight version to {version}, resuming collection")
        else:
            print(f"🔄 Updated weight version to {version}")

    def get_target_stats(self, target_weight: int) -> dict[str, float]:
        """Return dynamic-sampling generation statistics for a consumed target weight.

        Called by the trainer after it has sampled target_weight's groups from the replay
        buffer, once no more generation for that target will happen. Unlike
        _spawned_per_target/_buffered_per_target/_completed_per_target (cleared by
        _maybe_release_target as soon as the collector's own in-flight round finishes), the
        dynamic-sampling counters persist across however many rounds of gap-filling a target
        needed -- the trainer reads them at a later point in time. Pruned in
        set_weight_version once the trainer has moved past a target.
        """
        with self._counter_lock:
            spawned = self._dynsample_spawned_per_target.get(target_weight, 0)
            discarded = self._discarded_per_target.get(target_weight, 0)
            reward_sum = self._unfiltered_reward_sum.get(target_weight, 0.0)
            reward_count = self._unfiltered_reward_count.get(target_weight, 0)
            surplus = self._surplus_per_target.get(target_weight, 0)
            reassigned = self._reassigned_per_target.get(target_weight, 0)
        if spawned == 0:
            return {}
        num_prompts_per_step = int(self.master_config.grpo["num_prompts_per_step"])
        stats = {
            "dynamic_sampling_num_gen_batches": spawned / num_prompts_per_step,
            "dynamic_sampling_num_discarded_groups": float(discarded),
            "dynamic_sampling_num_surplus_groups": float(surplus),
            "dynamic_sampling_num_reassigned_groups": float(reassigned),
        }
        if reward_count > 0:
            stats["unfiltered_reward"] = reward_sum / reward_count
        return stats

    def _should_pause_for_generation_limits(self) -> bool:
        """Check if collection should be paused due to generation limits."""
        try:
            # Same candidates as _get_next_target_for_generation (incl. gap-filling the
            # current target, [CUSTOM-GAPFILL-CURRENT]) so "pause" and "nothing to
            # generate" can never disagree.
            with self._generation_check_lock:
                target_weight, _ = self._find_target_needing_generation(
                    self.current_weight_version
                )
            if target_weight is not None:
                return False  # Found a target that needs generation

            # Called on every re-check of a paused loop: log once per weight version.
            if self._last_pause_log_version != self.current_weight_version:
                target_weights = self._calculate_target_weights(
                    self.current_weight_version
                )
                print(
                    f"⏸️ All target weights {target_weights} already generated or in progress, pausing"
                )
                self._last_pause_log_version = self.current_weight_version
            return True
        except Exception:
            return False

    def start_collection(self, dataloader: StatefulDataLoader) -> None:
        """Start collecting trajectories from dataloader."""
        self.running = True
        self.dataloader = dataloader

        print("Started continuous trajectory collection")

        self.collection_thread = _threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        print("Collection thread started, start_collection returning")

    def _wait_until_can_launch(self) -> bool:
        """Block while collection is paused.

        Paused manually, for a refit, or because every target in range has its
        groups. Returns False if stopped.
        """
        # [CUSTOM-GAPFILL-CURRENT 2026-11-08] Every pause clears the event and
        # re-checks, so a wake (refit, released reservation) with nothing to do
        # pauses again instead of letting the loop spin; the timeout is only a
        # backstop.
        while self.running:
            # Check if manually paused and wait
            if not self._manual_pause_cleared.is_set():
                self._manual_pause_cleared.wait()

            # Check if refit is in progress and wait
            if not self._refit_pause_cleared.is_set() and self.running:
                print("⏸️ Pausing collection for refit...")
                self._refit_pause_cleared.wait()
                print("▶️ Refit completed, resuming collection")

            if not self.running:
                break

            # Check if generation limits require pausing collection
            if not self._should_pause_for_generation_limits():
                break
            # Only log warning once per weight version
            if self._last_limit_warning_version != self.current_weight_version:
                print(
                    "⏸️ Pausing collection: every target for weight version "
                    f"{self.current_weight_version} is generated or in progress. "
                    "Waiting for a weight update or a short target to gap-fill..."
                )
                self._last_limit_warning_version = self.current_weight_version
            self._generation_limit_cleared.clear()  # Clear the event to pause
            # Re-check after clearing, so a wake landing between the check and
            # the clear is not lost.
            if self._should_pause_for_generation_limits():
                self._generation_limit_cleared.wait(timeout=_GENERATION_LIMIT_RECHECK_S)

        return self.running

    def _launch_retries(self) -> int:
        """[CUSTOM] Launch the requeued prompts, if any. Returns how many were taken."""
        with self._pending_lock:
            self._relaunching = True
        retry_items = []
        try:
            retry_items = self._take_retry_prompts()
            if retry_items and self._wait_until_can_launch():
                if self._launch_prompts(retry_items) == 0:
                    time.sleep(0.5)
        finally:
            with self._pending_lock:
                self._relaunching = False
        return len(retry_items)

    def _collection_loop(self):
        """Run the collection loop in background thread.

        [CUSTOM] Reads the data max_num_epochs times (upstream reads it once and
        stops), regenerating requeued prompts first, skipping prompts solved in
        an earlier epoch, and keeping every prompt pending until it is trained
        on or discarded (see _register_prompt). After the last epoch only
        requeued prompts are generated, until nothing is left (is_finished).
        """
        try:
            while self.running:
                if (
                    self._max_data_epochs is not None
                    and self._epochs_completed >= self._max_data_epochs
                ):
                    if not self._data_exhausted:
                        self._data_exhausted = True
                        print(
                            f"🏁 Read all {self._max_data_epochs} data epochs; "
                            "generating only requeued prompts from now on"
                        )
                    if self._launch_retries() == 0:
                        time.sleep(0.5)
                    continue

                self.current_epoch = self._epochs_completed + 1
                print(f"Starting epoch {self.current_epoch} of trajectory collection")

                # Defensive: torchdata's multiprocess iterator asserts
                # "_snapshot" in next_iter_state. If a stale/partial state was
                # loaded elsewhere, drop it so iter() doesn't die here.
                next_state = getattr(self.dataloader, "next_iter_state", None)
                if (
                    isinstance(next_state, dict)
                    and (getattr(self.dataloader, "num_workers", 0) or 0) > 0
                    and "_snapshot" not in next_state
                ):
                    print(
                        "⚠️  Clearing incompatible dataloader next_iter_state "
                        "(missing '_snapshot'); using a fresh iterator."
                    )
                    self.dataloader.next_iter_state = None

                iterator = iter(self.dataloader)
                while self.running:
                    # Regenerate requeued / restored prompts first.
                    if self._launch_retries():
                        continue
                    # Fetch and register under one lock so a checkpoint never
                    # sees the dataloader past a prompt that is not pending.
                    with self._pending_lock:
                        try:
                            batch = next(iterator)
                        except StopIteration:
                            self._epochs_completed += 1
                            break
                        items = []
                        for single in (
                            batch.slice(i, i + 1) for i in range(batch.size)
                        ):
                            # Solved in an earlier epoch: skip it.
                            if (
                                self._skip_solved
                                and int(single["idx"][0]) in self._solved_prompts
                            ):
                                self._num_skipped_solved += 1
                                continue
                            items.append((self._register_prompt(single), single))
                    if not items:
                        continue

                    if not self._wait_until_can_launch():
                        break

                    if self._launch_prompts(items) == 0:
                        time.sleep(0.5)

                if self.running:
                    print(
                        f"Finished epoch {self.current_epoch} of trajectory collection"
                    )

        except Exception as e:
            print(f"❌ Error in trajectory collection: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.running = False
            print("🛑 Trajectory collection stopped")

    def _process_batch(self, batch: BatchedDataDict[DatumSpec]) -> bool:
        """Register every prompt of batch and launch it (see _launch_prompts).

        Returns whether any worker started. Prompts not launched are kept in
        the retry queue either way.
        """
        with self._pending_lock:
            items = [
                (self._register_prompt(single), single)
                for single in (batch.slice(i, i + 1) for i in range(batch.size))
            ]
        return self._launch_prompts(items) > 0

    def _launch_prompts(
        self, items: list[tuple[int, BatchedDataDict[DatumSpec]]]
    ) -> int:
        """Generate (uid, prompt) items for one target weight.

        [CUSTOM] Upstream's _process_batch, launching ceil(needed *
        groups_multiplier) groups for the target (the first `needed` to pass
        dynamic sampling are kept, see _claim_target) and sending the prompts it
        does not launch back to the retry queue instead of dropping them.
        Returns the number of workers started.
        """
        target_weight: Optional[int] = None
        to_launch: list[tuple[int, BatchedDataDict[DatumSpec]]] = []
        started = 0
        try:
            generation_weight_version = self.current_weight_version
            num_generations = self.master_config.grpo["num_generations_per_prompt"]
            num_prompts_in_batch = len(items)
            num_prompts_per_step = int(self.master_config.grpo["num_prompts_per_step"])
            max_age_steps = int(
                self.master_config.grpo["async_grpo"]["max_trajectory_age_steps"]
            )

            # Get the next target weight that needs generation
            target_weight = self._get_next_target_for_generation(
                generation_weight_version
            )

            if target_weight is None:
                print(
                    f"🔄 No targets need generation for weight {generation_weight_version}"
                )
                self._requeue_prompts(items)
                return 0

            print(
                f"🎯 Generating for target weight {target_weight} from generation_weight_version {generation_weight_version}"
            )

            trajectories_needed = ray.get(
                self.replay_buffer.get_trajectories_needed.remote(
                    target_weight, num_prompts_per_step, max_age_steps
                )
            )
            num_prompts_to_generate = min(
                num_prompts_in_batch,
                math.ceil(trajectories_needed * self._groups_multiplier),
            )
            if num_prompts_to_generate == 0:
                print(
                    f"🔄 Target {target_weight} already has enough trajectories, skipping"
                )
                with self._generation_check_lock:
                    self._generating_targets.discard(target_weight)
                self._requeue_prompts(items)
                return 0

            if num_prompts_to_generate < num_prompts_in_batch:
                print(
                    f"🎯 Gap-filling for target weight {target_weight}: "
                    f"generating {num_prompts_to_generate}/{num_prompts_in_batch} "
                    f"prompts (need {trajectories_needed} more trajectories)"
                )
            to_launch = items[:num_prompts_to_generate]
            # The prompts this round does not launch wait for the next one.
            self._requeue_prompts(items[num_prompts_to_generate:])

            # Generate only the prompt groups needed for this target. While the
            # spawn loop is open, workers may finish before later workers start,
            # so reservation release is deferred until spawning closes.
            with self._counter_lock:
                self._spawning_targets.add(target_weight)
            try:
                for prompt_idx, (prompt_uid, single_prompt_batch) in enumerate(
                    to_launch
                ):
                    # Wait for refit to complete if in progress
                    if not self._refit_pause_cleared.is_set() and self.running:
                        with self._threads_lock:
                            active_threads = len(self._inflight_threads)
                        print(
                            f"⏸️ Waiting for refit to complete before starting new generation ({active_threads} threads still active)"
                        )
                        print(
                            "   Note: With vLLM V1 async engine, active threads can complete during weight update"
                        )
                        self._refit_pause_cleared.wait()

                        # After refit finishes if weight version has updated, reflect that in the new trajectories
                        generation_weight_version = self.current_weight_version

                    repeated_batch = single_prompt_batch.repeat_interleave(
                        num_generations
                    )

                    # [CUSTOM] Decide at launch time whether this particular group is
                    # allowed to be discarded by dynamic sampling (see get_target_stats),
                    # mirroring the per-target launch cap: once dynamic_sampling_max_gen_batches
                    # * num_prompts_per_step groups have been launched for this target, further
                    # ones are accepted unconditionally so training cannot stall.
                    with self._counter_lock:
                        self._dynsample_spawned_per_target[target_weight] = (
                            self._dynsample_spawned_per_target.get(target_weight, 0) + 1
                        )
                        total_spawned = self._dynsample_spawned_per_target[
                            target_weight
                        ]
                    filterable = (
                        self._use_dynamic_sampling
                        and total_spawned <= self._max_filtered_launches_per_target
                    )

                    worker = _threading.Thread(
                        target=self._run_prompt_group_worker,
                        args=(
                            repeated_batch,
                            generation_weight_version,
                            target_weight,
                            prompt_idx,
                            filterable,
                            prompt_uid,
                        ),
                        daemon=True,
                    )
                    self._inflight_sema.acquire()
                    registered = False
                    try:
                        with self._threads_lock:
                            self._inflight_threads.add(worker)
                        with self._counter_lock:
                            self._spawned_per_target[target_weight] = (
                                self._spawned_per_target.get(target_weight, 0) + 1
                            )
                            spawned_count = self._spawned_per_target[target_weight]
                        registered = True
                        worker.start()
                    except Exception:
                        # The worker never ran, so it won't release its slot or
                        # run its finally block; undo the bookkeeping here.
                        with self._threads_lock:
                            self._inflight_threads.discard(worker)
                        if registered:
                            with self._counter_lock:
                                updated_count = (
                                    self._spawned_per_target.get(target_weight, 0) - 1
                                )
                                if updated_count > 0:
                                    self._spawned_per_target[target_weight] = (
                                        updated_count
                                    )
                                else:
                                    self._spawned_per_target.pop(target_weight, None)
                        with self._counter_lock:
                            updated_total = (
                                self._dynsample_spawned_per_target.get(target_weight, 0)
                                - 1
                            )
                            if updated_total > 0:
                                self._dynsample_spawned_per_target[target_weight] = (
                                    updated_total
                                )
                            else:
                                self._dynsample_spawned_per_target.pop(
                                    target_weight, None
                                )
                        self._inflight_sema.release()
                        # [CUSTOM] Never generated: try the prompt again later.
                        self._requeue_prompt(prompt_uid, failed=True)
                        raise
                    started += 1
                    print(
                        f"📊 Started worker {started}/{num_prompts_to_generate} for "
                        f"target_weight={target_weight} ({spawned_count} total)"
                    )
            finally:
                if started < num_prompts_to_generate:
                    print(
                        f"⚠️ Only {started}/{num_prompts_to_generate} workers "
                        f"started for target_weight={target_weight}"
                    )
                with self._counter_lock:
                    self._spawning_targets.discard(target_weight)
                self._maybe_release_target(target_weight)

            self._cleanup_finished_threads()

        except Exception as e:
            if target_weight is not None:
                with self._counter_lock:
                    self._spawning_targets.discard(target_weight)
                self._maybe_release_target(target_weight)
            # [CUSTOM] Prompts the failed round never started stay pending.
            self._requeue_prompts(to_launch[started:])
            print(f"❌ Error processing batch: {e}")
            import traceback

            traceback.print_exc()
        return started

    def get_weight_version(self) -> int:
        return self.current_weight_version

    def pause(self) -> None:
        """Pause trajectory collection."""
        self._manual_pause_cleared.clear()  # Signal collection to pause
        print("Trajectory collection paused")

    def resume(self) -> None:
        """Resume trajectory collection."""
        self._manual_pause_cleared.set()  # Signal collection to resume
        print("Trajectory collection resumed")

    def prepare_for_refit(self) -> None:
        """Pause new generation starts and optionally wait for pending generations.

        For backends with an async engine in-flight weight updates allows ongoing generations
        to continue with their current KV caches while weights are updated.
        This significantly improves async performance.

        For non-async engines, waits for all pending generations to complete before refit.
        """
        start_time = time.time()
        print("🔄 Preparing for refit: pausing new generations...")

        # Pause new generation starts
        self._refit_pause_cleared.clear()
        print("⏸️ New generation starts paused")

        # Check if we're using async engine
        generation_cfg = self.master_config.policy.get("generation", {})
        backend = generation_cfg.get("backend", "")
        if backend == "vllm":
            is_async_engine = generation_cfg.get("vllm_cfg", {}).get(
                "async_engine", False
            )
        elif backend == "megatron":
            is_async_engine = generation_cfg.get("mcore_generation_config", {}).get(
                "async_engine", False
            )
        else:
            is_async_engine = False
        in_flight_weight_updates = self.master_config.grpo.get("async_grpo", {}).get(
            "in_flight_weight_updates", False
        )

        if is_async_engine and in_flight_weight_updates:
            # async engines support in-flight weight updates
            # Ongoing generations will continue with their current KV caches
            # New generations (after weight update) will use the updated weights
            print(
                f"🚀 Using {backend} in-flight weight update - skipping wait for pending generations"
            )
            print(
                f"   {len(self._inflight_threads)} ongoing generations will complete with current weights"
            )
        else:
            # For non-async engines, wait for all pending generations to complete
            print(
                "⏸️ Non-async engine: waiting for all pending generations to complete..."
            )
            self.wait_for_pending_generations()

        elapsed = time.time() - start_time
        print(f"✅ Ready for refit (took {elapsed:.2f}s)")

    def resume_after_refit(self) -> None:
        """Resume new generation starts after refit is complete."""
        print("🔄 Resuming generation starts after refit")

        # Invalidate&recompute vLLM caches after the in-flight weight updates if
        # recompute_kv_cache_after_weight_updates is True (AREAL-style implementation).
        # Otherwise, keep using the stale KV caches (Magistral-style implementation).
        async_cfg = self.master_config.grpo.get("async_grpo", {})
        if async_cfg.get("in_flight_weight_updates", False) and async_cfg.get(
            "recompute_kv_cache_after_weight_updates", False
        ):
            try:
                print("🔄 Invalidating vLLM prefix/KV caches after weight update")
                invalidated = self.policy_generation.invalidate_kv_cache()
                if invalidated:
                    print("✅ Invalidated vLLM prefix/KV caches after weight update")
                else:
                    print(
                        "⚠️ vLLM cache invalidation reported partial/unsuccessful on some workers"
                    )
            except Exception as e:
                print(f"⚠️ Failed to invalidate vLLM caches: {e}")

        self._refit_pause_cleared.set()

    def wait_for_pending_generations(self) -> None:
        """Wait for all in-flight generation threads to complete."""
        start_time = time.time()

        while True:
            with self._threads_lock:
                finished = {t for t in self._inflight_threads if not t.is_alive()}
                for t in finished:
                    self._inflight_threads.remove(t)

                pending_count = len(self._inflight_threads)

            if pending_count == 0:
                print("✅ All generation threads completed")
                break

            elapsed = time.time() - start_time
            print(
                f"⏳ Waiting for {pending_count} pending generation threads... ({elapsed:.1f}s elapsed)"
            )
            time.sleep(0.5)

    def get_dataloader_state(self) -> dict:
        """Get the current dataloader state for checkpointing."""
        if hasattr(self, "dataloader") and hasattr(self.dataloader, "state_dict"):
            return self.dataloader.state_dict()
        return {}

    # ------------------------------------------------------------------
    # [CUSTOM] Group claims (groups_multiplier / flexible sampling)
    # ------------------------------------------------------------------

    def _claim_target(
        self, target_weight: int, generation_weight_version: int
    ) -> Optional[int]:
        """Reserve one group of a target before pushing it to the replay buffer.

        A target has its groups when the buffer holds, or workers have claimed,
        num_prompts_per_step of them. Returns target_weight if it still needs
        groups, otherwise (with flexible sampling) the lowest later target that
        does and that the group can still reach within the age window. None
        means the group is surplus and must not reach the replay buffer, which
        would keep it past its target and later evict it as stale.
        """
        num_prompts = int(self.master_config.grpo["num_prompts_per_step"])
        candidates = [target_weight]
        if self._reassign_surplus:
            first = max(target_weight + 1, self.current_weight_version)
            candidates += list(
                range(first, generation_weight_version + self._max_age + 1)
            )
        with self._claim_lock:
            for target in candidates:
                if target < self.current_weight_version:
                    continue
                needed = ray.get(
                    self.replay_buffer.get_trajectories_needed.remote(
                        target, num_prompts, self._max_age
                    )
                )
                if needed - self._claimed_per_target.get(target, 0) <= 0:
                    continue
                self._claimed_per_target[target] = (
                    self._claimed_per_target.get(target, 0) + 1
                )
                if target != target_weight:
                    with self._counter_lock:
                        self._reassigned_per_target[target_weight] = (
                            self._reassigned_per_target.get(target_weight, 0) + 1
                        )
                return target
        return None

    def _release_claim(self, target: int) -> None:
        """The claimed group reached the buffer (or never will)."""
        with self._claim_lock:
            left = self._claimed_per_target.get(target, 0) - 1
            if left > 0:
                self._claimed_per_target[target] = left
            else:
                self._claimed_per_target.pop(target, None)

    # ------------------------------------------------------------------
    # [CUSTOM] Prompt bookkeeping, solved prompts and checkpoint state
    # ------------------------------------------------------------------

    def _register_prompt(self, single_prompt_batch: BatchedDataDict[DatumSpec]) -> int:
        """Mark a prompt pending and return its uid. Caller holds _pending_lock."""
        uid = self._next_prompt_uid
        self._next_prompt_uid += 1
        self._pending_prompts[uid] = single_prompt_batch
        return uid

    def _finish_prompt(self, uid: Optional[int]) -> None:
        """A prompt was trained on or discarded by dynamic sampling."""
        if uid is None:
            return
        with self._pending_lock:
            self._pending_prompts.pop(uid, None)
            self._prompt_failures.pop(uid, None)

    def _requeue_prompt(self, uid: Optional[int], failed: bool = False) -> None:
        """Send a still-pending prompt back for generation.

        failed: the generation failed or the group expired. Such a prompt is
        dropped once it has failed more than max_prompt_retries times.
        """
        if uid is None:
            return
        with self._pending_lock:
            if uid not in self._pending_prompts or uid in self._retry_uids:
                return
            if failed:
                failures = self._prompt_failures.get(uid, 0) + 1
                if failures > self._max_prompt_retries:
                    self._pending_prompts.pop(uid)
                    self._prompt_failures.pop(uid, None)
                    self._num_dropped_prompts += 1
                    print(
                        f"🗑️ Dropped prompt uid {uid}: failed or expired {failures} times "
                        f"(max_prompt_retries {self._max_prompt_retries})"
                    )
                    return
                self._prompt_failures[uid] = failures
            self._retry_uids.append(uid)
            self._num_requeued_prompts += 1

    def _requeue_prompts(
        self, items: list[tuple[int, BatchedDataDict[DatumSpec]]]
    ) -> None:
        """Prompts a round did not launch go back to the queue, as not failed."""
        for uid, _ in items:
            self._requeue_prompt(uid)

    def _take_retry_prompts(self) -> list[tuple[int, BatchedDataDict[DatumSpec]]]:
        """Pop every prompt waiting in the retry queue."""
        with self._pending_lock:
            items = []
            while self._retry_uids:
                uid = self._retry_uids.popleft()
                if uid in self._pending_prompts:
                    items.append((uid, self._pending_prompts[uid]))
            return items

    def _mark_solved(self, repeated_batch: BatchedDataDict[DatumSpec]) -> None:
        """Record a prompt whose whole group was solved (all-1)."""
        idx = int(repeated_batch["idx"][0])
        with self._pending_lock:
            self._solved_prompts.setdefault(idx, self.current_epoch)

    def is_finished(self) -> bool:
        """Every data epoch is read and no prompt is left to generate.

        Checked under _pending_lock: a worker requeues its prompt before it
        exits, and the collection loop keeps _relaunching set until the workers
        of the prompts it took are started, so no prompt slips between checks.
        """
        if not self._data_exhausted:
            return False
        with self._pending_lock:
            if self._retry_uids or self._relaunching:
                return False
            with self._threads_lock:
                return not any(t.is_alive() for t in self._inflight_threads)

    def update_prompt_status(
        self, consumed_uids: list[int], requeue_uids: list[int]
    ) -> dict[str, float]:
        """Called by the trainer after each sample.

        consumed_uids: prompts of the groups the step trained on (finished).
        requeue_uids: prompts of groups the buffer evicted as expired.
        """
        for uid in consumed_uids:
            self._finish_prompt(uid)
        for uid in requeue_uids:
            self._requeue_prompt(uid, failed=True)
        with self._pending_lock:
            stats = {
                "async_num_pending_prompts": float(len(self._pending_prompts)),
                "async_num_requeued_prompts_total": float(self._num_requeued_prompts),
                "async_num_dropped_prompts_total": float(self._num_dropped_prompts),
                "async_data_epoch": float(self.current_epoch),
            }
            if self._skip_solved:
                stats["async_num_solved_prompts"] = float(len(self._solved_prompts))
                stats["async_num_skipped_solved_total"] = float(
                    self._num_skipped_solved
                )
            return stats

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Dataloader state and collection state, read under one lock.

        Every prompt the dataloader state counts as yielded is then either
        finished or in pending_prompts. The trainer saves "dataloader_state" as
        train_dataloader.pt (raw, as upstream) and "collector_state" as
        collector_state.pt (holds BatchedDataDicts, so it is read back with
        weights_only=False).
        """
        with self._pending_lock:
            return {
                "dataloader_state": self.get_dataloader_state(),
                "collector_state": {
                    "current_epoch": self.current_epoch,
                    "epochs_completed": self._epochs_completed,
                    "solved_prompts": dict(self._solved_prompts),
                    "pending_prompts": {
                        uid: self._pending_prompts[uid]
                        for uid in sorted(self._pending_prompts)
                    },
                    "next_prompt_uid": self._next_prompt_uid,
                },
            }

    def restore_collection_state(
        self, state: dict[str, Any], buffered_uids: Optional[list[int]] = None
    ) -> None:
        """On resume, restore epochs read, solved prompts and pending prompts.

        state is the collector_state.pt of the checkpoint.
        buffered_uids: prompt ids of the groups restored into the replay buffer;
        those prompts stay pending until trained on, every other pending prompt
        is queued for regeneration first.
        """
        buffered = set(buffered_uids or [])
        with self._pending_lock:
            self._epochs_completed = int(state.get("epochs_completed", 0))
            self.current_epoch = int(state.get("current_epoch", 0))
            self._solved_prompts = {
                int(idx): int(epoch)
                for idx, epoch in state.get("solved_prompts", {}).items()
            }
            pending = state.get("pending_prompts", {})
            queued = 0
            for uid in sorted(pending):
                uid = int(uid)
                self._pending_prompts[uid] = pending[uid]
                if uid not in buffered:
                    self._retry_uids.append(uid)
                    queued += 1
            self._next_prompt_uid = max(
                int(state.get("next_prompt_uid", 0)),
                max(self._pending_prompts, default=-1) + 1,
                self._next_prompt_uid,
            )
        print(
            f"♻️ Restored collection state: {self._epochs_completed} data epochs read, "
            f"{len(self._solved_prompts)} solved prompts, {len(pending)} pending prompts "
            f"({queued} queued for regeneration, {len(pending) - queued} in the replay buffer)"
        )

    def _cleanup_finished_threads(self) -> None:
        with self._threads_lock:
            finished = {t for t in self._inflight_threads if not t.is_alive()}
            for t in finished:
                self._inflight_threads.remove(t)

    def _maybe_release_target(self, target_weight_version: int) -> None:
        """Release a target's reservation once all its workers have completed.

        A worker counts as "completed" whether or not it managed to buffer a
        trajectory. The reservation is released exactly when the number of
        completed workers reaches the number spawned for the target and no more
        workers are being spawned for the same target. Safe to call repeatedly:
        the reservation is discarded at most once and the per-target counters
        are dropped on release (so a later re-reservation starts from a clean
        slate and the dicts don't grow unbounded).
        """
        with self._counter_lock:
            if target_weight_version in self._spawning_targets:
                return
            completed = self._completed_per_target.get(target_weight_version, 0)
            spawned = self._spawned_per_target.get(target_weight_version, 0)
            if completed < spawned:
                return
            buffered = self._buffered_per_target.get(target_weight_version, 0)
            self._spawned_per_target.pop(target_weight_version, None)
            self._buffered_per_target.pop(target_weight_version, None)
            self._completed_per_target.pop(target_weight_version, None)
            self._spawning_targets.discard(target_weight_version)

        with self._generation_check_lock:
            if target_weight_version in self._generating_targets:
                self._generating_targets.discard(target_weight_version)
                print(
                    "🧹 Released reservation for target weight "
                    f"{target_weight_version} ({spawned} workers completed, "
                    f"{buffered} buffered)"
                )

        # [CUSTOM-GAPFILL-WAKE 2026-11-07, reworked 2026-11-08] Wake a paused collection loop on
        # every release. A reservation can be released short of num_prompts_per_step (failed
        # prompt-group workers, dynamic-sampling discards) while the loop is paused with every
        # target reserved; only set_weight_version (a refit) used to wake it, and the trainer
        # waits for that very target, so nothing would start the gap-filling round (job
        # 63795158). The woken loop re-checks with the same candidates as the generation path
        # (_find_target_needing_generation) and pauses again if there is nothing to do,
        # holding its batch, so a wake without work costs nothing.
        if buffered < spawned:
            print(
                f"⚠️ Target weight {target_weight_version} released short "
                f"({buffered}/{spawned} buffered): waking collection to gap-fill"
            )
        self._generation_limit_cleared.set()

    def _compute_teacher_logprobs(
        self,
        input_ids: torch.Tensor,
        agent_refs: list[dict[str, Any]],
        input_lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, float]:
        """Compute teacher logprobs for non-colocated teachers.

        Groups samples by teacher, fans out in parallel, stitches results.

        Args:
            input_ids: [B, S] tokenized input tensor
            agent_refs: list of B agent reference dicts
            input_lengths: [B] per-sample lengths (required for sequence packing)

        Returns:
            ([B, S] teacher logprobs tensor, total_time_seconds)
        """
        opd_cfg = self.on_policy_distillation_cfg
        teacher_model_by_agent_name = opd_cfg.get("teacher_model_by_agent_name", {})
        default_teacher_alias = opd_cfg.get("default_teacher_alias")
        strict = opd_cfg.get("strict_agent_name_match", False)

        # Resolve each sample's agent -> the teacher alias it should be distilled
        # from: the agent name is looked up in teacher_model_by_agent_name; unmapped
        # agents fall back to default_teacher_alias (or raise if strict_agent_name_match).
        # Returns one alias per sample, index-aligned with agent_refs.
        reference_aliases = resolve_reference_aliases(
            agent_refs,
            teacher_model_by_agent_name,
            default_teacher_alias=default_teacher_alias,
            strict_agent_name_match=strict,
        )

        # Map aliases to actual group keys via deduplication mapping
        group_keys = [self.alias_to_group_alias.get(a, a) for a in reference_aliases]

        # Group sample indices by teacher group
        group_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, gk in enumerate(group_keys):
            group_to_indices[gk].append(i)

        B, S = input_ids.shape
        result = torch.zeros(B, S, dtype=torch.float32)
        if (
            not group_to_indices
        ):  # 0-sample batch: nothing to route (avoid max_workers=0)
            return result, 0.0

        def _get_logprobs_for_group(group_key, indices):
            twg = self.teacher_worker_groups[group_key]
            sub_input_ids = input_ids[indices]
            sub_lengths = input_lengths[indices] if input_lengths is not None else None

            # Pad batch to multiple of dp_size (required for DP sharding)
            dp_size = twg.sharding_annotations.get_axis_size("data_parallel")
            actual_batch_size = sub_input_ids.shape[0]
            remainder = actual_batch_size % dp_size
            if remainder != 0:
                pad_count = dp_size - remainder
                # Repeat last row to fill — can't slice [:pad_count] when
                # actual_batch_size < pad_count (e.g., 1 sample, dp_size=4)
                pad_rows = sub_input_ids[-1:].expand(pad_count, -1)
                sub_input_ids = torch.cat([sub_input_ids, pad_rows], dim=0)
                if sub_lengths is not None:
                    sub_lengths = torch.cat(
                        [sub_lengths, sub_lengths[-1:].expand(pad_count)], dim=0
                    )

            sub_data = BatchedDataDict({"input_ids": sub_input_ids})
            if sub_lengths is not None:
                sub_data["input_lengths"] = sub_lengths

            # Serialize calls per teacher to prevent NCCL collective desync
            t_lock_start = time.time()
            with self._teacher_locks[group_key]:
                t_inference_start = time.time()
                logprobs_result = twg.get_logprobs(sub_data)
            t_done = time.time()
            lock_wait = t_inference_start - t_lock_start
            inference_time = t_done - t_inference_start
            print(
                f"[teacher_logprob] group={group_key} samples={actual_batch_size} "
                f"lock_wait={lock_wait:.2f}s inference={inference_time:.2f}s"
            )
            logprobs = logprobs_result["reference_logprobs"]

            # Trim DP padding
            logprobs = logprobs[:actual_batch_size]

            return indices, logprobs

        # Fan out to teachers in parallel
        t_total_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(group_to_indices)
        ) as executor:
            futures = {
                executor.submit(_get_logprobs_for_group, gk, idxs): gk
                for gk, idxs in group_to_indices.items()
            }
            for future in concurrent.futures.as_completed(futures):
                indices, logprobs = future.result()
                result[indices] = logprobs
        total_time = time.time() - t_total_start
        print(
            f"[teacher_logprob] total={total_time:.2f}s for {B} samples across {len(group_to_indices)} teacher(s)"
        )

        return result, total_time

    def _run_prompt_group_worker(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        target_weight_version: int,
        prompt_idx: int,
        filterable: bool = False,
        prompt_uid: Optional[int] = None,
    ) -> None:
        # [CUSTOM] What happened to the group: "discarded" (dynamic sampling),
        # "surplus" (target already full), "pushed", or None (error/shutdown).
        outcome: Optional[str] = None
        claim_target: Optional[int] = None
        try:
            # Import here to avoid circular dependency
            from nemo_rl.algorithms.grpo import _should_use_nemo_gym
            from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout

            # Run rollout for this prompt group
            # Async engine supports concurrent generation; avoid locking
            # Check if we should use nemo_gym (similar to synchronous GRPO)
            if _should_use_nemo_gym(self.master_config):
                generation_config = self.master_config.policy["generation"]
                nemo_gym_rollout_result = run_async_nemo_gym_rollout(
                    policy_generation=self.policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=self.task_to_env,
                    max_seq_len=self.master_config.policy["max_total_sequence_length"],
                    generation_config=generation_config,
                    max_rollout_turns=None,
                    greedy=False,
                )
                final_batch = nemo_gym_rollout_result.final_batch
                rollout_metrics = nemo_gym_rollout_result.rollout_metrics
            else:
                final_batch, rollout_metrics = run_async_multi_turn_rollout(
                    policy_generation=self.policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=self.task_to_env,
                    max_seq_len=self.master_config.policy["max_total_sequence_length"],
                    max_rollout_turns=self.master_config.grpo["max_rollout_turns"],
                    greedy=False,
                )

            # Move to CPU and push to buffer (avoid blocking on GC/push)
            final_batch_cpu = final_batch.to("cpu")
            del final_batch

            if "total_reward" in final_batch_cpu:
                raw_rewards = final_batch_cpu["total_reward"]
                # [CUSTOM] Solved check on the raw rewards, so a correct but
                # long answer still counts as solved after reward shaping.
                if self._skip_solved and bool(
                    (raw_rewards >= self._solved_threshold).all()
                ):
                    self._mark_solved(repeated_batch)
                # [CUSTOM] Reward scaling and shaping (e.g. DAPO's overlong
                # penalty) as in the sync loop, which applies them before
                # dynamic sampling; the async path skipped them.
                grpo_cfg = self.master_config.grpo
                if grpo_cfg.get("reward_scaling"):
                    final_batch_cpu = scale_rewards(
                        final_batch_cpu, grpo_cfg["reward_scaling"]
                    )
                shaping_cfg = grpo_cfg.get("reward_shaping")
                if shaping_cfg and shaping_cfg.get("enabled"):
                    final_batch_cpu = apply_reward_shaping(final_batch_cpu, shaping_cfg)
                degeneration_cfg = grpo_cfg.get("degeneration_penalty")
                if degeneration_cfg and degeneration_cfg["enabled"]:
                    final_batch_cpu = apply_degeneration_penalty(
                        final_batch_cpu, degeneration_cfg
                    )

            # [CUSTOM] Dynamic sampling: drop this group instead of pushing it if its
            # rewards are all equal (zero advantage, no training signal) and we're still
            # under the per-target launch cap decided at spawn time (see _process_batch).
            # _get_next_target_for_generation's existing gap-filling (via
            # ReplayBuffer.get_trajectories_needed) picks up the slack and launches a
            # replacement automatically on its next pass -- no bespoke retry logic needed.
            discarded = False
            if filterable:
                rewards = final_batch_cpu.get("total_reward")
                if rewards is not None:
                    reward_values = (
                        rewards.flatten().tolist()
                        if hasattr(rewards, "flatten")
                        else list(rewards)
                    )
                    with self._counter_lock:
                        self._unfiltered_reward_sum[target_weight_version] = (
                            self._unfiltered_reward_sum.get(target_weight_version, 0.0)
                            + sum(reward_values)
                        )
                        self._unfiltered_reward_count[target_weight_version] = (
                            self._unfiltered_reward_count.get(target_weight_version, 0)
                            + len(reward_values)
                        )
                    # As the sync loop (upstream #2478): filter on the reward
                    # before shaping, so a length penalty alone does not make
                    # an all-correct (or all-wrong) group look informative.
                    filter_rewards = final_batch_cpu.get(
                        "unshaped_total_reward", rewards
                    )
                    filter_values = (
                        filter_rewards.flatten().tolist()
                        if hasattr(filter_rewards, "flatten")
                        else list(filter_rewards)
                    )
                    if filter_values and max(filter_values) == min(filter_values):
                        discarded = True
                        with self._counter_lock:
                            self._discarded_per_target[target_weight_version] = (
                                self._discarded_per_target.get(target_weight_version, 0)
                                + 1
                            )

            if discarded:
                print(
                    f"🎲 Discarded zero-advantage group (prompt_idx {prompt_idx}, "
                    f"target_weight {target_weight_version}): dynamic sampling will "
                    "relaunch a replacement."
                )
                # [CUSTOM] Filtered out: the prompt is done with.
                outcome = "discarded"
                self._finish_prompt(prompt_uid)
            else:
                # [CUSTOM] Reserve the group's place before pushing it. With
                # groups_multiplier > 1 the target may already have its groups
                # (surplus); with flexible sampling such a group goes to the
                # lowest later target that needs one, otherwise it is dropped
                # and its prompt generated again.
                claim_target = self._claim_target(
                    target_weight_version, generation_weight_version
                )
                if claim_target is None:
                    print(
                        f"♻️ Surplus group dropped (prompt_idx {prompt_idx}, "
                        f"target_weight {target_weight_version}): target already has its groups"
                        + (
                            " and no later target within the age window needs one"
                            if self._reassign_surplus
                            else ""
                        )
                    )
                    with self._counter_lock:
                        self._surplus_per_target[target_weight_version] = (
                            self._surplus_per_target.get(target_weight_version, 0) + 1
                        )
                    outcome = "surplus"
                    self._requeue_prompt(prompt_uid)
                elif claim_target != target_weight_version:
                    print(
                        f"🔁 Surplus group reassigned (prompt_idx {prompt_idx}): "
                        f"target_weight {target_weight_version} -> {claim_target}"
                    )

            if claim_target is not None:
                # Compute teacher logprobs at collection time (overlapped with async rollouts)
                if self._has_distillation_teachers and "agent_ref" in final_batch_cpu:
                    agent_refs = final_batch_cpu["agent_ref"]
                    if isinstance(agent_refs, list):
                        from nemo_rl.data.llm_message_utils import (
                            batched_message_log_to_flat_message,
                        )

                        flat_for_teacher, teacher_input_lengths = (
                            batched_message_log_to_flat_message(
                                final_batch_cpu["message_log"],
                                pad_value_dict={
                                    "token_ids": self.tokenizer.pad_token_id
                                },
                                make_sequence_length_divisible_by=self._teacher_seq_pad_multiple,
                            )
                        )
                        teacher_logprobs, teacher_logprob_time = (
                            self._compute_teacher_logprobs(
                                flat_for_teacher["token_ids"],
                                agent_refs,
                                input_lengths=teacher_input_lengths,
                            )
                        )
                        # Store inside batch dict so from_batches handles
                        # variable-length padding across prompt groups
                        final_batch_cpu["teacher_reference_logprobs"] = teacher_logprobs
                        rollout_metrics = dict(rollout_metrics)
                        rollout_metrics["teacher_logprob_time"] = teacher_logprob_time

                trajectory_group = {
                    "batch": final_batch_cpu,
                    "rollout_metrics": rollout_metrics,
                    "timestamp": time.time(),
                    "prompt_uid": prompt_uid,
                }

                # Use exponential backoff when buffer is full
                try:
                    backoff_delay = 0.01
                    while self.running:
                        status = ray.get(
                            self.replay_buffer.add.remote(
                                trajectory_group,
                                generation_weight_version,
                                claim_target,
                            )
                        )
                        if status == "success":
                            outcome = "pushed"
                            self._release_claim(claim_target)
                            with self._counter_lock:
                                self._buffered_per_target[target_weight_version] = (
                                    self._buffered_per_target.get(
                                        target_weight_version, 0
                                    )
                                    + 1
                                )
                                buffered_count = self._buffered_per_target[
                                    target_weight_version
                                ]
                                spawned_count = self._spawned_per_target.get(
                                    target_weight_version, 0
                                )
                            print(
                                f"📦 Buffered per-prompt group (prompt_idx {prompt_idx}, "
                                f"target_weight {claim_target}) "
                                f"[{buffered_count}/{spawned_count} buffered]"
                            )
                            break
                        elif status == "full":
                            # Exponential backoff up to 0.5 second
                            time.sleep(min(backoff_delay, 0.5))
                            backoff_delay *= 1.5
                        else:
                            # Unexpected status, wait briefly
                            time.sleep(0.01)
                except Exception as e:
                    print(f"❌ Failed to enqueue per-prompt group to buffer: {e}")
                    import traceback

                    traceback.print_exc()
        except Exception as e:
            print(f"❌ Error in prompt group worker: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if outcome is None:
                # [CUSTOM] Error or shutdown: free the claim, if any, and keep the
                # prompt pending (saved at the next checkpoint); while running
                # it is generated again, up to max_prompt_retries times.
                if claim_target is not None:
                    self._release_claim(claim_target)
                if self.running:
                    self._requeue_prompt(prompt_uid, failed=True)
            with self._counter_lock:
                self._completed_per_target[target_weight_version] = (
                    self._completed_per_target.get(target_weight_version, 0) + 1
                )
            self._maybe_release_target(target_weight_version)

            # Detach thread record when finished
            with self._threads_lock:
                current = _threading.current_thread()
                if current in self._inflight_threads:
                    self._inflight_threads.remove(current)
            try:
                self._inflight_sema.release()
            except Exception:
                import traceback

                traceback.print_exc()
