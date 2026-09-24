# Custom override of nemo_rl.algorithms.async_utils
# Adds NemoGym rollout support to AsyncTrajectoryCollector._run_prompt_group_worker
#
# This file is mounted over the upstream module via sys.modules patching
# in custom/script/run_grpo.py.


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

import threading as _threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import ray
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface

TokenizerType = PreTrainedTokenizerBase


@dataclass
class _TargetSlots:
    """Per-target-weight bookkeeping of prompt-group slots in the collector.

    A target weight version needs exactly num_prompts_per_step accepted groups.
    A slot is taken when a prompt group is launched for the target and settled
    when the group is either pushed to the replay buffer (accepted) or dropped
    by dynamic sampling (discarded), which re-opens the slot for a new prompt.
    """

    accepted: int = 0
    inflight: int = 0
    launched: int = 0
    discarded: int = 0
    generated_samples: int = 0
    generated_reward_sum: float = 0.0


@ray.remote  # pragma: no cover
class ReplayBuffer:
    """Replay buffer storing per-prompt groups.

    A single entry corresponds to 1 prompt repeated by
    grpo.num_generations_per_prompt (required to compute per-prompt advantages).
    """

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self.trajectories = []  # List[dict[str, Any]]
        # If trajectory_version is 1 and target_weight_version is 4 it means that weight version 1 was used for generating a trajectory and this trajectory will be used for training when weight version is 4.
        self.trajectory_versions = []  # it is the weight-version used for generation of a trajectory
        self.target_weight_versions = []  # it is the weight-version of the trainer where this trajectory will be used.

        self.last_target_weight_already_generated = -1
        self._lock = _threading.Lock()

    def push_with_wait_signal(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
        target_weight_version: int,
    ) -> str:
        """Add a per-prompt trajectory group with metadata.

        Args:
            trajectory: data dict
            weight_version: version of the model weights used for generation
            target_weight_version: version of the model weights this trajectory is intended for training
        """
        with self._lock:
            if len(self.trajectories) >= self.max_size:
                return "full"

            print("🔍 ReplayBuffer.push_with_wait_signal: Adding trajectory")
            self.trajectories.append(trajectory)
            self.trajectory_versions.append(weight_version)
            self.target_weight_versions.append(target_weight_version)
            self.last_target_weight_already_generated = max(
                self.last_target_weight_already_generated, target_weight_version
            )
            print(
                f"ReplayBuffer state: {len(self.trajectories)} groups, versions={self.trajectory_versions}, targets={self.target_weight_versions}, last_target_weight_already_generated={self.last_target_weight_already_generated}"
            )
            return "success"

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        return {
            "total_trajectories": len(self.trajectories),
            "trajectory_versions": self.trajectory_versions,
            "target_weight_versions": self.target_weight_versions,
            "max_size": self.max_size,
        }

    def get_last_target_weight_already_generated(self) -> int:
        with self._lock:
            return self.last_target_weight_already_generated

    def get_existing_target_weights(self) -> set[int]:
        """Get set of target weight versions that already have trajectories."""
        with self._lock:
            return set(self.target_weight_versions)

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample per-prompt trajectory groups intended for the current training step.

        Only returns trajectories with target_weight_version == current_weight_version.
        If insufficient trajectories are available, returns None to stall training
        until the remaining trajectories are generated. This ensures no trajectory
        loses its last chance to be used for its intended training step.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or None if insufficient data
        """
        with self._lock:
            if not self.trajectories:
                return None

            total_trajectories = len(self.trajectories)
            print("🔍 ReplayBuffer sampling debug:")
            print(f"   {current_weight_version=}, {max_age_steps=}")
            print(f"   {self.trajectory_versions=}")

            # For debugging: check for unexpected old trajectories
            from collections import Counter

            version_counts = Counter(self.trajectory_versions)
            print(f"   {version_counts=}")

            # Compute minimum valid version based on age window
            # max_age_steps=1 means trajectories from the last 1 step are valid
            min_valid_version = max(0, current_weight_version - max_age_steps)
            print(f"   {min_valid_version=}")

            # Check for unexpected old trajectories
            old_trajectories = [
                v for v in self.trajectory_versions if v < min_valid_version
            ]
            if old_trajectories:
                raise ValueError(
                    f"Found {len(old_trajectories)} trajectories older than min_valid_version {min_valid_version}"
                )

            # Filter for valid trajectories without modifying the buffer
            valid_indices = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if min_valid_version <= v <= current_weight_version
            ]
            print(
                f"   valid_indices: {len(valid_indices)}/{total_trajectories} trajectories within age window"
            )
            if not valid_indices:
                print("No trajectories available for sampling.")
                return None

            # Enforce exact number of groups if available; otherwise, signal to wait
            if len(valid_indices) < num_prompt_groups:
                print(
                    f"Insufficient valid groups: have {len(valid_indices)}, need {num_prompt_groups}. Waiting for buffer to fill."
                )
                return None

            # Only select trajectories intended for the current training step
            # This ensures no trajectory loses its "last chance" to be used for its intended step
            intended_indices = [
                i
                for i in valid_indices
                if self.target_weight_versions[i] == current_weight_version
            ]

            print(
                f"   🎯 Found {len(intended_indices)} trajectories intended for current step {current_weight_version}"
            )

            # Stall training if we don't have enough trajectories intended for this step
            if len(intended_indices) < num_prompt_groups:
                print(
                    f"   ⏸️ STALLING: Need {num_prompt_groups} trajectories for step {current_weight_version}, but only {len(intended_indices)} are ready"
                )
                print(
                    f"   ⏸️ Training will wait for remaining {num_prompt_groups - len(intended_indices)} trajectories to be generated"
                )
                return None

            # Select exactly the trajectories intended for this step (FIFO within same target)
            selected: list[int] = intended_indices[:num_prompt_groups]
            print(
                f"   ✅ Selected {len(selected)} trajectories all intended for step {current_weight_version}"
            )

            from collections import Counter

            sampled_weights = [self.trajectory_versions[i] for i in selected]
            avg_trajectory_age = current_weight_version - sum(sampled_weights) / len(
                sampled_weights
            )
            print(
                f"✅ Selected counts by generation weight-version: {Counter(sampled_weights)}"
            )
            print(f"📊 Average trajectory age: {avg_trajectory_age:.2f} steps")
            print(
                f"🎯 All selected trajectories target step {current_weight_version} (100% target match)"
            )

            sampled_items = [self.trajectories[i] for i in selected]

            # Remove selected items in reverse order to maintain correct indices
            for idx in sorted(selected, reverse=True):
                self.trajectory_versions.pop(idx)
                self.target_weight_versions.pop(idx)
                self.trajectories.pop(idx)
            print(
                f"🗑️ Consumed and removed {len(selected)} groups from buffer, old buffer size: {total_trajectories}, new buffer size: {len(self.trajectories)}, new target weight versions {self.target_weight_versions}"
            )

            return {
                "trajectories": sampled_items,
                "avg_trajectory_age": avg_trajectory_age,
            }

    def size(self) -> int:
        """Return current buffer size."""
        with self._lock:
            return len(self.trajectories)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.trajectories.clear()
            self.trajectory_versions.clear()
            self.target_weight_versions.clear()


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
    ):
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.master_config = master_config
        self.replay_buffer = replay_buffer
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

        # Track threads
        self._inflight_threads: set[_threading.Thread] = set()
        self._threads_lock: _threading.Lock = _threading.Lock()

        # Limit in-flight generator requests to num_prompts_per_step * max_trajectory_age_steps
        # This value limits the parallelism of the generation requests.
        max_inflight = (
            int(self.master_config["grpo"]["num_prompts_per_step"])
            * int(self.master_config["grpo"]["async_grpo"]["max_trajectory_age_steps"])
        ) or 1
        self._inflight_sema = _threading.Semaphore(max_inflight)

        # Simple lock to prevent race conditions when checking/spawning workers
        self._generation_check_lock: _threading.Lock = _threading.Lock()
        # Signalled whenever a slot may have opened: a group settled, the weight
        # version advanced, or collection stopped.
        self._slot_cv = _threading.Condition(self._generation_check_lock)
        # Slot bookkeeping per target weight version, guarded by _slot_cv.
        self._target_slots: dict[int, _TargetSlots] = {}

        grpo_cfg = self.master_config["grpo"]
        self._groups_per_target = int(grpo_cfg["num_prompts_per_step"])
        # Dynamic sampling (DAPO): drop prompt groups whose rewards are all
        # equal, since they carry zero advantage, and generate a replacement
        # prompt for the same target. After dynamic_sampling_max_gen_batches *
        # num_prompts_per_step launches for one target, remaining groups are
        # accepted unfiltered so training cannot stall on a hard/easy stretch.
        self._use_dynamic_sampling = bool(grpo_cfg["use_dynamic_sampling"])
        self._max_filtered_launches_per_target = (
            self._groups_per_target * int(grpo_cfg["dynamic_sampling_max_gen_batches"])
            if self._use_dynamic_sampling
            else 0
        )

        # Track current epoch for checkpointing (0 means not started)
        self.current_epoch: int = 0

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
        async_cfg = self.master_config.get("grpo", {}).get("async_grpo", {})
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

    def _find_open_target(self) -> Optional[int]:
        """Return the lowest target weight with an open slot, or None.

        Caller must hold self._slot_cv. Candidates are the targets reachable from
        the current weight version plus any not-yet-consumed target that is still
        short of groups (e.g. after dynamic sampling discarded some of them, the
        trainer may already be waiting on that target).
        """
        candidates = set(self._calculate_target_weights(self.current_weight_version))
        candidates.update(
            t for t in self._target_slots if t >= self.current_weight_version
        )
        for target_weight in sorted(candidates):
            slots = self._target_slots.get(target_weight)
            if slots is None or (
                slots.accepted + slots.inflight < self._groups_per_target
            ):
                return target_weight
        return None

    def _reserve_slot(self) -> Optional[tuple[int, bool]]:
        """Block until some target weight has an open slot, then take it.

        Returns:
            (target_weight, filterable), where filterable says whether dynamic
            sampling may discard this group, or None if collection stopped.
        """
        with self._slot_cv:
            while self.running:
                target_weight = self._find_open_target()
                if target_weight is not None:
                    slots = self._target_slots.setdefault(target_weight, _TargetSlots())
                    slots.inflight += 1
                    slots.launched += 1
                    filterable = (
                        self._use_dynamic_sampling
                        and slots.launched <= self._max_filtered_launches_per_target
                    )
                    return target_weight, filterable

                if self._last_limit_warning_version != self.current_weight_version:
                    print(
                        f"⏸️ Pausing collection: all target weights reachable from weight version "
                        f"{self.current_weight_version} are fully generated or in progress. Waiting..."
                    )
                    self._last_limit_warning_version = self.current_weight_version
                # Timeout so a stop request is noticed even without a notify.
                self._slot_cv.wait(timeout=1.0)
        return None

    def _settle_slot(
        self, target_weight: int, *, accepted: bool, discarded: bool
    ) -> None:
        """Release an in-flight slot of target_weight and wake waiting launchers."""
        with self._slot_cv:
            slots = self._target_slots.get(target_weight)
            if slots is None:
                # Already consumed and forgotten: the push landed and the trainer
                # moved past this target before the slot was settled.
                return
            slots.inflight -= 1
            if accepted:
                slots.accepted += 1
            if discarded:
                slots.discarded += 1
            self._slot_cv.notify_all()

    def get_target_stats(self, target_weight: int) -> dict[str, float]:
        """Return generation statistics for a consumed target weight.

        Called by the trainer after it sampled the groups of target_weight, when
        no more generation for that target can happen. The entry itself is kept
        until the weight version moves past it (see set_weight_version), so the
        target is never mistaken for a new, ungenerated one.
        """
        with self._slot_cv:
            slots = self._target_slots.get(target_weight)
        if slots is None:
            return {}
        stats = {
            "dynamic_sampling_num_gen_batches": slots.launched / self._groups_per_target,
            "dynamic_sampling_num_discarded_groups": float(slots.discarded),
        }
        if slots.generated_samples > 0:
            stats["unfiltered_reward"] = (
                slots.generated_reward_sum / slots.generated_samples
            )
        return stats

    def set_weight_version(self, version: int) -> None:
        with self._slot_cv:
            self.current_weight_version = version
            # Targets below the new version were consumed and can never be
            # candidates again (see _find_open_target), so forget them.
            for consumed_target in [t for t in self._target_slots if t < version]:
                del self._target_slots[consumed_target]
            self._slot_cv.notify_all()
        print(f"🔄 Updated weight version to {version}")

    def start_collection(self, dataloader: StatefulDataLoader) -> None:
        """Start collecting trajectories from dataloader."""
        self.running = True
        self.dataloader = dataloader

        print("Started continuous trajectory collection")

        self.collection_thread = _threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        print("Collection thread started, start_collection returning")

    def _collection_loop(self):
        """Run the collection loop in background thread.

        This loop runs indefinitely until self.running is set to False,
        restarting from epoch 0 each time the dataloader is exhausted.
        """
        try:
            while self.running:
                self.current_epoch += 1
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

                for batch in self.dataloader:
                    if not self.running:
                        break

                    # Check if manually paused and wait
                    if not self._manual_pause_cleared.is_set() and self.running:
                        self._manual_pause_cleared.wait()

                    # Check if refit is in progress and wait
                    if not self._refit_pause_cleared.is_set() and self.running:
                        print("⏸️ Pausing collection for refit...")
                        self._refit_pause_cleared.wait()
                        print("▶️ Refit completed, resuming collection")

                    if not self.running:
                        break

                    # Blocks per prompt until a target weight has an open slot.
                    self._process_batch(batch)

                if self.running:
                    print(
                        f"Completed epoch {self.current_epoch}, starting next epoch..."
                    )

        except Exception as e:
            print(f"❌ Error in trajectory collection: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.running = False
            print("🛑 Trajectory collection stopped")

    def _process_batch(self, batch: BatchedDataDict[DatumSpec]) -> None:
        """Launch one prompt group per prompt in batch, each for an open target slot.

        Each prompt reserves a slot of the lowest target weight that still needs
        groups, so one dataloader batch may feed several targets, and a target
        whose groups were discarded by dynamic sampling gets refilled first.
        """
        try:
            num_generations = self.master_config["grpo"]["num_generations_per_prompt"]

            for prompt_idx in range(batch.size):
                # Validation pauses collection; don't start new groups meanwhile.
                if not self._manual_pause_cleared.is_set() and self.running:
                    self._manual_pause_cleared.wait()

                single_prompt_batch = batch.slice(prompt_idx, prompt_idx + 1)
                repeated_batch = single_prompt_batch.repeat_interleave(num_generations)

                # A reserved slot is only released by the worker, so keep the
                # code between here and worker.start() free of fallible work.
                reservation = self._reserve_slot()
                if reservation is None:
                    return
                target_weight, filterable = reservation

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

                # Read after any refit wait so trajectories carry the weights that
                # actually generate them. The reserved target stays valid: it is
                # still unconsumed because this slot keeps it short of groups.
                generation_weight_version = self.current_weight_version
                print(
                    f"🎯 Generating for target weight {target_weight} from generation_weight_version {generation_weight_version}"
                )

                self._inflight_sema.acquire()
                worker = _threading.Thread(
                    target=self._run_prompt_group_worker,
                    args=(
                        repeated_batch,
                        generation_weight_version,
                        target_weight,
                        prompt_idx,
                        filterable,
                    ),
                    daemon=True,
                )
                with self._threads_lock:
                    self._inflight_threads.add(worker)
                worker.start()

            self._cleanup_finished_threads()

        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            import traceback

            traceback.print_exc()

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

        For vLLM V1 async engine, leverages in-flight weight updates via collective_rpc,
        allowing ongoing generations to continue with their current KV caches while
        weights are updated. This significantly improves async performance.

        For non-async engines, waits for all pending generations to complete before refit.
        """
        start_time = time.time()
        print("🔄 Preparing for refit: pausing new generations...")

        # Pause new generation starts
        self._refit_pause_cleared.clear()
        print("⏸️ New generation starts paused")

        # Check if we're using vLLM async engine
        vllm_cfg = (
            self.master_config.get("policy", {})
            .get("generation", {})
            .get("vllm_cfg", {})
        )
        is_async_engine = vllm_cfg.get("async_engine", False)
        in_flight_weight_updates = (
            self.master_config.get("grpo", {})
            .get("async_grpo", {})
            .get("in_flight_weight_updates", False)
        )

        if is_async_engine and in_flight_weight_updates:
            # vLLM V1 async engine supports in-flight weight updates
            # Ongoing generations will continue with their current KV caches
            # New generations (after weight update) will use the updated weights
            print(
                "🚀 Using vLLM V1 in-flight weight update - skipping wait for pending generations"
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
        async_cfg = self.master_config.get("grpo", {}).get("async_grpo", {})
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
        """Get the current dataloader state for checkpointing.

        Returns a dict containing:
            - dataloader_state: The StatefulDataLoader's state_dict
            - current_epoch: The current epoch number (1-indexed)
        """
        state = {"current_epoch": self.current_epoch}
        if hasattr(self, "dataloader") and hasattr(self.dataloader, "state_dict"):
            state["dataloader_state"] = self.dataloader.state_dict()
        return state

    def set_dataloader_state(self, state: dict) -> None:
        """Restore dataloader state from checkpoint.

        Args:
            state: Dict containing 'current_epoch' and optionally 'dataloader_state'
        """
        if "current_epoch" in state:
            self.current_epoch = state["current_epoch"]
            print(f"Restored epoch to {self.current_epoch}")

        if (
            "dataloader_state" in state
            and hasattr(self, "dataloader")
            and hasattr(self.dataloader, "load_state_dict")
        ):
            dl_state = state["dataloader_state"]
            num_workers = getattr(self.dataloader, "num_workers", 0) or 0
            # torchdata's _StatefulMultiProcessingDataLoaderIter asserts
            # "_snapshot" in next_iter_state; if the saved state lacks it we
            # must NOT load it or iter() will raise and kill the collector.
            if num_workers > 0 and isinstance(dl_state, dict) and "_snapshot" not in dl_state:
                print(
                    "⚠️  Saved dataloader state is missing '_snapshot'; "
                    "skipping restore and resuming with a fresh iterator."
                )
            else:
                try:
                    self.dataloader.load_state_dict(dl_state)
                    print("Restored dataloader state")
                except AssertionError as e:
                    print(
                        f"⚠️  Skipping dataloader state restore ({e}); resuming with a fresh iterator."
                    )

    def _cleanup_finished_threads(self) -> None:
        with self._threads_lock:
            finished = {t for t in self._inflight_threads if not t.is_alive()}
            for t in finished:
                self._inflight_threads.remove(t)

    def _run_prompt_group_worker(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        target_weight_version: int,
        prompt_idx: int,
        filterable: bool,
    ) -> None:
        # Whether this worker's slot was accepted/discarded; if neither happens
        # (error, shutdown) the slot is re-opened so the target is not starved.
        slot_settled = False
        try:
            # Check if NemoGym should be used
            env_config = self.master_config.get("env", {})
            use_nemo_gym = bool(env_config.get("should_use_nemo_gym"))

            if use_nemo_gym:
                # NemoGym handles rollouts via its own environment (chat completions API)
                generation_config = self.master_config["policy"]["generation"]
                nemo_gym_result = run_async_nemo_gym_rollout(
                    policy_generation=self.policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=self.task_to_env,
                    generation_config=generation_config,
                )
                final_batch = nemo_gym_result.final_batch
                rollout_metrics = nemo_gym_result.rollout_metrics
            else:
                # Standard multi-turn rollout with direct vLLM generation
                final_batch, rollout_metrics = run_async_multi_turn_rollout(
                    policy_generation=self.policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=self.task_to_env,
                    max_seq_len=self.master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=self.master_config["grpo"]["max_rollout_turns"],
                    greedy=False,
                )

            # Move to CPU and push to buffer (avoid blocking on GC/push)
            final_batch_cpu = final_batch.to("cpu")
            del final_batch

            # Dynamic sampling decides at group level: a group whose rewards are
            # all equal has zero advantage for every sample. (A per-sample
            # leave-one-out std would also be zero for the lone success in a
            # 1-of-N group, which must be kept.)
            rewards = final_batch_cpu["total_reward"]
            with self._slot_cv:
                slots = self._target_slots.get(target_weight_version)
                if slots is not None:
                    slots.generated_samples += rewards.numel()
                    slots.generated_reward_sum += float(rewards.sum())
            if filterable and bool((rewards == rewards[0]).all()):
                print(
                    f"🗑️ Dynamic sampling: discarded zero-variance group (prompt_idx {prompt_idx}, "
                    f"target_weight {target_weight_version}, reward {float(rewards[0]):.3f})"
                )
                self._settle_slot(target_weight_version, accepted=False, discarded=True)
                slot_settled = True
                return

            trajectory_group = {
                "batch": final_batch_cpu,
                "rollout_metrics": rollout_metrics,
                "timestamp": time.time(),
            }

            # Use exponential backoff when buffer is full
            try:
                backoff_delay = 0.01
                while self.running:
                    status = ray.get(
                        self.replay_buffer.push_with_wait_signal.remote(
                            trajectory_group,
                            generation_weight_version,
                            target_weight_version,
                        )
                    )
                    if status == "success":
                        print(
                            f"📦 Buffered per-prompt group (prompt_idx {prompt_idx}, target_weight {target_weight_version})"
                        )
                        self._settle_slot(
                            target_weight_version, accepted=True, discarded=False
                        )
                        slot_settled = True
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
            # Re-open the slot on error or shutdown so another prompt fills it.
            if not slot_settled:
                self._settle_slot(target_weight_version, accepted=False, discarded=False)
                print(
                    f"🧹 Released unfilled slot for target weight {target_weight_version} (prompt_idx {prompt_idx})"
                )

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
