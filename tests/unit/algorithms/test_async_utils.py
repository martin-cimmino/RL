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

import io
import os
import tempfile
import threading
import unittest.mock as mock
from typing import Optional

import pytest
import ray
import torch

# Set up Ray temp directory before any Ray operations
# Try multiple approaches to ensure Ray uses a writable directory
_temp_dir = tempfile.mkdtemp(prefix="ray_async_test_")
os.environ["RAY_TEMP_DIR"] = _temp_dir
os.environ["RAY_TMPDIR"] = _temp_dir  # Alternative env var
os.environ["TMPDIR"] = _temp_dir  # System temp dir

from nemo_rl.algorithms.async_utils import (
    AsyncTrajectoryCollector,
    ReplayBuffer,
    _TargetSlots,
)
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


@ray.remote(num_cpus=0)
class MockEnvironment(EnvironmentInterface):
    """Mock environment for testing async utilities."""

    def __init__(self, rewards: list[float]):
        self.rewards = rewards
        self._calls = 0

    def step(
        self, messages: list[LLMMessageLogType], env_info: list[dict]
    ) -> EnvironmentReturn:
        self._calls += 1
        return (
            [{"role": "environment", "content": "observation"}] * len(messages),
            [{}] * len(messages),
            [[]] * len(messages),
            self.rewards,
            [True] * len(messages),
            [None] * len(messages),
        )

    def get_calls(self):
        return self._calls

    def reset_calls(self):
        self._calls = 0
        return True

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        return batch, {}


class MockGenerationInterface:
    """Mock generation interface for testing."""

    def __init__(self):
        self.prepare_calls = 0
        self.finish_calls = 0

    def prepare_for_generation(self, **kwargs):
        self.prepare_calls += 1

    def finish_generation(self):
        self.finish_calls += 1


class TestReplayBuffer:
    """Test cases for ReplayBuffer."""

    def test_replay_buffer_initialization(self):
        """Test ReplayBuffer initialization."""
        buffer = ReplayBuffer.remote(max_size=10)
        size = ray.get(buffer.size.remote())
        assert size == 0

        debug_info = ray.get(buffer.get_debug_info.remote())
        assert debug_info["total_trajectories"] == 0
        assert debug_info["max_size"] == 10
        assert debug_info["trajectory_versions"] == []
        assert debug_info["target_weight_versions"] == []

        ray.kill(buffer)

    def test_replay_buffer_push_and_size(self):
        """Test pushing trajectories to buffer."""
        buffer = ReplayBuffer.remote(max_size=3)

        # Create mock trajectories
        trajectory1 = {"batch": {"data": "test1"}, "rollout_metrics": {"reward": 1.0}}
        trajectory2 = {"batch": {"data": "test2"}, "rollout_metrics": {"reward": 2.0}}

        # Push trajectories
        status1 = ray.get(
            buffer.push_with_wait_signal.remote(
                trajectory1, weight_version=0, target_weight_version=1
            )
        )
        assert status1 == "success"

        status2 = ray.get(
            buffer.push_with_wait_signal.remote(
                trajectory2, weight_version=1, target_weight_version=2
            )
        )
        assert status2 == "success"

        # Check size
        size = ray.get(buffer.size.remote())
        assert size == 2

        # Check debug info
        debug_info = ray.get(buffer.get_debug_info.remote())
        assert debug_info["total_trajectories"] == 2
        assert debug_info["trajectory_versions"] == [0, 1]
        assert debug_info["target_weight_versions"] == [1, 2]

        ray.kill(buffer)

    def test_replay_buffer_max_size_limit(self):
        """Test that buffer respects max size limit."""
        buffer = ReplayBuffer.remote(max_size=2)

        # Fill buffer to capacity
        trajectory1 = {"batch": {"data": "test1"}, "rollout_metrics": {"reward": 1.0}}
        trajectory2 = {"batch": {"data": "test2"}, "rollout_metrics": {"reward": 2.0}}
        trajectory3 = {"batch": {"data": "test3"}, "rollout_metrics": {"reward": 3.0}}

        # Push first two trajectories
        status1 = ray.get(
            buffer.push_with_wait_signal.remote(
                trajectory1, weight_version=0, target_weight_version=1
            )
        )
        status2 = ray.get(
            buffer.push_with_wait_signal.remote(
                trajectory2, weight_version=1, target_weight_version=2
            )
        )
        assert status1 == "success"
        assert status2 == "success"

        # Try to push third trajectory (should return "full")
        status3 = ray.get(
            buffer.push_with_wait_signal.remote(
                trajectory3, weight_version=2, target_weight_version=3
            )
        )
        assert status3 == "full"

        # Size should still be 2
        size = ray.get(buffer.size.remote())
        assert size == 2

        ray.kill(buffer)

    def test_replay_buffer_sampling_basic(self):
        """Test basic trajectory sampling."""
        buffer = ReplayBuffer.remote(max_size=10)

        # Push trajectories with different weight versions
        trajectories = []
        for i in range(3):
            trajectory = {
                "batch": {"data": f"test{i}"},
                "rollout_metrics": {"reward": float(i)},
            }
            trajectories.append(trajectory)
            ray.get(
                buffer.push_with_wait_signal.remote(
                    trajectory, weight_version=i, target_weight_version=i + 1
                )
            )

        # Sample trajectories intended for current step 2
        sample_result = ray.get(
            buffer.sample.remote(
                num_prompt_groups=1,
                current_weight_version=2,
                max_age_steps=2,
            )
        )

        assert sample_result is not None
        assert len(sample_result["trajectories"]) == 1
        assert "avg_trajectory_age" in sample_result

        # The trajectory should be intended for step 2 (target_weight_version=2)
        # But we pushed with target_weight_version=i+1, so trajectory at i=1 has target=2
        sampled_trajectory = sample_result["trajectories"][0]
        assert sampled_trajectory["batch"]["data"] == "test1"

        ray.kill(buffer)

    def test_replay_buffer_sampling_insufficient_trajectories(self):
        """Test sampling when insufficient trajectories are available."""
        buffer = ReplayBuffer.remote(max_size=10)

        # Push only one trajectory
        trajectory = {"batch": {"data": "test"}, "rollout_metrics": {"reward": 1.0}}
        ray.get(
            buffer.push_with_wait_signal.remote(
                trajectory, weight_version=0, target_weight_version=1
            )
        )

        # Try to sample more trajectories than available for current step
        sample_result = ray.get(
            buffer.sample.remote(
                num_prompt_groups=2,  # Request 2 but only 1 available
                current_weight_version=1,
                max_age_steps=1,
            )
        )

        assert sample_result is None  # Should return None when insufficient

        ray.kill(buffer)

    def test_replay_buffer_age_filtering(self):
        """Test that old trajectories are filtered out."""
        buffer = ReplayBuffer.remote(max_size=10)

        # Push trajectories with different ages
        old_trajectory = {"batch": {"data": "old"}, "rollout_metrics": {"reward": 1.0}}
        recent_trajectory = {
            "batch": {"data": "recent"},
            "rollout_metrics": {"reward": 2.0},
        }

        ray.get(
            buffer.push_with_wait_signal.remote(
                old_trajectory, weight_version=0, target_weight_version=1
            )
        )
        ray.get(
            buffer.push_with_wait_signal.remote(
                recent_trajectory, weight_version=2, target_weight_version=3
            )
        )

        # Sample with current_weight_version=3 and max_age_steps=1
        # This should filter out the trajectory with weight_version=0 (too old)
        with pytest.raises(
            ValueError, match="Found .* trajectories older than min_valid_version"
        ):
            ray.get(
                buffer.sample.remote(
                    num_prompt_groups=1,
                    current_weight_version=3,
                    max_age_steps=1,
                )
            )

        ray.kill(buffer)

    def test_replay_buffer_target_weight_matching(self):
        """Test that sampling only returns trajectories intended for current step."""
        buffer = ReplayBuffer.remote(max_size=10)

        # Push trajectories intended for different target steps
        trajectory1 = {
            "batch": {"data": "for_step_1"},
            "rollout_metrics": {"reward": 1.0},
        }
        trajectory2 = {
            "batch": {"data": "for_step_2"},
            "rollout_metrics": {"reward": 2.0},
        }

        ray.get(
            buffer.push_with_wait_signal.remote(
                trajectory1, weight_version=0, target_weight_version=1
            )
        )
        ray.get(
            buffer.push_with_wait_signal.remote(
                trajectory2, weight_version=1, target_weight_version=2
            )
        )

        # Sample for current step 1 - should only get trajectory intended for step 1
        sample_result = ray.get(
            buffer.sample.remote(
                num_prompt_groups=1,
                current_weight_version=1,
                max_age_steps=2,
            )
        )

        assert sample_result is not None
        assert len(sample_result["trajectories"]) == 1
        assert sample_result["trajectories"][0]["batch"]["data"] == "for_step_1"

        ray.kill(buffer)

    def test_replay_buffer_get_existing_target_weights(self):
        """Test getting existing target weight versions."""
        buffer = ReplayBuffer.remote(max_size=10)

        # Initially empty
        existing_weights = ray.get(buffer.get_existing_target_weights.remote())
        assert existing_weights == set()

        # Push trajectories with different target weights
        trajectory1 = {"batch": {"data": "test1"}, "rollout_metrics": {"reward": 1.0}}
        trajectory2 = {"batch": {"data": "test2"}, "rollout_metrics": {"reward": 2.0}}

        ray.get(
            buffer.push_with_wait_signal.remote(
                trajectory1, weight_version=0, target_weight_version=1
            )
        )
        ray.get(
            buffer.push_with_wait_signal.remote(
                trajectory2, weight_version=1, target_weight_version=3
            )
        )

        existing_weights = ray.get(buffer.get_existing_target_weights.remote())
        assert existing_weights == {1, 3}

        ray.kill(buffer)

    def test_replay_buffer_clear(self):
        """Test clearing the buffer."""
        buffer = ReplayBuffer.remote(max_size=10)

        # Push some trajectories
        trajectory = {"batch": {"data": "test"}, "rollout_metrics": {"reward": 1.0}}
        ray.get(
            buffer.push_with_wait_signal.remote(
                trajectory, weight_version=0, target_weight_version=1
            )
        )

        # Verify buffer has content
        size = ray.get(buffer.size.remote())
        assert size == 1

        # Clear buffer
        ray.get(buffer.clear.remote())

        # Verify buffer is empty
        size = ray.get(buffer.size.remote())
        assert size == 0

        debug_info = ray.get(buffer.get_debug_info.remote())
        assert debug_info["total_trajectories"] == 0
        assert debug_info["trajectory_versions"] == []
        assert debug_info["target_weight_versions"] == []

        ray.kill(buffer)


class TestAsyncTrajectoryCollector:
    """Test cases for AsyncTrajectoryCollector."""

    def create_mock_config(self) -> MasterConfig:
        """Create a mock master config for testing."""
        return {
            "grpo": {
                "num_prompts_per_step": 2,
                "num_generations_per_prompt": 3,
                "max_rollout_turns": 1,
                "use_dynamic_sampling": False,
                "async_grpo": {"max_trajectory_age_steps": 2},
            },
            "policy": {"max_total_sequence_length": 512},
        }

    def create_mock_batch(self, size: int = 2) -> BatchedDataDict[DatumSpec]:
        """Create a mock batch for testing."""
        message_logs = []
        for i in range(size):
            message_logs.append(
                [
                    {"role": "user", "content": f"Test prompt {i}"},
                ]
            )

        return BatchedDataDict[DatumSpec](
            {
                "task_name": ["test"] * size,
                "message_log": message_logs,
                "extra_env_info": [{}] * size,
                "loss_multiplier": torch.ones(size),
            }
        )

    def test_async_trajectory_collector_initialization(self):
        """Test AsyncTrajectoryCollector initialization."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Test basic functionality
        weight_version = ray.get(collector.get_weight_version.remote())
        assert weight_version == 0

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)

    def test_async_trajectory_collector_weight_version_updates(self):
        """Test weight version updates in trajectory collector."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Update weight version
        ray.get(collector.set_weight_version.remote(5))
        weight_version = ray.get(collector.get_weight_version.remote())
        assert weight_version == 5

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)

    def test_async_trajectory_collector_pause_resume(self):
        """Test pause and resume functionality."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Test pause and resume (these should not raise errors)
        ray.get(collector.pause.remote())
        ray.get(collector.resume.remote())

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)

    def test_async_trajectory_collector_prepare_for_refit(self):
        """Test prepare for refit functionality."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Test prepare for refit (should complete without hanging)
        ray.get(collector.prepare_for_refit.remote())
        ray.get(collector.resume_after_refit.remote())

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)

    def test_calculate_target_weights(self):
        """Test target weight calculation logic."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Test target weight calculation with different scenarios
        # Note: We can't directly test the private method, but we can test its effects
        # through the public interface behavior

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)

    def test_dataloader_state_retrieval(self):
        """Test getting dataloader state for checkpointing."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Test getting dataloader state (should return empty dict when no dataloader)
        state = ray.get(collector.get_dataloader_state.remote())
        assert isinstance(state, dict)

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)


class TestAsyncUtilsIntegration:
    """Integration tests for async utilities working together."""

    def create_mock_config(self) -> MasterConfig:
        """Create a mock master config for testing."""
        return {
            "grpo": {
                "num_prompts_per_step": 2,
                "num_generations_per_prompt": 2,
                "max_rollout_turns": 1,
                "use_dynamic_sampling": False,
                "async_grpo": {"max_trajectory_age_steps": 1},
            },
            "policy": {"max_total_sequence_length": 512},
        }

    def create_mock_batch(self, size: int = 2) -> BatchedDataDict[DatumSpec]:
        """Create a mock batch for testing."""
        message_logs = []
        for i in range(size):
            message_logs.append(
                [
                    {"role": "user", "content": f"Test prompt {i}"},
                ]
            )

        return BatchedDataDict[DatumSpec](
            {
                "task_name": ["test"] * size,
                "message_log": message_logs,
                "extra_env_info": [{}] * size,
                "loss_multiplier": torch.ones(size),
            }
        )

    def test_buffer_and_collector_integration(self):
        """Test that buffer and collector work together correctly."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Verify initial state
        buffer_size = ray.get(buffer.size.remote())
        assert buffer_size == 0

        weight_version = ray.get(collector.get_weight_version.remote())
        assert weight_version == 0

        # Test weight version synchronization
        ray.get(collector.set_weight_version.remote(3))
        updated_version = ray.get(collector.get_weight_version.remote())
        assert updated_version == 3

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)

    def test_concurrent_operations(self):
        """Test that concurrent operations don't cause race conditions."""
        buffer = ReplayBuffer.remote(max_size=5)

        # Push trajectories concurrently from multiple threads
        def push_trajectory(buffer, trajectory_id):
            trajectory = {
                "batch": {"data": f"test{trajectory_id}"},
                "rollout_metrics": {"reward": float(trajectory_id)},
            }
            return ray.get(
                buffer.push_with_wait_signal.remote(
                    trajectory,
                    weight_version=trajectory_id,
                    target_weight_version=trajectory_id + 1,
                )
            )

        # Use threading to simulate concurrent pushes
        threads = []
        results = []

        def worker(traj_id):
            result = push_trajectory(buffer, traj_id)
            results.append(result)

        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All pushes should succeed
        assert all(result == "success" for result in results)

        # Buffer should have correct size
        final_size = ray.get(buffer.size.remote())
        assert final_size == 3

        ray.kill(buffer)

    def test_error_handling(self):
        """Test error handling in async utilities."""
        # Test with invalid buffer size
        with pytest.raises(Exception):
            buffer = ReplayBuffer.remote(max_size=-1)
            ray.get(buffer.size.remote())

        # Test buffer operations
        buffer = ReplayBuffer.remote(max_size=1)

        # Test sampling from empty buffer
        sample_result = ray.get(
            buffer.sample.remote(
                num_prompt_groups=1,
                current_weight_version=0,
                max_age_steps=1,
            )
        )
        assert sample_result is None

        ray.kill(buffer)


class TestCollectorSlotScheduling:
    """Slot scheduling and dynamic sampling in AsyncTrajectoryCollector.

    Exercises the plain (non-actor) class so the scheduling state can be driven
    and inspected directly; rollouts and replay-buffer pushes are mocked.
    """

    def make_collector(
        self,
        *,
        use_dynamic_sampling: bool,
        num_prompts_per_step: int = 2,
        max_gen_batches: int = 3,
        max_age: int = 1,
        groups_multiplier: Optional[float] = None,
        flexible_sampling: bool = False,
    ):
        async_cfg = {"max_trajectory_age_steps": max_age}
        if groups_multiplier is not None:
            async_cfg["groups_multiplier"] = groups_multiplier
        if flexible_sampling:
            async_cfg["flexible_sampling"] = True
        grpo_cfg = {
            "num_prompts_per_step": num_prompts_per_step,
            "num_generations_per_prompt": 4,
            "max_rollout_turns": 1,
            "use_dynamic_sampling": use_dynamic_sampling,
            "async_grpo": async_cfg,
        }
        if use_dynamic_sampling:
            grpo_cfg["dynamic_sampling_max_gen_batches"] = max_gen_batches
        master_config = {
            "grpo": grpo_cfg,
            "policy": {"max_total_sequence_length": 512},
        }
        replay_buffer = mock.MagicMock()
        replay_buffer.push_with_wait_signal.remote.return_value = "success"
        collector = AsyncTrajectoryCollector.__ray_actor_class__(
            policy_generation=MockGenerationInterface(),
            tokenizer=mock.MagicMock(),
            task_to_env={},
            master_config=master_config,
            replay_buffer=replay_buffer,
            start_step=0,
        )
        collector.running = True
        return collector, replay_buffer

    def run_group(self, collector, *, target, filterable, rewards, prompt_uid=None):
        """Run one prompt-group worker whose rollout returns the given rewards."""
        final_batch = BatchedDataDict({"total_reward": torch.tensor(rewards)})
        collector._inflight_sema.acquire()
        with (
            mock.patch(
                "nemo_rl.algorithms.async_utils.run_async_multi_turn_rollout",
                return_value=(final_batch, {}),
            ),
            mock.patch("nemo_rl.algorithms.async_utils.ray.get", side_effect=lambda x: x),
        ):
            collector._run_prompt_group_worker(
                BatchedDataDict({}), 0, target, 0, filterable, prompt_uid
            )

    def test_without_dynamic_sampling_each_target_gets_exactly_its_groups(self):
        collector, buffer = self.make_collector(use_dynamic_sampling=False)

        reservations = [collector._reserve_slot() for _ in range(4)]

        # Initial window is targets [0, 1]; two groups each, never filterable.
        assert reservations == [(0, False), (0, False), (1, False), (1, False)]
        with collector._slot_cv:
            assert collector._find_open_target() is None

        self.run_group(collector, target=0, filterable=False, rewards=[1.0] * 4)
        assert buffer.push_with_wait_signal.remote.call_count == 1
        assert collector._target_slots[0].accepted == 1
        # An accepted group does not re-open the slot.
        with collector._slot_cv:
            assert collector._find_open_target() is None

    def test_zero_variance_group_is_discarded_and_slot_reopened(self):
        collector, buffer = self.make_collector(use_dynamic_sampling=True)
        for _ in range(4):
            collector._reserve_slot()

        self.run_group(collector, target=0, filterable=True, rewards=[0.0] * 4)

        buffer.push_with_wait_signal.remote.assert_not_called()
        assert collector._target_slots[0].discarded == 1
        # Target 0 is short again and is refilled before any later target.
        assert collector._reserve_slot() == (0, True)

    def test_lone_success_group_is_kept(self):
        # Leave-one-out std is zero for the lone success; the group-level test
        # must still keep the group.
        collector, buffer = self.make_collector(use_dynamic_sampling=True)
        collector._reserve_slot()

        self.run_group(collector, target=0, filterable=True, rewards=[1.0, 0.0, 0.0, 0.0])

        assert buffer.push_with_wait_signal.remote.call_count == 1
        assert collector._target_slots[0].accepted == 1
        assert collector._target_slots[0].discarded == 0

    def test_launch_cap_turns_filtering_off(self):
        collector, _ = self.make_collector(
            use_dynamic_sampling=True, num_prompts_per_step=1, max_gen_batches=2
        )

        first = collector._reserve_slot()
        self.run_group(collector, target=0, filterable=first[1], rewards=[0.0] * 4)
        second = collector._reserve_slot()
        self.run_group(collector, target=0, filterable=second[1], rewards=[0.0] * 4)
        third = collector._reserve_slot()

        # 2 launches x 1 prompt per step may be filtered; the 3rd is accepted as is.
        assert first == (0, True)
        assert second == (0, True)
        assert third == (0, False)
        self.run_group(collector, target=0, filterable=third[1], rewards=[0.0] * 4)
        assert collector._target_slots[0].accepted == 1

    def test_short_current_target_stays_open_after_version_advances(self):
        collector, _ = self.make_collector(use_dynamic_sampling=True, max_age=1)
        for _ in range(4):
            collector._reserve_slot()
        self.run_group(collector, target=1, filterable=True, rewards=[1.0, 0.0, 1.0, 0.0])
        self.run_group(collector, target=1, filterable=True, rewards=[1.0] * 4)

        collector.set_weight_version(1)

        # Target 0 is forgotten; target 1 is outside the window [2] of version 1
        # but still short of one group, so it is refilled first.
        assert 0 not in collector._target_slots
        assert collector._reserve_slot() == (1, True)

    def test_target_stats_report_all_generated_groups(self):
        collector, _ = self.make_collector(use_dynamic_sampling=True)
        for _ in range(2):
            collector._reserve_slot()
        self.run_group(collector, target=0, filterable=True, rewards=[0.0] * 4)
        collector._reserve_slot()
        self.run_group(collector, target=0, filterable=True, rewards=[1.0, 0.0, 0.0, 0.0])
        self.run_group(collector, target=0, filterable=True, rewards=[1.0, 1.0, 0.0, 0.0])

        stats = collector.get_target_stats(0)

        assert stats["dynamic_sampling_num_gen_batches"] == 1.5
        assert stats["dynamic_sampling_num_discarded_groups"] == 1.0
        assert stats["unfiltered_reward"] == pytest.approx(3.0 / 12.0)

    def test_failed_worker_reopens_slot(self):
        collector, buffer = self.make_collector(use_dynamic_sampling=False)
        for _ in range(4):
            collector._reserve_slot()

        collector._inflight_sema.acquire()
        with mock.patch(
            "nemo_rl.algorithms.async_utils.run_async_multi_turn_rollout",
            side_effect=RuntimeError("rollout failed"),
        ):
            collector._run_prompt_group_worker(BatchedDataDict({}), 0, 0, 0, False)

        buffer.push_with_wait_signal.remote.assert_not_called()
        assert collector._reserve_slot() == (0, False)

    # [CUSTOM] async_grpo.groups_multiplier

    def test_groups_multiplier_over_launches_each_target(self):
        collector, _ = self.make_collector(
            use_dynamic_sampling=True, num_prompts_per_step=2, groups_multiplier=2
        )

        reservations = [collector._reserve_slot()[0] for _ in range(8)]

        # Initial window [0, 1]: each target needs 2 groups and may launch 4.
        assert reservations == [0, 0, 0, 0, 1, 1, 1, 1]
        with collector._slot_cv:
            assert collector._find_open_target() is None

    def test_groups_multiplier_scales_inflight_cap(self):
        collector, _ = self.make_collector(
            use_dynamic_sampling=True,
            num_prompts_per_step=3,
            max_age=2,
            groups_multiplier=1.5,
        )

        assert collector._inflight_sema._value == 9  # ceil(3 * 2 * 1.5)

    def test_groups_multiplier_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="groups_multiplier"):
            self.make_collector(use_dynamic_sampling=True, groups_multiplier=0.5)

    def test_surplus_group_is_dropped_not_pushed(self):
        collector, buffer = self.make_collector(
            use_dynamic_sampling=True, num_prompts_per_step=1, groups_multiplier=3
        )
        for _ in range(3):
            assert collector._reserve_slot() == (0, True)

        mixed = [1.0, 0.0, 1.0, 0.0]
        self.run_group(collector, target=0, filterable=True, rewards=mixed)
        self.run_group(collector, target=0, filterable=True, rewards=mixed)

        assert buffer.push_with_wait_signal.remote.call_count == 1
        slots = collector._target_slots[0]
        assert (slots.accepted, slots.claimed, slots.surplus) == (1, 0, 1)
        assert collector.get_target_stats(0)["dynamic_sampling_num_surplus_groups"] == 1.0
        with collector._slot_cv:
            assert collector._has_open_slot(slots) is False

    def test_open_slots_shrink_with_remaining_need(self):
        collector, _ = self.make_collector(
            use_dynamic_sampling=True, num_prompts_per_step=2, groups_multiplier=2
        )
        for _ in range(4):
            collector._reserve_slot()
        slots = collector._target_slots[0]

        self.run_group(collector, target=0, filterable=True, rewards=[1.0, 0.0, 0.0, 0.0])
        # 1 group still needed -> at most 2 unclaimed in flight; 3 are.
        with collector._slot_cv:
            assert collector._has_open_slot(slots) is False

        self.run_group(collector, target=0, filterable=True, rewards=[0.0] * 4)
        self.run_group(collector, target=0, filterable=True, rewards=[0.0] * 4)
        with collector._slot_cv:
            assert collector._has_open_slot(slots) is True

    def test_group_for_consumed_target_is_dropped(self):
        collector, buffer = self.make_collector(
            use_dynamic_sampling=True, num_prompts_per_step=1, groups_multiplier=2
        )
        collector._reserve_slot()
        collector._reserve_slot()
        self.run_group(collector, target=0, filterable=True, rewards=[1.0, 0.0, 0.0, 0.0])
        collector.set_weight_version(1)

        self.run_group(collector, target=0, filterable=True, rewards=[1.0, 0.0, 0.0, 0.0])

        assert buffer.push_with_wait_signal.remote.call_count == 1

    def test_failed_push_releases_claim(self):
        collector, buffer = self.make_collector(
            use_dynamic_sampling=True, num_prompts_per_step=1, groups_multiplier=2
        )
        collector._reserve_slot()
        final_batch = BatchedDataDict({"total_reward": torch.tensor([1.0, 0.0, 0.0, 0.0])})
        collector._inflight_sema.acquire()
        with (
            mock.patch(
                "nemo_rl.algorithms.async_utils.run_async_multi_turn_rollout",
                return_value=(final_batch, {}),
            ),
            mock.patch(
                "nemo_rl.algorithms.async_utils.ray.get",
                side_effect=RuntimeError("push failed"),
            ),
        ):
            collector._run_prompt_group_worker(BatchedDataDict({}), 0, 0, 0, True)

        slots = collector._target_slots[0]
        assert (slots.accepted, slots.claimed, slots.inflight) == (0, 0, 0)
        assert collector._reserve_slot() == (0, True)


class TestFlexibleSampling:
    """[CUSTOM] Flexible replay-buffer sampling and early-consumption release.

    Uses the plain (non-actor) classes so no Ray cluster is needed.
    """

    @staticmethod
    def make_buffer(entries):
        """Build a buffer from (generation_version, target_version) pairs."""
        buffer = ReplayBuffer.__ray_actor_class__(max_size=100)
        for i, (gen, target) in enumerate(entries):
            buffer.push_with_wait_signal(
                {"batch": i, "rollout_metrics": {}}, gen, target
            )
        return buffer

    def test_strict_mode_still_waits_for_the_exact_target(self):
        # Step 5 has only 1 of its 2 groups; 3 groups of later targets are ready.
        buffer = self.make_buffer([(4, 5), (4, 6), (4, 6), (4, 7)])

        assert buffer.sample(2, 5, 10) is None

    def test_flexible_mode_fills_the_step_from_later_targets(self):
        buffer = self.make_buffer([(4, 5), (4, 6), (4, 6), (4, 7)])

        result = buffer.sample(2, 5, 10, flexible=True)

        assert [t["batch"] for t in result["trajectories"]] == [0, 1]
        assert result["target_weight_versions"] == [5, 6]
        assert buffer.size() == 2

    def test_flexible_mode_takes_oldest_generation_first(self):
        # Generation versions 3, 5, 4, 4 -> oldest first, FIFO among ties.
        buffer = self.make_buffer([(3, 7), (5, 6), (4, 8), (4, 6)])

        result = buffer.sample(3, 5, 10, flexible=True)

        assert [t["batch"] for t in result["trajectories"]] == [0, 2, 3]

    def test_flexible_mode_still_waits_when_buffer_is_short(self):
        buffer = self.make_buffer([(4, 5), (4, 6)])

        assert buffer.sample(3, 5, 10, flexible=True) is None
        assert buffer.size() == 2

    def test_flexible_mode_ignores_groups_generated_after_the_step(self):
        # A group generated at version 6 cannot serve step 5.
        buffer = self.make_buffer([(4, 5), (6, 7)])

        assert buffer.sample(2, 5, 10, flexible=True) is None

    def test_expired_group_is_dropped_in_flexible_mode_but_raises_in_strict(self):
        entries = [(0, 3), (4, 5), (4, 5)]

        with pytest.raises(ValueError, match="older than"):
            self.make_buffer(entries).sample(2, 5, 3)

        buffer = self.make_buffer(entries)
        result = buffer.sample(2, 5, 3, flexible=True)  # min_valid_version = 2
        assert [t["batch"] for t in result["trajectories"]] == [1, 2]
        assert buffer.expired_dropped == 1
        assert buffer.size() == 0
        assert result["expired_dropped_total"] == 1

    def test_release_consumed_reopens_later_targets_only(self):
        collector, _ = TestCollectorSlotScheduling().make_collector(
            use_dynamic_sampling=True, num_prompts_per_step=2, max_age=2
        )
        for _ in range(6):
            collector._reserve_slot()
        for target in (0, 1, 1, 2, 2):
            collector._target_slots[target].accepted += 1
        # Step 0 took one group of its own and groups of targets 1 and 2.
        collector.release_consumed(0, [0, 1, 2, 2])

        assert collector._target_slots[0].accepted == 1  # current target untouched
        assert collector._target_slots[1].accepted == 1
        assert collector._target_slots[2].accepted == 0
        assert collector._target_slots[2].consumed_early == 2

    def test_release_ignores_forgotten_targets(self):
        collector, _ = TestCollectorSlotScheduling().make_collector(
            use_dynamic_sampling=True, max_age=2
        )

        collector.release_consumed(0, [5, 6])  # no slots yet: nothing to do

        assert collector._target_slots == {}

    # [CUSTOM] Surplus reassignment (on with flexible_sampling)

    @staticmethod
    def pushed_targets(buffer):
        return [c.args[2] for c in buffer.push_with_wait_signal.remote.call_args_list]

    def test_surplus_group_is_reassigned_to_next_target(self):
        collector, buffer = TestCollectorSlotScheduling().make_collector(
            use_dynamic_sampling=True,
            num_prompts_per_step=1,
            groups_multiplier=3,
            max_age=2,
            flexible_sampling=True,
        )
        for _ in range(3):
            assert collector._reserve_slot()[0] == 0
        mixed = [1.0, 0.0, 1.0, 0.0]
        harness = TestCollectorSlotScheduling()

        harness.run_group(collector, target=0, filterable=True, rewards=mixed)
        harness.run_group(collector, target=0, filterable=True, rewards=mixed)

        assert self.pushed_targets(buffer) == [0, 1]
        assert collector._target_slots[0].accepted == 1
        assert collector._target_slots[0].reassigned == 1
        assert collector._target_slots[0].surplus == 0
        assert collector._target_slots[1].accepted == 1
        assert collector._target_slots[1].claimed == 0
        assert collector.get_target_stats(0)["dynamic_sampling_num_reassigned_groups"] == 1.0

    def test_surplus_dropped_when_no_reachable_target_needs_groups(self):
        collector, buffer = TestCollectorSlotScheduling().make_collector(
            use_dynamic_sampling=True,
            num_prompts_per_step=1,
            groups_multiplier=3,
            max_age=1,
            flexible_sampling=True,
        )
        for _ in range(3):
            collector._reserve_slot()
        mixed = [1.0, 0.0, 1.0, 0.0]
        harness = TestCollectorSlotScheduling()
        # Generated at version 0 with age 1: only targets 0 and 1 are reachable.
        for _ in range(3):
            harness.run_group(collector, target=0, filterable=True, rewards=mixed)

        assert self.pushed_targets(buffer) == [0, 1]
        assert collector._target_slots[0].surplus == 1

    def test_without_flexible_sampling_surplus_is_still_dropped(self):
        collector, buffer = TestCollectorSlotScheduling().make_collector(
            use_dynamic_sampling=True, num_prompts_per_step=1, groups_multiplier=3
        )
        for _ in range(3):
            collector._reserve_slot()
        harness = TestCollectorSlotScheduling()
        for _ in range(2):
            harness.run_group(collector, target=0, filterable=True, rewards=[1.0, 0.0, 0.0, 0.0])

        assert self.pushed_targets(buffer) == [0]
        assert collector._target_slots[0].surplus == 1

    def test_reassigned_slot_counts_against_the_later_target(self):
        collector, _ = TestCollectorSlotScheduling().make_collector(
            use_dynamic_sampling=True,
            num_prompts_per_step=1,
            groups_multiplier=1,
            max_age=2,
            flexible_sampling=True,
        )
        collector._target_slots[0] = _TargetSlots(accepted=1)

        assert collector._claim_slot(0, 0) == 1
        # Target 1 now has its one group claimed: no more launches for it.
        with collector._slot_cv:
            assert collector._has_open_slot(collector._target_slots[1]) is False

    def test_early_consumption_before_settle_is_not_double_counted(self):
        collector, _ = TestCollectorSlotScheduling().make_collector(
            use_dynamic_sampling=True, num_prompts_per_step=2, max_age=2,
            flexible_sampling=True,
        )
        collector._reserve_slot()  # target 0
        slots = collector._target_slots.setdefault(1, _TargetSlots())
        slots.inflight, slots.claimed = 1, 1
        # Step 0 took the target-1 group before the collector settled its push.
        collector.release_consumed(0, [1])
        collector._settle_slot(1, accepted=True, discarded=False, claimed=True)

        assert (slots.accepted, slots.claimed, slots.release_pending) == (0, 0, 0)
        assert slots.consumed_early == 1


class TestPromptBookkeeping:
    """[CUSTOM] No prompt taken from the dataloader is lost: each one ends trained
    on or discarded by dynamic sampling; anything else is generated again."""

    @staticmethod
    def setup(**kw):
        harness = TestCollectorSlotScheduling()
        collector, buffer = harness.make_collector(use_dynamic_sampling=True, **kw)
        return harness, collector, buffer

    @staticmethod
    def register(collector, name):
        with collector._pending_lock:
            return collector._register_prompt(BatchedDataDict({"name": [name]}))

    def test_dynamic_sampling_discard_finishes_the_prompt(self):
        harness, collector, _ = self.setup()
        uid = self.register(collector, "a")
        collector._reserve_slot()

        harness.run_group(collector, target=0, filterable=True, rewards=[0.0] * 4, prompt_uid=uid)

        assert collector._pending_prompts == {}
        assert not collector._retry_uids

    def test_buffered_group_stays_pending_until_trained(self):
        harness, collector, buffer = self.setup()
        uid = self.register(collector, "a")
        collector._reserve_slot()

        harness.run_group(collector, target=0, filterable=True, rewards=[1.0, 0.0, 0.0, 0.0], prompt_uid=uid)

        pushed = buffer.push_with_wait_signal.remote.call_args.args[0]
        assert pushed["prompt_uid"] == uid
        assert uid in collector._pending_prompts
        stats = collector.update_prompt_status([uid], [])
        assert collector._pending_prompts == {}
        assert stats["async_num_pending_prompts"] == 0.0

    def test_surplus_group_is_requeued_not_lost(self):
        harness, collector, buffer = self.setup(num_prompts_per_step=1, groups_multiplier=2)
        uids = [self.register(collector, n) for n in "ab"]
        collector._reserve_slot()
        collector._reserve_slot()
        mixed = [1.0, 0.0, 1.0, 0.0]

        harness.run_group(collector, target=0, filterable=True, rewards=mixed, prompt_uid=uids[0])
        harness.run_group(collector, target=0, filterable=True, rewards=mixed, prompt_uid=uids[1])

        assert buffer.push_with_wait_signal.remote.call_count == 1
        assert list(collector._retry_uids) == [uids[1]]
        assert collector._take_retry_prompts()[0][1]["name"] == ["b"]

    def test_failed_worker_requeues_its_prompt(self):
        _, collector, _ = self.setup()
        uid = self.register(collector, "a")
        collector._reserve_slot()
        collector._inflight_sema.acquire()
        with mock.patch(
            "nemo_rl.algorithms.async_utils.run_async_multi_turn_rollout",
            side_effect=RuntimeError("rollout failed"),
        ):
            collector._run_prompt_group_worker(BatchedDataDict({}), 0, 0, 0, True, uid)

        assert list(collector._retry_uids) == [uid]

    def test_expired_groups_are_requeued(self):
        _, collector, _ = self.setup()
        uid = self.register(collector, "a")

        collector.update_prompt_status([], [uid])
        collector.update_prompt_status([], [uid])  # not queued twice

        assert list(collector._retry_uids) == [uid]
        assert collector._num_requeued_prompts == 1

    def test_checkpoint_saves_pending_prompts_and_resume_queues_them_first(self):
        _, collector, _ = self.setup()
        uids = [self.register(collector, n) for n in "abc"]
        collector.update_prompt_status([uids[1]], [])  # "b" was trained on

        state = collector.get_dataloader_state()
        assert [p["name"] for p in state["pending_prompts"]] == [["a"], ["c"]]

        _, resumed, _ = self.setup()
        resumed.restore_pending_prompts(state["pending_prompts"])
        assert [p["name"] for _, p in resumed._take_retry_prompts()] == [["a"], ["c"]]
        assert len(resumed._pending_prompts) == 2

    def test_buffer_hands_back_expired_prompt_uids_once(self):
        buffer = ReplayBuffer.__ray_actor_class__(max_size=100)
        buffer.push_with_wait_signal({"batch": 0, "rollout_metrics": {}, "prompt_uid": 7}, 0, 3)
        for i in (1, 2):
            buffer.push_with_wait_signal({"batch": i, "rollout_metrics": {}, "prompt_uid": 10 + i}, 4, 5)

        first = buffer.sample(1, 5, 3, flexible=True)
        second = buffer.sample(1, 5, 3, flexible=True)

        assert first["expired_prompt_uids"] == [7]
        assert second["expired_prompt_uids"] == []

    def test_collection_loop_registers_batches_and_runs_retries_first(self):
        _, collector, _ = self.setup()
        collector.dataloader = [
            BatchedDataDict({"name": ["p0", "p1"]}),
            BatchedDataDict({"name": ["p2", "p3"]}),
        ]
        collector.restore_pending_prompts([BatchedDataDict({"name": ["old"]})])
        launched = []

        def fake_launch(items):
            launched.append([p["name"][0] for _, p in items])
            if len(launched) == 3:
                collector.running = False

        collector._launch_prompts = fake_launch
        collector._collection_loop()

        assert launched == [["old"], ["p0", "p1"], ["p2", "p3"]]
        # Nothing finished yet, so all five prompts are still pending.
        assert sorted(p["name"][0] for p in collector._pending_prompts.values()) == [
            "old", "p0", "p1", "p2", "p3"
        ]


class TestSkipSolvedPrompts:
    """[CUSTOM] Prompts whose whole group is solved (all-1) are skipped in later
    epochs, and max_num_epochs then counts passes over the data."""

    @staticmethod
    def setup(max_num_epochs=2, **kw):
        harness = TestCollectorSlotScheduling()
        collector, buffer = harness.make_collector(use_dynamic_sampling=True, **kw)
        collector._skip_solved = True
        collector._solved_threshold = 1.0
        collector._max_data_epochs = max_num_epochs
        return harness, collector, buffer

    @staticmethod
    def group(idx):
        return BatchedDataDict({"idx": [idx] * 4})

    def run_group(self, harness, collector, idx, rewards, filterable=True):
        """run_group of TestCollectorSlotScheduling, with the prompt's idx."""
        collector._reserve_slot()
        collector._inflight_sema.acquire()
        final_batch = BatchedDataDict({"total_reward": torch.tensor(rewards)})
        with (
            mock.patch(
                "nemo_rl.algorithms.async_utils.run_async_multi_turn_rollout",
                return_value=(final_batch, {}),
            ),
            mock.patch("nemo_rl.algorithms.async_utils.ray.get", side_effect=lambda x: x),
        ):
            collector._run_prompt_group_worker(self.group(idx), 0, 0, 0, filterable, None)

    def test_only_all_one_groups_are_marked_solved(self):
        harness, collector, _ = self.setup(num_prompts_per_step=4)
        collector.current_epoch = 1
        self.run_group(harness, collector, 10, [1.0] * 4)
        self.run_group(harness, collector, 11, [0.0] * 4)
        self.run_group(harness, collector, 12, [1.0, 1.0, 1.0, 0.0])
        # Accepted unfiltered (dynamic sampling launch cap reached): still solved.
        self.run_group(harness, collector, 13, [1.0] * 4, filterable=False)

        assert collector._solved_prompts == {10: 1, 13: 1}

    def test_solved_prompts_are_skipped_in_later_epochs_and_reading_stops(self):
        _, collector, _ = self.setup(max_num_epochs=2)
        collector.dataloader = [
            BatchedDataDict({"idx": [0, 1]}),
            BatchedDataDict({"idx": [2, 3]}),
        ]
        launched = []

        def fake_launch(items):
            idxs = [p["idx"][0] for _, p in items]
            launched.append(idxs)
            if 1 in idxs:  # prompt 1 is solved in the first epoch
                collector._solved_prompts[1] = collector.current_epoch

        def stop_when_idle(_):
            collector.running = False

        collector._launch_prompts = fake_launch
        with mock.patch("nemo_rl.algorithms.async_utils.time.sleep", stop_when_idle):
            collector._collection_loop()

        assert launched == [[0, 1], [2, 3], [0], [2, 3]]
        assert collector._epochs_completed == 2
        assert collector._num_skipped_solved == 1
        assert collector._data_exhausted
        stats = collector.update_prompt_status([], [])
        assert stats["async_num_solved_prompts"] == 1.0
        assert stats["async_num_skipped_solved_total"] == 1.0

    def test_requeued_prompts_are_still_generated_after_the_last_epoch(self):
        _, collector, _ = self.setup(max_num_epochs=1)
        collector.dataloader = [BatchedDataDict({"idx": [0]})]
        launched = []
        uids = []

        def fake_launch(items):
            launched.append(([p["idx"][0] for _, p in items], collector._data_exhausted))
            uids.append(items[0][0])

        def stop_when_idle(_):
            if len(launched) == 1:
                # Idle after the last epoch, then the group expires in the buffer.
                collector._requeue_prompt(uids[0])
            else:
                collector.running = False

        collector._launch_prompts = fake_launch
        with mock.patch("nemo_rl.algorithms.async_utils.time.sleep", stop_when_idle):
            collector._collection_loop()

        assert launched == [([0], False), ([0], True)]

    def test_is_finished_waits_for_retries_and_running_workers(self):
        _, collector, _ = self.setup()
        assert not collector.is_finished()  # data not read yet

        collector._data_exhausted = True
        assert collector.is_finished()

        with collector._pending_lock:
            uid = collector._register_prompt(BatchedDataDict({"idx": [5]}))
        collector._requeue_prompt(uid)
        assert not collector.is_finished()
        collector._take_retry_prompts()

        collector._relaunching = True
        assert not collector.is_finished()
        collector._relaunching = False

        alive = mock.MagicMock()
        alive.is_alive.return_value = True
        collector._inflight_threads.add(alive)
        assert not collector.is_finished()
        alive.is_alive.return_value = False
        assert collector.is_finished()

    def test_checkpoint_round_trip_and_old_checkpoints(self):
        _, collector, _ = self.setup()
        collector._epochs_completed = 3
        collector._solved_prompts = {4: 1, 9: 3}
        state = collector.get_dataloader_state()
        state.pop("pending_prompts")
        # train_dataloader.pt is read back with weights_only=True.
        buf = io.BytesIO()
        torch.save(state, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=True)

        _, resumed, _ = self.setup()
        resumed.restore_collection_state(loaded)
        assert resumed._epochs_completed == 3
        assert resumed._solved_prompts == {4: 1, 9: 3}

        _, old, _ = self.setup()
        old.restore_collection_state({"current_epoch": 2, "dataloader_state": {}})
        assert old._epochs_completed == 1
        assert old._solved_prompts == {}

    def test_skip_config_reads_max_num_epochs(self):
        harness = TestCollectorSlotScheduling()
        collector, _ = harness.make_collector(use_dynamic_sampling=True)
        assert collector._max_data_epochs is None
        cfg = collector.master_config
        cfg["grpo"]["max_num_epochs"] = 5
        cfg["grpo"]["async_grpo"]["skip_solved_prompts"] = True
        enabled = AsyncTrajectoryCollector.__ray_actor_class__(
            policy_generation=MockGenerationInterface(),
            tokenizer=mock.MagicMock(),
            task_to_env={},
            master_config=cfg,
            replay_buffer=mock.MagicMock(),
            start_step=0,
        )
        assert enabled._skip_solved and enabled._max_data_epochs == 5
        assert enabled._solved_threshold == 1.0


class TestAsyncRewardShaping:
    """[CUSTOM] The async collector applies reward shaping before dynamic
    sampling, as the sync loop does; the solved check uses the raw rewards."""

    def run_group(self, collector, response_lengths, rewards):
        final_batch = BatchedDataDict(
            {
                "total_reward": torch.tensor(rewards),
                "message_log": [
                    [
                        {"role": "user", "token_ids": torch.zeros(5)},
                        {"role": "assistant", "token_ids": torch.zeros(n)},
                    ]
                    for n in response_lengths
                ],
            }
        )
        collector._reserve_slot()
        collector._inflight_sema.acquire()
        with (
            mock.patch(
                "nemo_rl.algorithms.async_utils.run_async_multi_turn_rollout",
                return_value=(final_batch, {}),
            ),
            mock.patch("nemo_rl.algorithms.async_utils.ray.get", side_effect=lambda x: x),
        ):
            collector._run_prompt_group_worker(
                BatchedDataDict({"idx": [7] * 4}), 0, 0, 0, True, None
            )

    def make(self):
        harness = TestCollectorSlotScheduling()
        collector, buffer = harness.make_collector(use_dynamic_sampling=True)
        collector.master_config["grpo"]["reward_shaping"] = {
            "enabled": True,
            "overlong_buffer_length": 10,
            "overlong_buffer_penalty": 0.5,
            "max_response_length": 100,
        }
        collector._skip_solved = True
        collector._solved_threshold = 1.0
        return collector, buffer

    def test_overlong_penalty_breaks_ties_and_solved_uses_raw_rewards(self):
        collector, buffer = self.make()
        # All correct; one answer ends 5 tokens into the 10-token buffer.
        self.run_group(collector, [50, 60, 70, 95], [1.0] * 4)

        pushed = buffer.push_with_wait_signal.remote.call_args.args[0]
        assert pushed["batch"]["total_reward"].tolist() == [1.0, 1.0, 1.0, 0.75]
        assert collector._solved_prompts == {7: collector.current_epoch}

    def test_short_all_correct_group_is_still_discarded(self):
        collector, buffer = self.make()
        self.run_group(collector, [50, 60, 70, 80], [1.0] * 4)

        assert buffer.push_with_wait_signal.remote.call_count == 0
        assert 7 in collector._solved_prompts
