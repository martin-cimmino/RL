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

import os
import tempfile
import threading
import unittest.mock as mock

import pytest
import ray
import torch

# Set up Ray temp directory before any Ray operations
# Try multiple approaches to ensure Ray uses a writable directory
_temp_dir = tempfile.mkdtemp(prefix="ray_async_test_")
os.environ["RAY_TEMP_DIR"] = _temp_dir
os.environ["RAY_TMPDIR"] = _temp_dir  # Alternative env var
os.environ["TMPDIR"] = _temp_dir  # System temp dir

from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector, ReplayBuffer
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
    ):
        grpo_cfg = {
            "num_prompts_per_step": num_prompts_per_step,
            "num_generations_per_prompt": 4,
            "max_rollout_turns": 1,
            "use_dynamic_sampling": use_dynamic_sampling,
            "async_grpo": {"max_trajectory_age_steps": max_age},
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

    def run_group(self, collector, *, target, filterable, rewards):
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
                BatchedDataDict({}), 0, target, 0, filterable
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
