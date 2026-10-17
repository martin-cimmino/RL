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

import json
from types import SimpleNamespace

import pytest

from nemo_rl.data.datasets.response_datasets.nemogym_dataset import NemoGymDataset
from nemo_rl.data.utils import build_source_ranges, resolve_source


def _write_jsonl(path, num_rows):
    with open(path, "w") as f:
        for i in range(num_rows):
            f.write(json.dumps({"row": i}) + "\n")
    return str(path)


@pytest.fixture
def two_files(tmp_path):
    """a.jsonl (3 rows) and b.jsonl (2 rows, repeated twice), concatenated."""
    path_a = _write_jsonl(tmp_path / "a.jsonl", 3)
    path_b = _write_jsonl(tmp_path / "b.jsonl", 2)
    data_list = [NemoGymDataset(path_a), NemoGymDataset(path_b, repeat=2)]
    cfgs = [{"data_path": path_a}, {"data_path": path_b}]
    return path_a, path_b, build_source_ranges(data_list, cfgs)


def test_build_source_ranges_offsets(two_files):
    path_a, path_b, ranges = two_files

    assert [
        (r["data_path"], r["start"], r["num_rows"], r["num_source_rows"])
        for r in ranges
    ] == [
        (path_a, 0, 3, 3),
        (path_b, 3, 4, 2),
    ]


@pytest.mark.parametrize(
    "idx, expected",
    [
        (0, ("a", 0)),
        (2, ("a", 2)),
        (3, ("b", 0)),
        (4, ("b", 1)),
        # Second repeat of b.jsonl maps back to the same lines.
        (5, ("b", 0)),
        (6, ("b", 1)),
    ],
)
def test_resolve_source(two_files, idx, expected):
    path_a, path_b, ranges = two_files
    paths = {"a": path_a, "b": path_b}

    assert resolve_source(idx, ranges) == (paths[expected[0]], expected[1])


def test_resolve_source_out_of_range(two_files):
    _, _, ranges = two_files

    assert resolve_source(7, ranges) == (None, None)


def test_row_is_unknown_without_num_source_rows():
    # Datasets that don't report num_source_rows (e.g. ones shuffled by a
    # train/validation split) only resolve to their file.
    data = SimpleNamespace(dataset=[0, 1, 2], task_name="hf")
    ranges = build_source_ranges([data], [{"data_path": None}])

    assert ranges[0]["num_source_rows"] is None
    assert resolve_source(1, ranges) == (None, None)
