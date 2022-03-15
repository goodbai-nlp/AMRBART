# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""AMR dataset."""


from inspect import EndOfBlock
import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """

There are three features:
  - src: text.
  - tgt: Linearized AMR.
"""


_SRC = "src"
_TGT = "tgt"


class AMRData(datasets.GeneratorBasedBuilder):
    """AMR Dataset."""

    # Version 1.0.0 expands coverage, includes ids, and removes web contents.
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {_SRC: datasets.Value("string"), _TGT: datasets.Value("string"),}
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        train_path = self.config.data_files["train"]
        dev_path = self.config.data_files["validation"]
        test_path = self.config.data_files["test"]
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_path}
            ),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                json_dict = json.loads(line.strip())
                src = json_dict["src"]
                tgt = json_dict["tgt"]
                yield idx, {_SRC: src, _TGT: tgt}
