# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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

"""Filtered Tamil ASR corpus collected from common_voice 11, fleurs, openslr65, openslr127 and ucla corpora filtered for duration between 5 - 25 secs"""


import json
import os

import datasets

_CITATION = """\
@misc{mile_1,
  doi = {10.48550/ARXIV.2207.13331},
  url = {https://arxiv.org/abs/2207.13331},
  author = {A, Madhavaraj and Pilar, Bharathi and G, Ramakrishnan A},
  title = {Subword Dictionary Learning and Segmentation Techniques for Automatic Speech Recognition in Tamil and Kannada},
  publisher = {arXiv},
  year = {2022},
}

@misc{mile_2,
  doi = {10.48550/ARXIV.2207.13333},
  url = {https://arxiv.org/abs/2207.13333},
  author = {A, Madhavaraj and Pilar, Bharathi and G, Ramakrishnan A},
  title = {Knowledge-driven Subword Grammar Modeling for Automatic Speech Recognition in Tamil and Kannada},
  publisher = {arXiv},
  year = {2022},
}

@inproceedings{he-etal-2020-open,
    title = {{Open-source Multi-speaker Speech Corpora for Building Gujarati, Kannada, Malayalam, Marathi, Tamil and Telugu Speech Synthesis Systems}},
    author = {He, Fei and Chu, Shan-Hui Cathy and Kjartansson, Oddur and Rivera, Clara and Katanova, Anna and Gutkin, Alexander and Demirsahin, Isin and Johny, Cibu and Jansche, Martin and Sarin, Supheakmungkol and Pipatsrisawat, Knot},
    booktitle = {Proceedings of The 12th Language Resources and Evaluation Conference (LREC)},
    month = may,
    year = {2020},
    address = {Marseille, France},
    publisher = {European Language Resources Association (ELRA)},
    pages = {6494--6503},
    url = {https://www.aclweb.org/anthology/2020.lrec-1.800},
    ISBN = "{979-10-95546-34-4},
  }

@misc{https://doi.org/10.48550/arxiv.2211.09536,
  doi = {10.48550/ARXIV.2211.09536},
  
  url = {https://arxiv.org/abs/2211.09536},
  
  author = {Kumar, Gokul Karthik and S, Praveen and Kumar, Pratyush and Khapra, Mitesh M. and Nandakumar, Karthik},
  
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), Sound (cs.SD), Audio and Speech Processing (eess.AS), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  
  title = {Towards Building Text-To-Speech Systems for the Next Billion Users},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

@inproceedings{commonvoice:2020,
  author = {Ardila, R. and Branson, M. and Davis, K. and Henretty, M. and Kohler, M. and Meyer, J. and Morais, R. and Saunders, L. and Tyers, F. M. and Weber, G.},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)},
  pages = {4211--4215},
  year = 2020
}

@misc{https://doi.org/10.48550/arxiv.2205.12446,
  doi = {10.48550/ARXIV.2205.12446},
  
  url = {https://arxiv.org/abs/2205.12446},
  
  author = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), Sound (cs.SD), Audio and Speech Processing (eess.AS), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  
  title = {FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

"""

_DESCRIPTION = """\
The corpus contains roughly 1000 hours of audio and trasncripts in Tamil language. The transcripts have beedn de-duplicated using exact match deduplication.
"""

_HOMEPAGE = ""

_LICENSE = "https://creativecommons.org/licenses/"


_METADATA_URLS = {
    "train": "data/train.jsonl",
    "test": "data/test.jsonl"
}
_URLS = {
    "train": "data/train.tar.gz",
    "test": "data/test.tar.gz",
    
}

class SampleDataset(datasets.GeneratorBasedBuilder):
    """Tamil ASR Corpus contains transcribed speech corpus for training ASR systems for Tamil language."""

    VERSION = datasets.Version("1.1.0")
    def _info(self):
        features = datasets.Features(
            {
                "audio": datasets.Audio(sampling_rate=16_000),
                "path": datasets.Value("string"),
                "sentence": datasets.Value("string"),
                "length": datasets.Value("float")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=("sentence", "label"),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_paths = dl_manager.download(_METADATA_URLS)
        train_archive = dl_manager.download(_URLS["train"])
        test_archive = dl_manager.download(_URLS["test"])
        local_extracted_train_archive = dl_manager.extract(train_archive) if not dl_manager.is_streaming else None
        local_extracted_test_archive = dl_manager.extract(test_archive) if not dl_manager.is_streaming else None
        test_archive = dl_manager.download(_URLS["test"])
        train_dir = "train"
        test_dir = "test"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_path": metadata_paths["train"],
                    "local_extracted_archive": local_extracted_train_archive,
                    "path_to_clips": train_dir,
                    "audio_files": dl_manager.iter_archive(train_archive),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "metadata_path": metadata_paths["test"],
                    "local_extracted_archive": local_extracted_test_archive,
                    "path_to_clips": test_dir,
                    "audio_files": dl_manager.iter_archive(test_archive),
                },
            ),
            
        ]
        
    def _generate_examples(self, metadata_path, local_extracted_archive, path_to_clips, audio_files):
        """Yields examples as (key, example) tuples."""
        examples = {}
        with open(metadata_path, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                examples[data["path"]] = data
        inside_clips_dir = False
        id_ = 0
        for path, f in audio_files:
            if path.startswith(path_to_clips):
                inside_clips_dir = True
                if path in examples:
                    result = examples[path]
                    path = os.path.join(local_extracted_archive, path) if local_extracted_archive else path
                    result["audio"] = {"path": path, "bytes": f.read()}
                    result["path"] = path
                    yield id_, result
                    id_ += 1
            elif inside_clips_dir:
                break

