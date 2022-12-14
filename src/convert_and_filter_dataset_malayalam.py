import pathlib
from functools import partial
from io import BytesIO
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import scipy.io.wavfile as wavf
from datasets import load_dataset, Audio, DatasetDict, concatenate_datasets
from pydub import AudioSegment, effects
from shortuuid import uuid
from tqdm import tqdm


def load_data_splits():
    data_dict = {}
    data_dict["openslr_dataset_train"] = load_dataset(
        "openslr", "SLR63", split="train", use_auth_token=True
    )
    data_dict["ucla_dataset_train"] = load_dataset(
        "audiofolder", data_dir="../data/malayalam/", drop_labels=True
    )["train"]
    data_dict["common_voice_train"] = load_dataset(
        "mozilla-foundation/common_voice_11_0", "ml", split="train", use_auth_token=True
    )
    data_dict["common_voice_test"] = load_dataset(
        "mozilla-foundation/common_voice_11_0", "ml", split="test", use_auth_token=True
    )
    data_dict["fleurs_dataset_train"] = load_dataset(
        "google/fleurs", "ml_in", split="train", use_auth_token=True
    ).rename_column("transcription", "sentence")
    data_dict["fleurs_dataset_val"] = load_dataset(
        "google/fleurs", "ml_in", split="validation", use_auth_token=True
    ).rename_column("transcription", "sentence")
    data_dict["fleurs_dataset_test"] = load_dataset(
        "google/fleurs", "ml_in", split="test", use_auth_token=True
    ).rename_column("transcription", "sentence")

    for k in data_dict:
        data_dict[k] = data_dict[k].remove_columns(
            [
                col
                for col in data_dict[k].column_names
                if col not in ["audio", "sentence"]
            ]
        )
        data_dict[k] = data_dict[k].cast_column("audio", Audio(sampling_rate=16000))

    dataset_dict = DatasetDict()
    train_datasets = []
    test_datasets = []
    for k in data_dict:
        if k.endswith("train") or k.endswith("val"):
            train_datasets.append(data_dict[k])
        if k.endswith("test"):
            test_datasets.append(data_dict[k])
    dataset_dict["train"] = concatenate_datasets(train_datasets)
    dataset_dict["test"] = concatenate_datasets(test_datasets)
    return dataset_dict


def audio_from_array(array):
    file = BytesIO()
    wavf.write(file, 16000, array)
    audio_segment = AudioSegment.from_file(file)
    audio_segment.set_frame_rate(16000).set_channels(1)
    audio_segment = effects.normalize(audio_segment)
    return audio_segment


def filter_nans_and_short(example):
    sentence = example["sentence"]
    length = example["length"]
    if sentence is None:
        return False
    elif length < 3 or length > 30:
        return False
    else:
        return True


def export_audio_and_sentence(example, split):
    audio, sentence = example["audio"], example["sentence"]
    new_name = pathlib.Path(uuid(audio["path"])).with_suffix(".mp3").name
    new_path = pathlib.Path(f"../data/filtered_datasets/malayalam/{split}/{new_name}")
    new_path.parent.mkdir(parents=True, exist_ok=True)
    length = len(audio["array"]) / 16000
    if new_path.is_file():
        return {
            "path": str(new_path),
            "sentence": sentence,
            "length": length,
            "split": split,
        }
    else:
        if not split == "test":
            is_not_filtered = filter_nans_and_short(
                {"length": length, "sentence": sentence}
            )
        else:
            is_not_filtered = True
        if is_not_filtered:
            try:
                new_path.parent.mkdir(exist_ok=True, parents=True)
                audio_segment = audio_from_array(audio["array"])
                audio_segment.export(new_path, format="mp3", bitrate="16k")
                segment = AudioSegment.from_mp3(new_path)
            except:
                return None
            return {
                "path": str(new_path),
                "sentence": sentence,
                "length": length,
                "split": split,
            }
        else:
            return None


if __name__ == "__main__":

    dataset_dict = load_data_splits()
    exports = []
    with Pool(cpu_count() - 1) as pool:
        for k in dataset_dict:
            dataset = dataset_dict[k].shuffle(seed=np.random.randint(1000))
            exporter = partial(export_audio_and_sentence, split=k)
            results = pool.imap_unordered(exporter, dataset, chunksize=10)
            for result in tqdm(results, total=len(dataset)):
                if result:
                    exports.append(result)
    exports = pd.DataFrame(exports)
    exports.to_json(
        "../data/filtered_datasets/malayalam/metadata.jsonl",
        lines=True,
        orient="records",
    )
