from datasets import load_dataset, Audio, IterableDatasetDict, DatasetDict, concatenate_datasets
from pydub import AudioSegment, effects
import numpy as np
import scipy.io.wavfile as wavf
from io import BytesIO
from shortuuid import uuid
import pathlib
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial



def load_data_splits(is_streaming=True, stopping_strategy="all_exhausted"):
    dataset_dict = {}
    
    print("Loading commonvoice")
    # load commonvoice 
    dataset_dict["common_voice_train"] = load_dataset(
        "mozilla-foundation/common_voice_11_0", "ta",
        split="train", 
        use_auth_token=True,
        streaming=is_streaming)
    
    dataset_dict["common_voice_val"] = load_dataset(
        "mozilla-foundation/common_voice_11_0", "ta",
        split="validation", 
        use_auth_token=True,
        streaming=is_streaming)
    
    
    dataset_dict["common_voice_test"] = load_dataset(
        "mozilla-foundation/common_voice_11_0", "ta", 
        split="test", 
        use_auth_token=True,
        streaming=is_streaming)
    
    print("Loading openslr65")
    # load openslr65
    dataset_dict["openslr_train"]  = load_dataset(
        "openslr", "SLR65",
        split="train",
        use_auth_token=True,
        streaming=is_streaming)
    
    print("Loading fleurs")
    # load google fleurs
    dataset_dict["fleurs_train"]  = load_dataset(
        "google/fleurs", "ta_in", 
        split="train",
        use_auth_token=True,
        streaming=is_streaming).rename_columns({"transcription": "sentence"})    
    
    dataset_dict["fleurs_val"]  = load_dataset(
        "google/fleurs", "ta_in", 
        split="validation",
        use_auth_token=True,
        streaming=is_streaming).rename_columns({"transcription": "sentence"})    
    

    dataset_dict["fleurs_test"] = load_dataset(
        "google/fleurs", "ta_in",
        split="test",
        use_auth_token=True,
        streaming=is_streaming).rename_columns({"transcription": "sentence"})     

    print("Loading ucla")
    # load ucla
    dataset_dict["ucla_train"] = load_dataset(
        "parambharat/ucla_dataset",
        split="train",
        use_auth_token=True,
        streaming=is_streaming)
    
    print("Loading mile")
    # load mile
    dataset_dict["mile_train"] = load_dataset(
        "parambharat/mile_dataset",
        split="train",
        use_auth_token=True,
        streaming=is_streaming)
    
    dataset_dict["mile_val"] = load_dataset(
        "parambharat/mile_dataset",
        split="test",
        use_auth_token=True,
        streaming=is_streaming)


    print("Interleaving all datasets")
    for k,v in dataset_dict.items():
        dataset_dict[k] = v.cast_column("audio", Audio(sampling_rate=16_000))
        dataset_dict[k] = dataset_dict[k].remove_columns(
            set(dataset_dict[k].features.keys()) - set(["audio", "sentence"])
        )
    
    if is_streaming:
        data_dict = IterableDatasetDict()
    else:
        data_dict = DatasetDict()
    data_dict["train"] = concatenate_datasets(
        [
            dataset_dict["common_voice_train"],
            dataset_dict["common_voice_val"],
            dataset_dict["openslr_train"],
            dataset_dict["fleurs_train"],
            dataset_dict["fleurs_val"],
            dataset_dict["ucla_train"],
            dataset_dict["mile_train"],
            dataset_dict["mile_val"],
        ],
#          stopping_strategy=stopping_strategy
    )
    data_dict["test"] = concatenate_datasets(
        [
            dataset_dict["common_voice_test"],
            dataset_dict["fleurs_test"]
        ],
#      stopping_strategy=stopping_strategy
    )
    return data_dict



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
    new_path = pathlib.Path(f"datasets/filtered_dataset/{split}/{new_name}")
    length = len(audio["array"])/16000
    if new_path.is_file():
        return {"path": str(new_path),"sentence": sentence, "length": length, "split": split}
    else:
        if not split ==  "test":
            is_not_filtered = filter_nans_and_short({"length": length, "sentence": sentence})
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
            return {"path": str(new_path),"sentence": sentence, "length": length, "split": split}
        else:
            return None

if __name__ == "__main__":

    dataset_dict = load_data_splits(is_streaming=False)
    exports = []
    with Pool(cpu_count()-1) as pool:
        for k in dataset_dict:
            dataset = dataset_dict[k].shuffle(seed=np.random.randint(1000))
            exporter = partial(export_audio_and_sentence, split=k)
            results = pool.imap_unordered(exporter,dataset,chunksize=10)
            for result in tqdm(results, total=len(dataset)):
                if result:
                    exports.append(result)