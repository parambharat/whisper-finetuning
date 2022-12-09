import os
import pathlib
from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count, Pool

from pydub import AudioSegment, effects
from tqdm import tqdm



def get_mp3_path(wavfile):
    mp3dir = list(wavfile.parents)[1] / "mp3"
    mp3dir.mkdir(parents=True, exist_ok=True)
    mp3file = mp3dir / wavfile.with_suffix(".mp3").name
    return mp3file


def convert_wav_to_mp3(wavfile):
    mp3file = get_mp3_path(wavfile)
    try:
        wavaudio = AudioSegment.from_wav(wavfile)
        wavaudio = wavaudio.set_frame_rate(16000).set_channels(1)
        wavaudio = effects.normalize(wavaudio)
        wavaudio.export(mp3file, format="mp3", bitrate="16k")
        wavfile.unlink()
        return mp3file
    except Exception as e:
        print(f"Unable to convert {wavfile} to mp3")
        wavfile.unlink()
        
def main():
    base_dir = pathlib.Path("mile_tamil_asr_corpus")
    train_dir = base_dir/"train/audio_files"
    test_dir = base_dir/"test/audio_files"
    train_wav_files = list(train_dir.rglob("*.wav"))
    test_wav_files = list(test_dir.rglob("*.wav"))
    mp3files = []
    with Pool(cpu_count() - 1) as pool:
        results = pool.imap_unordered(convert_wav_to_mp3, train_wav_files)
        print("Converting wav to mp3")
        for mp3file in tqdm(results, total=len(train_wav_files)):
            mp3files.append(mp3file)
    with Pool(cpu_count() - 1) as pool:
        results = pool.imap_unordered(convert_wav_to_mp3, test_wav_files)
        print("Converting wav to mp3")
        for mp3file in tqdm(results, total=len(test_wav_files)):
            mp3files.append(mp3file)

    return mp3files

if __name__ == "__main__":
    main()
    