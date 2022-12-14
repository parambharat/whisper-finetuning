import os
import pathlib
from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count, Pool
from zipfile import ZipFile

from keras.utils import get_file
from pydub import AudioSegment, effects
from tqdm import tqdm


def download_archive(url):
    filename = os.path.split(url)[-1]
    try:
        filename = get_file(
            fname=filename, origin=url, cache_dir="../data", extract=False
        )
    except:
        print(f"Unable to download {url}")
        return None
    return filename


def extract_zipfile(filename):
    filename = pathlib.Path(filename)
    dirname = filename.parent / filename.stem
    if not dirname.is_dir():
        try:
            with ZipFile(filename, "r") as zipf:
                for zipinfo in zipf.infolist():
                    if zipinfo.filename[-1] == "/":
                        continue
                    zipinfo.filename = pathlib.Path(zipinfo.filename).name
                    if pathlib.Path(zipinfo.filename).suffix == ".wav":
                        zipf.extract(zipinfo, dirname / "wav")
                    else:
                        zipf.extract(zipinfo, dirname)
            wavfiles = list((dirname / "wav").rglob("*.wav"))
            return wavfiles
        except:
            return None


def get_mp3_path(wavfile):
    mp3dir = list(wavfile.parents)[1] / "mp3"
    mp3dir.mkdir(parents=True, exist_ok=True)
    mp3file = mp3dir / wavfile.with_suffix(".mp3").name
    return mp3file


def convert_wav_to_mp3(wavfile):
    mp3file = get_mp3_path(wavfile)
    if mp3file.is_file():
        if wavfile.is_file():
            wavfile.unlink()
            return mp3file
        return mp3file
    try:
        wavaudio = AudioSegment.from_wav(wavfile)
        wavaudio = wavaudio.set_frame_rate(16000).set_channels(1)
        wavaudio = effects.normalize(wavaudio)
        wavaudio.export(mp3file, format="mp3", bitrate="16k")
        wavfile.unlink()
        return mp3file
    except Exception as e:
        if wavfile.is_file():
            wavfile.unlink()


def main():
    mp3files = []
    source_files = list(pathlib.Path("../data/source_lists/").rglob("*_file_list.txt"))
    for source_file in source_files:
        print(f"downloading {source_file}")
        urls = list(map(lambda x: x.strip(), source_file.open("r").readlines()))
        print("Downloading archives")
        zipfiles = []
        with ThreadPoolExecutor(cpu_count() * 4) as executor:
            results = executor.map(download_archive, urls)
            for url in tqdm(results):
                zipfiles.append(url)

        with ThreadPoolExecutor(max_workers=len(zipfiles)) as executor:
            print("Extracting archives")
            extracted = [executor.submit(extract_zipfile, file) for file in zipfiles]
            for wavfiles in as_completed(extracted):
                wavfiles = wavfiles.result()
                if wavfiles:
                    with Pool(cpu_count() - 1) as pool:
                        results = pool.imap_unordered(convert_wav_to_mp3, wavfiles)
                        print("Converting wav to mp3")
                        for mp3file in tqdm(results, total=len(wavfiles)):
                            mp3files.append(mp3file)


if __name__ == "__main__":
    main()
