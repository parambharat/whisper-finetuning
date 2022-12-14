import json
import pathlib
from collections import defaultdict

import pandas as pd
from shortuuid import uuid
from tqdm import tqdm


def map_new_path(row):
    filename = uuid(str(row["old_path"]) + row["sentence"])
    lang_file = "train" / pathlib.Path(filename)
    lang_file = row["lang"] / lang_file
    lang_file = ("../data" / lang_file).with_suffix(".mp3")
    return lang_file


def move_data_to_new_path(df):
    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            source = row["old_path"]
            destination = row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            rows.append(row)
        except:
            pass
    rows_df = pd.DataFrame(rows)
    rows_df = rows_df[["path", "sentence", "duration", "gender",]].reset_index(
        drop=True
    )
    return rows_df


def main():
    lang_dirs = defaultdict(list)
    source_lists = list(pathlib.Path("../data/source_lists/").rglob("*_file_list.txt"))
    for source in source_lists:
        for url in source.open("r").readlines():
            source_dir = pathlib.Path(pathlib.Path(url.strip()).stem)
            source_dir = "../data/datasets" / source_dir
            lang_dirs[source.stem.split("_")[0]].append(source_dir)

    lang_dfs = []
    for lang, dirs in lang_dirs.items():
        lang_df = []
        for directory in dirs:
            media_dir = directory / "mp3"
            data_file = directory / "data.json"
            if data_file.is_file():
                data = json.load(data_file.open("r"))
                data_df = pd.DataFrame(data)
                data_df["old_path"] = data_df["audioFilename"].map(
                    lambda x: media_dir / pathlib.Path(x).with_suffix(".mp3")
                )
                is_sentence = data_df["text"].map(
                    lambda x: x is not None and len(x) > 3
                )
                data_df = data_df[is_sentence]
                is_file = data_df["old_path"].map(lambda x: x.is_file())
                data_df = data_df[is_file]
                if not data_df.empty:
                    data_df = data_df.rename({"text": "sentence"}, axis=1)
                    data_df = data_df.drop_duplicates(["sentence"])
                    data_df["lang"] = lang
                    data_df["path"] = data_df.apply(map_new_path, axis=1)
                    data_df = data_df[
                        ["old_path", "path", "sentence", "duration", "gender", "lang"]
                    ]
                    lang_df.append(data_df)
            else:
                print(directory)
        lang_df = pd.concat(lang_df).reset_index(drop=True)
        lang_dfs.append(lang_df)

    moved_dfs = []
    for lang_df in tqdm(lang_dfs):
        moved_df = move_data_to_new_path(lang_df)
        moved_dfs.append(moved_df)

    metadata_paths = []
    for df in moved_dfs:
        metadata_path = df.iloc[0]["path"].parent.parent / "metadata.jsonl"
        metadata_paths.append(metadata_path)

    for df, metadata_path in zip(moved_dfs, metadata_paths):
        df["path"] = df["path"].str.split("/").map(lambda x: "train/" + x[-1])
        df["file_name"] = df["path"]
        df.to_json(metadata_path, lines=True, orient="records")


if __name__ == "__main__":
    main()
