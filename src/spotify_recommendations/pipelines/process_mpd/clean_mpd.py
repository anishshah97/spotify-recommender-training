import concurrent.futures
import json
import os
from itertools import repeat
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm


# Note: Assuming the workspace the script is coming from is the relational path from which to parse data from
def determine_paths_to_scrape():
    data_path = Path(Path().resolve(), os.environ.get("DATA_DIR", "data"))
    raw_mpd_dir = Path(data_path, "01_raw", "spotify_mpd")
    raw_mpd_slice_paths = {path.name.replace(
        ".json", ""): path for path in raw_mpd_dir.glob("mpd.slice.*.json")}
    cleaned_mpd_dir = Path(data_path, "02_intermediate", "cleaned_spotify_mpd")
    cleaned_mpd_slice_paths = {path.name.replace(".parquet", "").replace(
        "_", "."): path for path in cleaned_mpd_dir.glob("mpd_slice_*")}
    slices_to_clean = set(raw_mpd_slice_paths.keys()) - \
        set(cleaned_mpd_slice_paths.keys())
    logger.info(f"""
    Number of files: {len(raw_mpd_slice_paths)}
    Number of files cleaned: {len(cleaned_mpd_slice_paths)}
    Number of files to clean: {len(slices_to_clean)}
    """)
    raw_slices_to_clean_paths = [
        (name, path) for name, path in raw_mpd_slice_paths.items() if name in slices_to_clean]
    logger.info(len(raw_slices_to_clean_paths))
    return raw_slices_to_clean_paths, cleaned_mpd_dir


def parse_track_id_from_uri(uri):
    return uri.split(":")[2]


# TODO: Match form of scraping for application and use this as these files to seed insert into a db
# TODO: Read from DB and do operations like scrape songs whose features arent in the feature db
def clean_mpd_slice(raw_slice_to_clean_path, base_path):
    name = raw_slice_to_clean_path[0]
    path = raw_slice_to_clean_path[1]
    with open(path) as f:
        play_lst = []
        track_lst = []
        seen_tracks = set()
        data = json.load(f)
        playlists = data["playlists"]

        # for each playlist
        for playlist in playlists:
            for track in playlist["tracks"]:
                if track["track_uri"] not in seen_tracks:
                    seen_tracks.add(track["track_uri"])
                    track_lst.append(track)
            playlist["track_ids"] = [parse_track_id_from_uri(
                x["track_uri"]) for x in playlist["tracks"]]
            play_lst.append(playlist)

        playlist_df = pd.DataFrame(play_lst)
        playlist_table = pa.Table.from_pandas(playlist_df)

        tracks_df = pd.DataFrame(track_lst)
        tracks_df["track_id"] = tracks_df.apply(
            lambda row: parse_track_id_from_uri(row["track_uri"]), axis=1)
        tracks_table = pa.Table.from_pandas(tracks_df)

        cleaned_slice_dir = Path(base_path, name.replace(".", "_"))
        cleaned_slice_dir.mkdir(parents=True, exist_ok=True)

        pq.write_table(playlist_table, Path(
            cleaned_slice_dir, "playlist.parquet"), version="2.0")
        pq.write_table(tracks_table, Path(
            cleaned_slice_dir, "tracks.parquet"), version="2.0")

    return True


def multiprocess_clean_mpd(**kwargs):
    MAX_THREADS = 30
    raw_slices_to_clean_paths, cleaned_mpd_dir = determine_paths_to_scrape()
    threads = min(MAX_THREADS, len(raw_slices_to_clean_paths))
    if threads < 1:
        threads = 1
    # BUG: ThreadPool or ProcessPool?
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        result = list(
            tqdm(executor.map(clean_mpd_slice, raw_slices_to_clean_paths, repeat(cleaned_mpd_dir))))
    return result
