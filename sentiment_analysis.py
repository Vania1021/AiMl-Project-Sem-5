#!/usr/bin/env python3
"""
Final pipeline:
- loads Kaggle dataset
- samples N unique trailers
- tries yt-dlp (cookiefile + remote_components) to get comments
- if yt-dlp fails/returns empty, fallback to YouTube Data API v3 (needs API_KEY)
- computes VADER sentiment + 5 category counts
- saves CSV with source info (yt-dlp or youtube_api)
"""

import os
import time
import json
import random
import argparse
from urllib.parse import urlparse, parse_qs
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import yt_dlp
import kagglehub
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional Google API client
try:
    from googleapiclient.discovery import build
except Exception:
    build = None

# ---------------- CONFIG ----------------
COOKIES_FILE = "cookies.txt"    # place your exported cookies.txt here
API_KEY = ""                    # <-- PUT YOUR YouTube Data API KEY HERE (or leave empty to disable fallback)
SAMPLE_N = 200
MAX_COMMENTS = 20
SLEEP_MIN, SLEEP_MAX = 2.0, 6.0
MAX_RETRIES = 3
BACKOFF_BASE = 2.0
OUTPUT_CSV = "final_200_movie_content_scores_with_fallback.csv"
# ----------------------------------------

# sanity checks
if not os.path.exists(COOKIES_FILE):
    raise FileNotFoundError(f"cookies.txt not found at path: {COOKIES_FILE}. Place it next to this script.")

if API_KEY == "":
    print("WARNING: API_KEY is empty. The script will attempt yt-dlp only and will NOT call YouTube Data API fallback.")

# helper: parse video id from youtube url or accept direct id
def extract_video_id(url_or_id: str) -> Optional[str]:
    if not isinstance(url_or_id, str) or url_or_id.strip() == "":
        return None
    s = url_or_id.strip()
    # if looks like a raw id (11 chars)
    if len(s) == 11 and all(c.isalnum() or c in ["-", "_"] for c in s):
        return s
    # parse URL
    try:
        p = urlparse(s)
        if p.hostname and "youtube" in p.hostname:
            q = parse_qs(p.query)
            if "v" in q:
                return q["v"][0]
        # youtu.be shortlink
        if p.hostname and "youtu.be" in p.hostname:
            return p.path.lstrip("/")
    except Exception:
        pass
    return None

# Initialize YouTube API client if API_KEY provided
youtube_service = None
if API_KEY and build:
    youtube_service = build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)
elif API_KEY and not build:
    print("googleapiclient not available; install google-api-python-client to use API fallback.")

# VADER & keyword lists
analyzer = SentimentIntensityAnalyzer()
VIOLENCE_WORDS = ["kill","fight","blood","gun","shoot","war","violence","murder","gore","stab","dead"]
SEX_WORDS = ["sex","nudity","nude","naked","topless","intimate","kiss","erotic","bed scene"]
PROFANITY_WORDS = ["fuck","shit","bitch","asshole","damn","whore","cunt","dick","bastard","fucker","motherfucker"]
DRUGS_WORDS = ["drug","weed","alcohol","cocaine","heroin","smoke","drunk","marijuana","cannabis"]
INTENSE_WORDS = ["scary","frightening","intense","horror","jump scare","terrifying","panic","scream","suspense"]

def classify_and_score(comments: List[str]) -> dict:
    if not comments:
        return {
            "num_comments": 0,
            "avg_sentiment": 0.0,
            "violence_count": 0,
            "sex_nudity_count": 0,
            "profanity_count": 0,
            "drugs_count": 0,
            "intense_count": 0
        }
    sentiments = []
    v=s=p=d=i = 0,0,0,0,0  # workaround for typing below
    v = s = p = d = i = 0
    for c in comments:
        text = str(c).lower()
        sentiments.append(analyzer.polarity_scores(text)["compound"])
        v += sum(1 for w in VIOLENCE_WORDS if w in text)
        s += sum(1 for w in SEX_WORDS if w in text)
        p += sum(1 for w in PROFANITY_WORDS if w in text)
        d += sum(1 for w in DRUGS_WORDS if w in text)
        i += sum(1 for w in INTENSE_WORDS if w in text)
    return {
        "num_comments": len(comments),
        "avg_sentiment": float(np.mean(sentiments)) if sentiments else 0.0,
        "violence_count": int(v),
        "sex_nudity_count": int(s),
        "profanity_count": int(p),
        "drugs_count": int(d),
        "intense_count": int(i)
    }

# yt-dlp comment fetcher (uses cookies and remote components)
def fetch_comments_yt_dlp(url: str, max_comments: int = 20, cookiefile: str = COOKIES_FILE) -> Tuple[Optional[str], List[str]]:
    """
    Returns (error_message_or_None, comments_list).
    """
    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "cookiefile": os.path.abspath(cookiefile),
        # enable remote components (EJS) to solve JS challenges if needed
        "remote_components": "ejs:github",
        # socket timeout for safety
        "socket_timeout": 30
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        return (str(e), [])

    comments = []
    if info and isinstance(info, dict) and "comments" in info and info["comments"]:
        for c in info["comments"][:max_comments]:
            txt = c.get("text") if isinstance(c, dict) else str(c)
            comments.append(txt)
    return (None, comments)

# YouTube Data API fallback
def fetch_comments_youtube_api(video_id: str, max_results: int = 20) -> List[str]:
    if not youtube_service:
        return []
    comments = []
    try:
        req = youtube_service.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results),
            textFormat="plainText",
            order="relevance"
        )
        res = req.execute()
        while res:
            for item in res.get("items", []):
                txt = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(txt)
            if "nextPageToken" in res and len(comments) < max_results:
                req = youtube_service.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=res["nextPageToken"],
                    maxResults=min(100, max_results - len(comments)),
                    textFormat="plainText",
                    order="relevance"
                )
                res = req.execute()
            else:
                break
    except Exception as e:
        print("YouTube API error:", e)
    return comments[:max_results]

# Main pipeline
def main(sample_n=SAMPLE_N, max_comments=MAX_COMMENTS, output_csv=OUTPUT_CSV):
    # download Kaggle dataset
    path = kagglehub.dataset_download("dineshvasired/movies-youtube-trailers-and-sentimentdinesh-dinesh")
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV found in downloaded dataset path: " + str(path))
    csv_file = csv_files[0]
    df = pd.read_csv(os.path.join(path, csv_file))
    print("Loaded", csv_file, "with", len(df), "rows")
    # select columns
    possible_link_cols = ["trailer_link", "youtube_link", "youtube_trailer_url", "trailer", "video_url", "video_id"]
    link_col = next((c for c in possible_link_cols if c in df.columns), None)
    if link_col is None:
        raise ValueError("No trailer link column found. Update possible_link_cols.")
    possible_title_cols = ["title", "name", "movie", "movie_title"]
    title_col = next((c for c in possible_title_cols if c in df.columns), df.columns[0])

    # filter, dedupe, sample
    df_links = df[df[link_col].notna() & df[link_col].astype(str).str.contains("youtube", na=False)].copy()
    df_links = df_links.drop_duplicates(subset=[link_col]).reset_index(drop=True)
    if df_links.empty:
        raise ValueError("No valid YouTube links found in dataset.")
    sample_n = min(sample_n, len(df_links))
    df_sample = df_links.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print("Selected", len(df_sample), "unique movies for processing.")

    rows = []
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Movies"):
        title = row.get(title_col, f"movie_{idx}")
        url = str(row.get(link_col))
        print(f"\n[{idx+1}/{len(df_sample)}] {title} -> {url}")

        # 1) try yt-dlp with retries/backoff
        comments = []
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            err, comments = fetch_comments_yt_dlp(url, max_comments=max_comments, cookiefile=COOKIES_FILE)
            if err:
                last_err = err
                wait = BACKOFF_BASE ** attempt
                print(f"yt-dlp attempt {attempt} error: {err} — backing off {wait}s")
                time.sleep(wait)
                continue
            # success but maybe empty list
            break

        source = "yt-dlp"
        # 2) if no comments from yt-dlp, fallback to YouTube Data API (if available)
        if (not comments or len(comments) == 0) and API_KEY and youtube_service:
            vid = extract_video_id(url)
            if vid:
                print("yt-dlp returned no comments; falling back to YouTube Data API for video id:", vid)
                comments = fetch_comments_youtube_api(vid, max_results=max_comments)
                source = "youtube_api" if comments else source
            else:
                print("Could not extract video id from URL, skipping API fallback.")

        # 3) log if still empty
        if not comments:
            print(f"No comments obtained for {title}. last yt-dlp err: {last_err}")

        # compute features
        features = classify_and_score(comments)
        features.update({
            "title": title,
            "trailer_url": url,
            "comments_json": json.dumps(comments, ensure_ascii=False),
            "source": source,
            "yt_dlp_error": last_err
        })
        rows.append(features)

        # polite sleep
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print("Saved results to:", output_csv)
    return df_out

if __name__ == "__main__":
    # optional CLI override
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-n", type=int, default=SAMPLE_N)
    parser.add_argument("--max-comments", type=int, default=MAX_COMMENTS)
    parser.add_argument("--output", type=str, default=OUTPUT_CSV)
    args = parser.parse_args()
    main(sample_n=args.sample_n, max_comments=args.max_comments, output_csv=args.output)
