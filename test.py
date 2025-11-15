import os
import time
import json
import random
import pandas as pd
import numpy as np
import yt_dlp
import kagglehub
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- CONFIG ----------
COOKIES_FILE = "cookies.txt"    # <- place your exported cookies.txt here
SAMPLE_N = 200
MAX_COMMENTS = 20
SLEEP_MIN, SLEEP_MAX = 2.0, 6.0
MAX_RETRIES = 3
BACKOFF_BASE = 2.0
OUTPUT_CSV = "final_200_movie_content_scores.csv"
# ----------------------------

# quick checks
if not os.path.exists(COOKIES_FILE):
    raise FileNotFoundError(f"cookies.txt not found at path: {COOKIES_FILE}. Place it next to this script.")

# install check: make sure kagglehub dataset path download was already done in previous flow.
path = kagglehub.dataset_download("dineshvasired/movies-youtube-trailers-and-sentimentdinesh-dinesh")
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("No CSV found in downloaded dataset path: " + str(path))
csv_file = csv_files[0]

df = pd.read_csv(os.path.join(path, csv_file))
print("Loaded", csv_file, "with", len(df), "rows")
print("Columns:", df.columns.tolist())

# choose trailer link & title column robustly
possible_link_cols = ["trailer_link", "youtube_link", "youtube_trailer_url", "trailer", "video_url", "video_id"]
link_col = next((c for c in possible_link_cols if c in df.columns), None)
if link_col is None:
    raise ValueError("No trailer link column found. Add the correct column name to possible_link_cols.")

possible_title_cols = ["title", "name", "movie", "movie_title"]
title_col = next((c for c in possible_title_cols if c in df.columns), None)
if title_col is None:
    title_col = df.columns[0]  # fallback to first column

print("Using link column:", link_col, "and title column:", title_col)

# filter, remove NaNs, ensure contains 'youtube'
df_links = df[df[link_col].notna() & df[link_col].astype(str).str.contains("youtube", na=False)].copy()
df_links = df_links.drop_duplicates(subset=[link_col]).reset_index(drop=True)
if df_links.empty:
    raise ValueError("No valid YouTube links found in dataset.")

SAMPLE_N = min(SAMPLE_N, len(df_links))
df_sample = df_links.sample(n=SAMPLE_N, random_state=42).reset_index(drop=True)
print("Selected", len(df_sample), "unique movies for processing.")

# keyword lists (you may expand)
VIOLENCE_WORDS = ["kill","fight","blood","gun","shoot","war","violence","murder","gore","stab","dead"]
SEX_WORDS = ["sex","nudity","nude","naked","topless","intimate","kiss","erotic","bed scene"]
PROFANITY_WORDS = ["fuck","shit","bitch","asshole","damn","whore","cunt","dick","bastard","fucker","motherfucker"]
DRUGS_WORDS = ["drug","weed","alcohol","cocaine","heroin","smoke","drunk","marijuana","cannabis"]
INTENSE_WORDS = ["scary","frightening","intense","horror","jump scare","terrifying","panic","scream","suspense"]

analyzer = SentimentIntensityAnalyzer()

def classify_and_score(comments):
    """Return aggregated features from comment list."""
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
    v=s=p=d=i=0
    for c in comments:
        if not isinstance(c, str):
            c = str(c)
        text = c.lower()
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

def fetch_comments_with_cookies(url, max_comments=20, cookiefile=COOKIES_FILE):
    """
    Use yt-dlp with a cookiefile to extract top-level comments.
    Returns list of comment strings (may be empty).
    """
    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "cookiefile": os.path.abspath(cookiefile),
        # optionally: "socket_timeout": 30
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        # pass exception message up for logging
        return {"error": str(e), "comments": []}

    # collect comments
    comments = []
    if info and isinstance(info, dict) and "comments" in info and info["comments"]:
        for c in info["comments"][:max_comments]:
            # comment structure: {"text": "...", ...}
            txt = c.get("text") if isinstance(c, dict) else str(c)
            comments.append(txt)
    return {"error": None, "comments": comments}

results = []
for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Movies"):
    title = row.get(title_col, f"movie_{idx}")
    url = row.get(link_col)
    if not isinstance(url, str) or not url.strip():
        print(f"[{idx}] skipping {title} (no url)")
        continue

    print(f"\n[{idx+1}/{len(df_sample)}] Scraping: {title} -> {url}")

    # retry logic
    comments = []
    last_err = None
    for attempt in range(1, MAX_RETRIES+1):
        res = fetch_comments_with_cookies(url, max_comments=MAX_COMMENTS, cookiefile=COOKIES_FILE)
        if res["error"]:
            last_err = res["error"]
            wait = BACKOFF_BASE ** attempt
            print(f"Attempt {attempt} failed: {last_err}. Backing off {wait}s...")
            time.sleep(wait)
            continue
        comments = res["comments"]
        break

    if last_err and not comments:
        print(f"Failed to fetch comments for {title} after {MAX_RETRIES} attempts. Error: {last_err}")
    else:
        print(f"Fetched {len(comments)} comments.")

    features = classify_and_score(comments)
    features.update({
        "title": title,
        "trailer_url": url,
        "comments_json": json.dumps(comments, ensure_ascii=False)
    })
    results.append(features)

    # polite sleep between videos to reduce risk of rate limits
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

# build dataframe and save
df_out = pd.DataFrame(results)
print("Saving", OUTPUT_CSV, "with", len(df_out), "rows.")
df_out.to_csv(OUTPUT_CSV, index=False)
print("Done. File saved:", OUTPUT_CSV)
