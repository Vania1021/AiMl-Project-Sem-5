import os, json
import yt_dlp

api_key = "AIzaSyB-rllfudN_11ffdQGmS1_HXqwmy4b4eYg"


COOKIES = "cookies.txt"   # path to your exported cookies.txt
VIDEO = "https://www.youtube.com/watch?v=LDxgPIsv6sY"  # replace one video that returned 0 comments

ydl_opts = {
    "skip_download": True,
    "cookiefile": os.path.abspath(COOKIES),
    "quiet": False,    # show messages
    "no_warnings": False,
    # debug/verbose options
    # "verbose": True   # yt-dlp CLI supports -v; in code we keep quiet False
}

def inspect_video(url):
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        print("yt-dlp exception:", e)
        info = None

    if not info:
        print("No info returned (possible block or cookie problem).")
        return
    
    return info.get("id",None)

    
print(inspect_video(VIDEO))


from googleapiclient.discovery import build

youtube = build("youtube", "v3", developerKey=api_key)

def fetch_comments_api(video_id, max_results=5):
    comments = []
    req = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=min(max_results,5),
        textFormat="plainText",
        order="relevance"
    )
    res = req.execute()
    while res:
        for item in res.get("items", []):
            txt = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(txt)
        if "nextPageToken" in res and len(comments) < max_results:
            req = youtube.commentThreads().list(
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
    return comments


print(fetch_comments_api(inspect_video(VIDEO)))