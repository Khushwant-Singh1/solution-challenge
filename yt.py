import yt_dlp

def download_video(url, path='.'):
    ydl_opts = {
        'outtmpl': f'{path}/%(title)s.%(ext)s',
        'format': 'bestvideo+bestaudio/best',
        'noplaylist': True,
        'quiet': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download completed successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    video_url = input("Enter YouTube URL: ")
    download_dir = input("Output directory (Enter for current): ") or '.'
    download_video(video_url, download_dir)