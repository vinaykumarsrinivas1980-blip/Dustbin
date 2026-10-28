import os
import re
import sys
import time
import hashlib
import argparse
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

# Default search terms for each category
SEARCH_QUERIES = {
    "biomedical-waste": [
        "biomedical waste hospital",
        "biohazard waste syringes blood",
        "hospital hazardous infectious waste",
        "used medical syringes vials waste",
        "clinical biological waste containers"
    ],
    "medical_waste": [
        "medical waste surgical masks gloves",
        "hospital plastic medical waste",
        "used surgical gloves hospital disposal",
        "clinical non-hazardous waste",
        "medical pill blister packs discarded"
    ],
    "normal": [
        "empty waste bin hopper tray",
        "clean plastic container background",
        "white tabletop clean conveyor",
        "empty dustbin interior top view"
    ]
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/118.0.0.0 Safari/537.36"
    )
}


def fetch_image_urls(query, max_results=100):
    """
    Search DuckDuckGo Images and extract direct image URLs.
    """
    urls = []
    try:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")
        
        # Extract image links from HTML
        found = re.findall(r'uddg=([^&"\']+)', html)
        for item in found:
            decoded = urllib.parse.unquote(item)
            if any(decoded.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.webp')):
                if decoded not in urls:
                    urls.append(decoded)
            if len(urls) >= max_results:
                break
    except Exception as e:
        print(f"  [Warning] Query '{query}' error: {e}")
    return urls


def download_single_image(url, save_path, timeout=8):
    """
    Download a single image file and verify it is non-empty.
    """
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
        
        # Verify non-trivial size (> 2KB)
        if len(data) < 2048:
            return False

        with open(save_path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def process_category(category, queries, target_dir, limit=50, max_workers=6):
    """
    Fetch and download images for a given category.
    """
    cat_dir = os.path.join(target_dir, category)
    os.makedirs(cat_dir, exist_ok=True)
    
    print(f"\n[Category: {category}] Collecting URLs...")
    all_urls = []
    for q in queries:
        urls = fetch_image_urls(q, max_results=limit // len(queries) + 10)
        all_urls.extend(urls)
        time.sleep(0.5)

    all_urls = list(dict.fromkeys(all_urls))[:limit]
    print(f"[Category: {category}] Found {len(all_urls)} target URLs. Starting download...")

    downloaded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, url in enumerate(all_urls):
            ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png', '.webp'):
                ext = '.jpg'
            file_name = f"ref_{category}_{i+1:04d}{ext}"
            file_path = os.path.join(cat_dir, file_name)
            futures.append(executor.submit(download_single_image, url, file_path))

        for f in futures:
            if f.result():
                downloaded += 1

    print(f"[Category: {category}] Successfully downloaded {downloaded} images to '{cat_dir}'")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download reference images for waste classification (stored separately from training data).")
    parser.add_argument("--limit", type=int, default=30, help="Number of images per category to download (default: 30)")
    parser.add_argument("--output", type=str, default="reference_images", help="Target output folder (default: reference_images)")
    args = parser.parse_args()

    base_output = os.path.abspath(args.output)
    os.makedirs(base_output, exist_ok=True)

    print("=" * 65)
    print("AI SMART DUSTBIN — REFERENCE IMAGE DOWNLOADER")
    print(f"Target Output Directory: {base_output}")
    print(f"Images Per Category: {args.limit}")
    print("=" * 65)

    total_downloaded = 0
    for category, queries in SEARCH_QUERIES.items():
        total_downloaded += process_category(category, queries, base_output, limit=args.limit)

    print("\n" + "=" * 65)
    print(f"DOWNLOAD COMPLETE: {total_downloaded} total reference images acquired.")
    print(f"Location: {base_output}")
    print("Note: These images are kept isolated from dataset/ for reference only.")
    print("=" * 65)


if __name__ == "__main__":
    main()
