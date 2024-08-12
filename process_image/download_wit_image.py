import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import random
import time
from urllib.request import urlopen

import PIL
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

PIL.Image.MAX_IMAGE_PIXELS = 1_000_000_000

def random_delay():
    delay = random.uniform(0.2, 1.0)
    time.sleep(delay)

def download_image(url, save_path):
    if os.path.exists(save_path):
        try:
            image = Image.open(save_path)
            image.verify()
            image.close()
            return
        except Exception as e:
            print(f"Exception: {e}")
            print(f"Image file {save_path} is corrupted. Downloading again.")
        # print("File already exists:", save_path)
    # Create a requests session
    session = requests.Session()
    
    # Define a retry strategy
    retries = Retry(total=20, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    # Define headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    try:
        # Send a HTTP GET request to the image URL
        response = session.get(url, headers=headers, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Open a local file with write-binary mode
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            # print(f"Image successfully downloaded: {save_path}")
        else:
            print(f"Failed to retrieve image. HTTP Status code: {response.status_code}, url: {url}")
    except Exception as e:
        print(f"An error occurred: {e}, url: {url}")

def process_batch(batch):
    image_ids = batch["id"]
    image_urls = batch["image_url"]
    for image_id, image_url in zip(image_ids, image_urls):
        save_path = os.path.join(output_dir, f"{image_id}.jpg")
        download_image(image_url, save_path)

if __name__ == '__main__':
    dataset = load_dataset("Vi-VLM/Vista", name="vi_wit", split="train")

    output_dir = '/mnt/disks/dev/data/images/wit/images'
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 20000  # Adjust batch size as needed
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            futures.append(executor.submit(process_batch, batch))
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()

    print("Image download completed.")

