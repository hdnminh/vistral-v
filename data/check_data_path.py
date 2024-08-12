import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="data/vi_sharegpt4v.json", help='Path to the dataset')
    parser.add_argument('--image-folder', type=str, default="/mnt/disks/dev/data/images", help='Path to the image folder')
    args = parser.parse_args()
    
    data_path = args.data_path
    image_folder = args.image_folder
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File {data_path} not found")
    
    error = None
    try:
        with open(data_path, "r") as f:
            data = json.load(f)

        for idx, conv in enumerate(data):
            image = conv["image"]
            conversations = conv["conversations"]
            image_path = os.path.join(image_folder, image)

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} not found in conversation {idx}")

            if len(conversations) < 2:
                print(conversations)
                print("=" * 10)
                raise ValueError(f"Conversation {image} with len {len(conversations)} is not a 2-turn conversation")
    except Exception as e:
        print(f"Error: {e}")
        error = e
        
    if error is None:
        print("All images and conversations are valid")
    