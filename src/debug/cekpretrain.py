import torch
import os
import requests

PRETRAIN_URL = "https://github.com/mims-harvard/UniTS/releases/download/ckpt/units_x128_pretrain_checkpoint.pth"
PRETRAIN_PATH = "units_x128_pretrain_checkpoint.pth"

def download_pretrain_model():
    if not os.path.exists(PRETRAIN_PATH):
        print("Downloading pretrain model...")
        response = requests.get(PRETRAIN_URL, stream=True)
        with open(PRETRAIN_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print("Download complete.")
    else:
        print("Pretrain model already exists.")

def check_checkpoint():
    if not os.path.exists(PRETRAIN_PATH):
        print("Checkpoint file not found.")
        return
    
    checkpoint = torch.load(PRETRAIN_PATH, map_location=torch.device('cpu'))
    print("Checkpoint keys:", checkpoint.keys())

if __name__ == "__main__":
    download_pretrain_model()
    check_checkpoint()
