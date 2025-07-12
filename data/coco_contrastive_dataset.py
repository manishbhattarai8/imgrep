import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class CocoContrastiveDataset(Dataset):
    def __init__(self, image_folder, caption_file, tokenizer, transform, max_len=40):
        with open(caption_file, 'r') as f:
            data = json.load(f)

        self.image_folder = image_folder
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

        id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        self.samples = [
            (os.path.join(image_folder, id_to_filename[ann['image_id']]), ann['caption'])
            for ann in data['annotations']
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, caption = self.samples[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        tokens = self.tokenizer.encode(caption, self.max_len)
        tokens = torch.tensor(tokens, dtype=torch.long)
        return image, tokens
