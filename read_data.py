import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPImageProcessor
from datasets import load_dataset
from torchvision import transforms
import random
import numpy as np
from PIL import Image
import datasets


def gen_meta(cur_fold=1):
    info = {
        'normal': {
            'length': 120,
            'category': 'normal ',
            'category_2': 'normal ',
            'category_3': 'np ',
            'grade': 0,
            'fold': [25, 49, 73, 97, 120],
            'left': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 33, 36, 43, 44, 45, 46, 47, 48, 49,
                     50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 69, 71, 73, 75, 77, 78, 80, 82,
                     83, 85, 87, 89, 92, 95, 96, 98, 100, 102, 103, 104, 105, 108, 109, 110, 115, 117]
        },
        'npdr': {
            'length': 172,
            'category': 'npdr ',
            'category_2': 'dr ',
            'category_3': 'np ',
            'grade': 1,
            'fold': [35, 69, 106, 140, 172],
            'left': [2, 4, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 31, 33, 35, 37, 40, 41, 42, 45, 47,
                     49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89,
                     91, 93, 95, 102, 107, 108, 109, 110, 111, 112, 113, 115, 117, 119, 121, 123, 125, 127,
                     129, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 159, 161,
                     163, 165, 167, 169, 171]
        },
        'pdr': {
            'length': 110,
            'category': 'pdr ',
            'category_2': 'dr ',
            'category_3': 'pdr ',
            'grade': 2,
            'fold': [23, 44, 66, 89, 110],
            'left': [2, 4, 10, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 38, 40, 42, 44, 46, 48,
                     50, 51, 53, 54, 58, 60, 62, 64, 65, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
                     87, 88, 89, 92, 94, 95, 97, 99, 101, 103, 105, 107, 109]
        }
    }

    data_list = {
        '1': [],
        '2': [],
        '3': [],
        '4': [],
        '5': [],
    }
    for label in info:
        for i in range(info[label]['length']):
            side = 'left fundus' if i in info[label]['left'] else 'right fundus'
            fold = 0
            for fold_, max_id in enumerate(info[label]['fold']):
                if i < max_id:
                    fold = fold_ + 1
                    break
            for period in ["an early", "a late"]:
                text_list = [
                    "a ffa of a fundus",

                    period + " ffa of a fundus",
                    "a ffa of a " + side,
                    period + " ffa of a " + side,

                    "a ffa of a " + info[label]['category'] + "fundus"
                    "a ffa of a " + info[label]['category'] + side,
                    period + " ffa of a " + info[label]['category'] + "fundus",
                    period + " ffa of a " + info[label]['category'] + side,

                    "a ffa of a " + info[label]['category_2'] + "fundus"
                    "a ffa of a " + info[label]['category_2'] + side,
                    period + " ffa of a " + info[label]['category_2'] + "fundus",
                    period + " ffa of a " + info[label]['category_2'] + side,

                    "a ffa of a " + info[label]['category_3'] + "fundus"
                    "a ffa of a " + info[label]['category_3'] + side,
                    period + " ffa of a " + info[label]['category_3'] + "fundus",
                    period + " ffa of a " + info[label]['category_3'] + side,
                ]

                period_ = period.replace('a ', '').replace('an ', '')
                d = {
                    "file_name": f"images/{label}/{i}_{period_}.bmp",
                    "category": info[label]['category'].replace(' ', ''),
                    "embeds_name": info[label]['category'].replace(' ', '') + f'_{i}_{period_}.tensor',
                    "text": period + " ffa of a " + info[label]['category'] + side,
                    "text_list": text_list,
                    "index": i,
                    "side": 0 if i in info[label]['left'] else 1,
                    "period": 0 if period_ == "early" else 1,
                    "grade": info[label]['grade'],
                }
                data_list[f'{fold}'].append(d)

    with open(f'datasets/UWFA/metadata_all_{cur_fold}.jsonl', 'w') as file:
        for j in range(5):
            for data in data_list[f'{j + 1}']:
                data['embeds'] = f'image_embeds/fold{cur_fold}_' + data['embeds_name']
                json_data = json.dumps(data)
                file.write(json_data + "\n")
    with open(f'datasets/UWFA/metadata.jsonl', 'w') as file:
        for j in range(5):
            if j != cur_fold:
                for data in data_list[f'{j + 1}']:
                    data['embeds'] = f'image_embeds/fold{cur_fold}_' + data['embeds_name']
                    json_data = json.dumps(data)
                    file.write(json_data + "\n")
    with open(f'datasets/UWFA/metadata_test_{cur_fold}.jsonl', 'w') as file:
        for data in data_list[f'{cur_fold}']:
            data['embeds'] = f'image_embeds/fold{cur_fold}_' + data['embeds_name']
            json_data = json.dumps(data)
            file.write(json_data + "\n")


class UWFADataset(Dataset):
    def __init__(self, root,
                 model_id_org,
                 fold=1,
                 split='train',
                 is_train=True):
        if not os.path.exists(f'datasets/UWFA/metadata_all_{fold}.jsonl'):
            gen_meta(fold)
        self.root = root
        self.processor = CLIPImageProcessor.from_pretrained(model_id_org)
        self.is_train = is_train
        self.data_list = []
        assert split in ['train', 'test', 'all']
        if split == 'train':
            json_name = 'metadata.jsonl'
        else:
            json_name = f'metadata_{split}_{fold}.jsonl'
        with open(os.path.join(root, json_name), 'r') as f:
            for line in f:
                d = json.loads(line)
                self.data_list.append(d)

        self.length = len(self.data_list)

    def get_pixel_values(self, img):
        img = self.processor(images=img, return_tensors='pt', padding=True)
        img = img['pixel_values'].squeeze(0)
        return img

    def __getitem__(self, item):
        d = self.data_list[item]

        text_early = random.choice(d['text_list']).replace('late', 'early')
        text_late = random.choice(d['text_list']).replace('early', 'late')

        data_image_org = Image.open(os.path.join(self.root, d['file_name'])).resize((224, 224))
        data_image_early = Image.open(os.path.join(self.root, d['file_name'].replace('late', 'early'))).resize((224, 224))
        data_image_late = Image.open(os.path.join(self.root, d['file_name'].replace('early', 'late'))).resize((224, 224))

        if self.is_train:
            if random.random() < 0.5:
                data_image_early = data_image_early.transpose(Image.FLIP_LEFT_RIGHT)
                if 'left' in text_early:
                    text_early = text_early.replace('left', 'right')
                elif 'right' in text_early:
                    text_early = text_early.replace('right', 'left')

                data_image_late = data_image_late.transpose(Image.FLIP_LEFT_RIGHT)
                if 'left' in text_late:
                    text_late = text_late.replace('left', 'right')
                elif 'right' in text_late:
                    text_late = text_late.replace('right', 'left')

        data_image_org = self.processor(images=data_image_org, return_tensors='pt', padding=True)
        pixel_values_org = data_image_org['pixel_values'].squeeze(0)
        data_image_early = self.processor(images=data_image_early, return_tensors='pt', padding=True)
        pixel_values_early = data_image_early['pixel_values'].squeeze(0)
        data_image_late = self.processor(images=data_image_late, return_tensors='pt', padding=True)
        pixel_values_late = data_image_late['pixel_values'].squeeze(0)

        return {
            'file_name': d['file_name'],
            'embed': d['embeds'],
            'text_early': text_early,
            'text_late': text_late,
            'pixel_values_org': pixel_values_org,
            'pixel_values_early': pixel_values_early,
            'pixel_values_late': pixel_values_late,
            'side': d['side'],
            'grade': d['grade'],
        }

    def __len__(self):
        return self.length


if __name__ == '__main__':
    gen_meta()
