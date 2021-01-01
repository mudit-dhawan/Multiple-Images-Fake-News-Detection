import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import torch
import numpy as np
import transformers
from torchvision import models, transforms
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
import re
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class FakeNewsDataset(Dataset):
    """Fake News Dataset"""

    def __init__(self, df, image_transform, tokenizer, MAX_LEN):
        """
        Args:
            csv_file (string): Path to the csv file with text and img name.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_data = df
        self.image_transform = image_transform
        self.tokenizer_bert = tokenizer
        self.MAX_LEN = MAX_LEN

    def __len__(self):
        return self.csv_data.shape[0]
    
    def pre_processing_BERT(self, sent):
        # Create empty lists to store outputs
        input_ids = []
        attention_mask = []
        
        encoded_sent = self.tokenizer_bert.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=self.MAX_LEN,                  # Max length to truncate/pad
            padding='max_length',         # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,      # Return attention mask
            truncation=True
            )
        
        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')
        
        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        return input_ids, attention_mask
    
    def pre_process_images(self, img_list):
        
        final_img_inp = [] ## Stor multiple images 
        
        for img_name in img_list:
            if img_name == "not downloadable":
                continue
            try:
                image = Image.open(img_name).convert("RGB")
            except Exception as e:
#                 print(str(e))
                continue
            image = self.image_transform(image).unsqueeze(0)
#             print(image.size())
            final_img_inp.append(image)
            if len(final_img_inp) == 2:
                break
        
#         print("all loaded")
#         if len(final_img_inp) == 0:
#             print("onono")
        final_img_inp = torch.cat(final_img_inp, dim=0).unsqueeze(0)

        return final_img_inp
     
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_names = self.csv_data['image_status'][idx].split(";")
        
        images = self.pre_process_images(img_names)
        
        text = self.csv_data['text'][idx]
        tensor_input_id, tensor_input_mask = self.pre_processing_BERT(text)

        label = self.csv_data['label'][idx]
        label = torch.tensor(label, dtype=torch.long)

        sample = {'image': images, 'BERT_ip': [tensor_input_id, tensor_input_mask], 'label':label}
#         sample = {'image': images}


        return sample