import torch
import pandas as pd
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
import os
import re
import math
from torch.utils.tensorboard import SummaryWriter

from mult_img_data_loader import *
from multiple_imgs_model import *
from train_val import *

from utils import *

csv_name = "/home/muditd/FakeNewsDetection/datasets/FakeNewsNet/Final_dataset/PoltiFact_data.csv"

df = clean_data(csv_name)

## Split data 
msk = np.random.rand(len(df)) < 0.8
df_train = df[msk].reset_index(drop=True) 
df_test = df[~msk].reset_index(drop=True)


if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# define a callable image_transform with Compose
image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
# Specify `MAX_LEN`
MAX_LEN = 500
    
        
# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

transformed_dataset_train = FakeNewsDataset(df_train, image_transform, tokenizer, MAX_LEN)
train_dataloader = DataLoader(transformed_dataset_train, batch_size=4,
                        shuffle=True, num_workers=0)

transformed_dataset_val = FakeNewsDataset(df_test, image_transform, tokenizer, MAX_LEN)
val_dataloader = DataLoader(transformed_dataset_val, batch_size=4,
                        shuffle=True, num_workers=0)

# Specify loss function
loss_fn = nn.CrossEntropyLoss() ## Fake news detects loss (Sub task 1)

pdist = nn.PairwiseDistance(p=2)
hinge_loss = nn.HingeEmbeddingLoss(margin=1, reduction='none')

def sim_loss(x, y, pdist, hinge_loss):

    centroid = torch.mean(x, dim=1, keepdim=True)

    centroid = centroid.repeat(1, x.size(1), 1)

    y = y.unsqueeze(1).repeat(1, x.size(1))

    batch_size = x.shape[0]

    # Squash samples and timesteps into a single axis
    x_reshape = x.contiguous().view(-1, x.size(-1))  # (b_s * num_components, latent_dim)

    centroid_reshape = centroid.contiguous().view(-1, centroid.size(-1))  # (b_s * num_components, latent_dim)


    dist_mat = pdist(x_reshape, centroid_reshape).view(batch_size, -1)

    total_loss = torch.mean(torch.mean(hinge_loss(dist_mat, (y-0.5)*-2), dim=1))

    # dist_mat = torch.cdist(latent_tensors, latent_tensors, p=p)
    # idx = torch.nonzero(torch.triu(dist_mat) , as_tuple=True) 
    # clust_dists = dist_mat[idx[0], idx[1], idx[2]].view(batch_size, -1)
    # clust_loss = torch.var(clust_dists, -1, keepdim=True)
    # total_loss = torch.mean(clust_loss)

    return total_loss 



def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
parameter_dict_model={
                'single_enc_img_dim': 4096,
                'single_img_fc1_out': 1024,
                'single_img_latent_dim': 256,
                'hidden_size': 256,
                'num_layers': 1,
                'multiple_enc_img_dim': 128,
                'bidirectional': True,
                'text_fc1_out': 512,
                'latent_text': 128,
                'fine_tune_vgg': True,
                'freeze_bert': True,
                'multimodal_latent_dim': 128,
                'nb_classes': 2,
                'latent_fused': 128,
                'dropout_p': 0.40,
                'num_classes': 2
                }

parameter_dict_opt={'l_r': 1e-5,
                    'eps': 1e-8
                    }


EPOCHS = 30

set_seed(42)  

final_model = Multiple_Images_Model(parameter_dict_model)
final_model = final_model.to(device)

# Create the optimizer
optimizer = AdamW(final_model.parameters(),
                  lr=parameter_dict_opt['l_r'],
                  eps=parameter_dict_opt['eps'])

# Total number of training steps
total_steps = len(train_dataloader) * EPOCHS

# Set up the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, # Default value
                                            num_training_steps=total_steps)

## Instantiate the tensorboard summary writer
writer = SummaryWriter('runs/exp_margin_1_normalize_1')

train(model=final_model, loss_fn_fnd=loss_fn, loss_fn_sim=sim_loss, pdist=pdist, hinge_loss=hinge_loss, optimizer=optimizer, scheduler=scheduler, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=EPOCHS, evaluation=True, device=device, param_dict_model=parameter_dict_model, param_dict_opt=parameter_dict_opt, save_best=True, file_path='./saved_models/best_model_margin_1_normalize_1.pt', writer=writer)