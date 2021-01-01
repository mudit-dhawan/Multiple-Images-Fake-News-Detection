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
from sklearn.metrics import *

def train(model, loss_fn_fnd, loss_fn_sim, pdist, hinge_loss, optimizer, scheduler, train_dataloader, val_dataloader=None, epochs=4, evaluation=False, device='cpu', param_dict_model=None, param_dict_opt=None, save_best=False, file_path='./saved_models/best_model.pt', writer=None):
    """Train the BertClassifier model.
    """
    # Start training loop
    best_acc_val = 0
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            img_ip , text_ip, label = batch["image"], batch["BERT_ip"], batch['label']
            
            b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)
            
            imgs_ip = img_ip.to(device)
            
            b_labels = label.to(device)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits, latent_vectors = model(text=[b_input_ids, b_attn_mask], image=imgs_ip, label=b_labels)

            # Compute loss and accumulate the loss values
            loss_fnd = loss_fn_fnd(logits, b_labels)
            loss_sim = loss_fn_sim(latent_vectors, b_labels, pdist, hinge_loss)
            
            loss = loss_fnd + loss_sim
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()
            
#             break 
            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                
                ## Write onto tensorboard
                if writer != None:
                    writer.add_scalar('Training Loss', (batch_loss / batch_counts), epoch_i*len(train_dataloader)+step)
                
                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, loss_fn_fnd, loss_fn_sim, pdist, hinge_loss, val_dataloader, device)            
            
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
            
            ## Write onto tensorboard
            if writer != None:
                writer.add_scalar('Validation Loss', val_loss, epoch_i+1)
                writer.add_scalar('Validation Accuracy', val_accuracy, epoch_i+1)
            
            # Save the best model
            if save_best: 
                if val_accuracy > best_acc_val:
                    best_acc_val = val_accuracy
                    torch.save({
                                'epoch': epoch_i+1,
                                'model_params': param_dict_model,
                                'opt_params': param_dict_opt,
                                'model_state_dict': model.state_dict(),
                                'opt_state_dict': optimizer.state_dict(),
                                'sch_state_dict': scheduler.state_dict()
                               }, file_path)
                    
        print("\n")
    
    print("Training complete!")
    
    
    
def evaluate(model, loss_fn_fnd, loss_fn_sim, pdist, hinge_loss, val_dataloader, device):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    
    y_pred = []
    y_true = [] 
    
    # For each batch in our validation set...
    for batch in val_dataloader:
        img_ip , text_ip, label = batch["image"], batch["BERT_ip"], batch['label']
            
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)

        imgs_ip = img_ip.to(device)

        b_labels = label.to(device)

        # Compute logits
        with torch.no_grad():
            logits, latent_vectors = model(text=[b_input_ids, b_attn_mask], image=imgs_ip, label=b_labels)

        # Compute loss
        loss_fnd = loss_fn_fnd(logits, b_labels)
        loss_sim = loss_fn_sim(latent_vectors, b_labels, pdist, hinge_loss)
        loss = loss_fnd + loss_sim
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
        
        y_pred.append(preds.squeeze().cpu().numpy())
        y_true.append(label.numpy())

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("TN: {} | FP: {} | FN: {} | TP: {} ".format(tn, fp, fn, tp))

    if tp==0 : 
        prec = 0
        rec = 0
        f1_score = 0
    else: 
        ## calculate the Precision
        prec = (tp/ (tp+fp))*100

        ## calculate the Recall
        rec = (tp/ (tp + fn))*100
        
        ## calculate the F1-score
        f1_score = 2*prec*rec/(prec+rec)

    print("Precision: {} | Recall: {} | F1-score: {}".format(prec, rec, f1_score))

    return val_loss, val_accuracy