import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from transformers import BertModel

# Create the Bert custom class 
class Text_Encoder(nn.Module):
    """Bert Model for Encoding
    """
    def __init__(self, text_fc1_out, latent_dim, freeze_bert=False, dropout_p=0.20):
        """
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(Text_Encoder, self).__init__()
        
        self.text_fc1_out = text_fc1_out
        self.latent_dim = latent_dim
        self.freeze_bert = freeze_bert
        
        self.dropout = nn.Dropout(dropout_p)


        ## For a fc layer to convert encoded dim into latent dim 
#         self.latent_dim = latent_dim
       
        D_in = 768 ## Output dim for BERT Model 

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(
                    'bert-base-uncased',
#                     output_attentions = True, 
                    return_dict=True)


        # Instantiate an one-layer feed-forward to convert BERT output into latent space 
        self.fc1 = nn.Sequential(
            nn.Linear(D_in, self.text_fc1_out),
            nn.ReLU()
        )
        
        self.latent = nn.Sequential(
            nn.Linear(self.text_fc1_out, self.latent_dim),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(dropout_p)

        self.fine_tune()
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        ## odict_keys(['last_hidden_state', 'pooler_output', 'attentions'])
        ## last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) 

        out_cls = self.dropout(out.last_hidden_state[:, 0, :])

        out_emb = self.fc1(out_cls)
        
        out_final = self.dropout(self.latent(self.dropout(out_emb)))

        return out_final, out_emb     
    
    def fine_tune(self):
        """
        keep the weights fixed or not  
        """
        for p in self.bert.parameters():
            p.requires_grad = self.freeze_bert
            

class VisualCNN(nn.Module):
    def __init__(self, enc_img_dim, img_fc1_out, img_latent_dim, fine_tune_vgg, dropout_p=0.20):
        super(VisualCNN, self).__init__()
        
        vgg = models.vgg19(pretrained=True)
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])
        
        self.vis_encoder = vgg
        
        self.dropout = nn.Dropout(dropout_p)

        self.vis_enc_fc1 = nn.Sequential(
            nn.Linear(enc_img_dim, img_fc1_out),
            nn.ReLU()
        )

        self.vis_enc_fc2 = nn.Sequential(
            nn.Linear(img_fc1_out, img_latent_dim),
            nn.ReLU()
        )
        
        self.fine_tune(fine_tune_vgg)
    
    def forward(self, x):

        x_cnn = self.dropout(self.vis_encoder(x))

        x = self.dropout(self.vis_enc_fc1(x_cnn))

        x = self.dropout(self.vis_enc_fc2(x))

        return x
    
    def fine_tune(self, fine_tune=True):
        """
        
        :param fine_tune: Allow?
        """
        for p in self.vis_encoder.parameters():
            p.requires_grad = False
            
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.vis_encoder.children())[-3:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        ## x.size() -- (batch_size, nb_images or time_steps , H, W, C)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3), x.size(-2), x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class MultiVisualEncoder(nn.Module):
    def __init__(self, single_enc_img_dim, single_img_fc1_out, single_img_latent_dim, hidden_size, num_layers, multiple_enc_img_dim, bidirectional, fine_tune_vgg, dropout_p):
        super(MultiVisualEncoder, self).__init__()

        self.single_img_latent_dim = single_img_latent_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        ## Extract visual features from 1 image
        self.visual_cnn = VisualCNN(single_enc_img_dim, single_img_fc1_out, self.single_img_latent_dim, fine_tune_vgg, dropout_p)

        ## Extract from multiple images 
        self.time_distributed_cnn = TimeDistributed(self.visual_cnn, batch_first=True)

        ## Merge the features from multiple images 
        self.visual_rnn = nn.LSTM(self.single_img_latent_dim, hidden_size, num_layers=num_layers, bidirectional=bidirectional, 
                                    batch_first=True)

        ## Combined Latent representation of multiple representations  
        self.mult_vis_enc_fc1 = nn.Sequential(
            nn.Linear(self.hidden_size*self.hidden_factor, multiple_enc_img_dim),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):

        batch_size = x.size(0)

        emb_x = self.time_distributed_cnn(x) # (samples, timesteps, single_img_latent_dim)

#         x = self.visual_rnn(x) 

        _, (hidden, hidden_) = self.visual_rnn(emb_x)
        
#         print(hidden.size())

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()
        
        x = self.dropout(hidden)
        x = self.dropout(self.mult_vis_enc_fc1(x))
        
        return x, emb_x

class SimilarityModule(nn.Module):
    def __init__(self, vis_emb_dim, text_emb_dim, latent_dim):
        super(SimilarityModule, self).__init__()

        self.vis_latent_space = nn.Linear(vis_emb_dim, latent_dim)

        ## multiple images to multimodal space  
        self.vis_latent_vec = TimeDistributed(self.vis_latent_space, batch_first=True)

        self.text_latent_vec = nn.Linear(text_emb_dim, latent_dim)
    
    def forward(self, x_text, x_vis):

        x_vis = self.vis_latent_vec(x_vis)

        x_text = self.text_latent_vec(x_text)


        x_latent_vec = torch.cat(
            [x_text.unsqueeze(1), x_vis], dim=1
        )

        x_latent_vec = F.normalize(x_latent_vec, dim=1)

        return x_latent_vec

class Multiple_Images_Model(nn.Module):
    def __init__(self, parameter_dict_model):
        super(Multiple_Images_Model, self).__init__()

        self.text_encoder = Text_Encoder(parameter_dict_model['text_fc1_out'], parameter_dict_model['latent_text'], parameter_dict_model['freeze_bert'], parameter_dict_model['dropout_p'])

        self.visual_encoder = MultiVisualEncoder(parameter_dict_model['single_enc_img_dim'], parameter_dict_model['single_img_fc1_out'], parameter_dict_model['single_img_latent_dim'], parameter_dict_model['hidden_size'], parameter_dict_model['num_layers'], parameter_dict_model['multiple_enc_img_dim'], parameter_dict_model['bidirectional'], parameter_dict_model['fine_tune_vgg'], parameter_dict_model['dropout_p'])

        self.sim_module = SimilarityModule(parameter_dict_model['single_img_latent_dim'], parameter_dict_model['text_fc1_out'], parameter_dict_model['multimodal_latent_dim'])
        self.fusion = torch.nn.Linear(
            in_features=(parameter_dict_model['multiple_enc_img_dim'] + parameter_dict_model['latent_text']), 
            out_features=parameter_dict_model['latent_fused']
        )
        

        self.fc = torch.nn.Linear(
            in_features=parameter_dict_model['latent_fused'], 
            out_features=parameter_dict_model['num_classes']
        )

        self.dropout = nn.Dropout(parameter_dict_model['dropout_p'])
    
    def forward(self, text, image, label=None):
        text_features, emb_text = self.text_encoder(text[0], text[1])
#         print(text_features.size())

        imgs_feature, emb_imgs = self.visual_encoder(image)
#         print(imgs_feature.size())

        sim_vec = self.sim_module(emb_text, emb_imgs)

        combined = torch.cat(
            [text_features, imgs_feature], dim=1
        )

        fused = self.dropout(
            F.relu(
            self.fusion(combined)
            )
        )

        logits = self.fc(fused)

        return logits , sim_vec