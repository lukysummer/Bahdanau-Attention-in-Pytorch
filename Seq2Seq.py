import torch
from torch import nn
import torch.nn.functional as F
import random

from Encoder import Encoder
from Decoder import Decoder


class Seq2Seq(nn.Module):
    
    def __init__(self, n_vocab, n_embed_enc, n_embed_dec, n_hidden_enc, n_hidden_dec, n_layers, n_output, dropout):
        
        super().__init__()
        
        self.encoder = Encoder(n_vocab=n_vocab, 
                               n_embed=n_embed_enc, 
                               n_hidden_enc=n_hidden_enc, 
                               n_layers=n_layers, 
                               n_hidden_dec=n_hidden_dec, 
                               dropout=dropout)
        
        self.decoder = Decoder(n_output=n_output, 
                               n_embed=n_embed_dec, 
                               n_hidden_enc=n_hidden_enc, 
                               n_hidden_dec=n_hidden_dec, 
                               n_layers=n_layers, 
                               dropout=dropout)
        
        
    def forward(self, inputs, speakers, tags, targets, tf_ratio=0.5):
        ''' inputs:  [b, input_seq_len(200)]
            targets: [b, input_seq_len(200)]'''
            
        ###########################  1. ENCODER  ##############################
        h = self.encoder.init_hidden(inputs.size(0))
        #h = tuple([each.data for each in h])
        last_layer_enc, last_h_enc = self.encoder(inputs, speakers, tags, h)
        
        trg_seq_len = targets.size(1)
        
            
        ###########################  2. DECODER  ##############################
        hidden_dec = last_h_enc       #[b, n_layers, n_hidden_dec]       
        
        b = inputs.size(0)
        n_output = self.decoder.n_output
        output = targets[:, 0]
        
        outputs = torch.zeros(b, n_output, trg_seq_len).cuda()
        
        for t in range(1, trg_seq_len, 1):
            output, hidden_dec = self.decoder(output, hidden_dec, last_layer_enc)
            outputs[:, :, t] = output #output: [b, n_output]
            #print(t)
            if random.random() < tf_ratio:
                output = targets[:, t]
                #output, hidden_dec = self.decoder(targets[:, t], hidden_dec, last_layer_enc) #targets[:, t] : [b]
                
            else:
                output = output.max(dim=1)[1]
                #output, hidden_dec = self.decoder(output.max(dim=1)[1], hidden_dec, last_layer_enc)
        
        return outputs  #[b, n_output, trg_seq_len]
    