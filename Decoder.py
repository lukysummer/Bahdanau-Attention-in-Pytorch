import torch
from torch import nn
import torch.nn.functional as F
from .Attention import Attention


class Decoder(nn.Module):
    
    def __init__(self, n_output, n_embed, n_hidden_enc, n_hidden_dec, n_layers, dropout):
        super().__init__()

        self.n_output = n_output
        self.embedding = nn.Embedding(n_output, n_embed)
        
        self.GRU = nn.GRU(n_embed + n_hidden_enc*2, n_hidden_dec, n_layers,
                          bidirectional=False, batch_first=True,
                          dropout=dropout) 
        
        self.attention = Attention(n_hidden_enc, n_hidden_dec)
        
        self.fc_final = nn.Linear(n_embed + n_hidden_enc*2 + n_hidden_dec, n_output)
        
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, target, hidden_dec, last_layer_enc):
        ''' 
            target:         [b] 
            hidden_dec:     [b, n_layers, n_hidden_dec]      (1st hidden_dec = encoder's last_h's last layer)
            last_layer_enc: [b, seq_len, n_hidden_enc * 2]
        '''
        
        ######################## 1. TARGET EMBEDDINGS ######################### 
        target = target.unsqueeze(1) #[b, 1] : since n_output = 1
        embedded_trg = self.embedding(target)  #[b, 1, n_embed]


        ################## 2. CALCULATE ATTENTION WEIGHTS #####################      
        att_weights = self.attention(hidden_dec, last_layer_enc)  #[b, seq_len]
        att_weights = att_weights.unsqueeze(1)   #[b, 1, seq_len]


        ###################### 3. CALCULATE WEIGHTED SUM ######################
        weighted_sum = torch.bmm(att_weights, last_layer_enc) #[b, 1, n_hidden_enc*2]
        
        
        ############################# 4. GRU LAYER ############################
        gru_input = torch.cat((embedded_trg, weighted_sum), dim=2) #[b, 1, n_embed + n_hidden_enc*2]

        last_layer_dec, last_h_dec = self.GRU(gru_input, hidden_dec.permute(1, 0, 2))
        # last_layer_dec: [b, trg_seq_len, n_hidden_dec]
        last_h_dec = last_h_dec.permute(1, 0, 2)  #[b, n_layers, n_hidden_dec]
        
        
        ########################### 5. FINAL FC LAYER #########################
        fc_in = torch.cat((embedded_trg.squeeze(1),           #[b, n_embed]
                           weighted_sum.squeeze(1),           #[b, n_hidden_enc*2]
                           last_layer_dec.squeeze(1)), dim=1) #[b, n_hidden_dec]                           
                           
       
        output = self.fc_final(fc_in) #[b, n_output]
        
        return output, last_h_dec, att_weights
        