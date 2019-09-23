import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    
    def __init__(self, n_hidden_enc, n_hidden_dec):
        super().__init__()
        
        self.h_hidden_enc = n_hidden_enc
        self.h_hidden_dec = n_hidden_dec
        
        self.W = nn.Linear(2*n_hidden_enc + n_hidden_dec, n_hidden_dec, bias=False) 
        self.V = nn.Parameter(torch.rand(n_hidden_dec))
        #self.V = nn.Linear(n_hidden_dec, 1, bias=False)
        
    
    def forward(self, hidden_dec, last_layer_enc):
        ''' hidden_dec:     [b, n_layers, n_hidden_dec] 
                            (1st hidden_dec = encoder's last_h's last layer)
                            
            last_layer_enc: [b, seq_len, n_hidden_enc * 2] 
        '''
        
        batch_size = last_layer_enc.size(0)
        src_seq_len = last_layer_enc.size(1)

        #if t==0:
            # hidden_dec = [b, n_hidden_dec]
            #hidden_dec = hidden_dec.unsqueeze(1).repeat(1, src_seq_len, 1)
            # Now, [b, seq_len, n_hidden_dec]
        #else:
            # hidden_dec = [b, n_layers, n_hidden_dec]
        #hidden_dec = hidden_dec.permute(1, 0, 2)    #[n_layers, b, n_hidden_dec]
        hidden_dec = hidden_dec[:, -1, :].unsqueeze(1).repeat(1, src_seq_len, 1)  #[b, seq_len, n_hidden_dec]

        #s_h =  #[b, seq_len, 2*n_hidden_enc + n_hidden_dec]
        tanh_W_s_h = torch.tanh(self.W(torch.cat((hidden_dec, last_layer_enc), dim=2)))   #[b, seq_len, n_hidden_dec]
        #tanh_W_s_h = torch.tanh(W_s_h)   #[b, seq_len, n_hidden_dec]
        tanh_W_s_h = tanh_W_s_h.permute(0, 2, 1)     #[b, n_hidde_dec, seq_len]
        
        #e = self.V(W_s_h).squeeze(2)  #[b, seq_len, 1]
        
        #self.V = nn.Parameter(torch.rand(n_hidden_dec))
        V = self.V.repeat(batch_size, 1).unsqueeze(1)  #[b, 1, n_hidden_dec]
        e = torch.bmm(V, tanh_W_s_h).squeeze(1) #[b, seq_len]
        
        att_weights = F.softmax(e, dim=1) #[b, seq_len]
        
        return att_weights
    



class Attn(nn.Module):
    
    def __init__(self, n_hidden_enc, n_hidden_dec):
        super().__init__()
        
        self.W = nn.Linear(self.n_hidden_enc*2, n_hidden_dec)
        self.V = nn.Parameter(torch.rand(1, n_hidden_dec))
        
    
    def forward(self, hidden_dec, last_layer_enc):  
        ''' hidden_dec:     [b, n_layers, n_hidden_dec] 
                            (1st hidden_dec = encoder's last_h's last layer)
                            
            last_layer_enc: [b, seq_len, n_hidden_enc * 2] 
        '''
        
        seq_len_enc = last_layer_enc.size(1)
        
        e = Variable(torch.zeros(seq_len_enc)).cuda()
        
        for t in range(seq_len_enc):
            W_s_h = self.W(torch.cat((hidden_dec, last_layer_enc[t]), dim=1)) #[b, n_layers, n_hidden_dec] 
            e[t] = torch.bmm(self.V, torch.tanh(W_s_h))
            e[t] = self.score(hidden_dec, last_layer_enc[t])
        
        
        
        
        
        
        
        
        
        ### 1. Project the query (decoder state)
        query = self.query_layer(query)
        
        scores = self.energy_layer(torch.tanh(query +  proj_key))
        
        batch_size = last_layer_enc.size(0)
        src_seq_len = last_layer_enc.size(1)

        #if t==0:
            # hidden_dec = [b, n_hidden_dec]
            #hidden_dec = hidden_dec.unsqueeze(1).repeat(1, src_seq_len, 1)
            # Now, [b, seq_len, n_hidden_dec]
        #else:
            # hidden_dec = [b, n_layers, n_hidden_dec]
        #hidden_dec = hidden_dec.permute(1, 0, 2)    #[n_layers, b, n_hidden_dec]
        hidden_dec = hidden_dec[:, -1, :]                               #[b, n_hidden_dec]
        hidden_dec = hidden_dec.unsqueeze(1).repeat(1, src_seq_len, 1)  #[b, seq_len, n_hidden_dec]

        s_h = torch.cat((hidden_dec, last_layer_enc), dim=2) #[b, seq_len, 2*n_hidden_enc + n_hidden_dec]
        W_s_h = self.W(s_h)          #[b, seq_len, n_hidden_dec]
        tanh_W_s_h = torch.tanh(W_s_h)   #[b, seq_len, n_hidden_dec]
        tanh_W_s_h = tanh_W_s_h.permute(0, 2, 1)     #[b, n_hidde_dec, seq_len]
        
        V = self.V.repeat(batch_size, 1).unsqueeze(1)  #[b, 1, n_hidden_dec]
        e = torch.bmm(V, tanh_W_s_h) #[b, 1, seq_len]
        e = e.squeeze(1)             #[b, seq_len]
        
        att_weights = F.softmax(e, dim=1) #[b, seq_len]
        
        return att_weights
        