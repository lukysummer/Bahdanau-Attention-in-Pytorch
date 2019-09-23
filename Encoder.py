import torch
from torch import nn


class Encoder(nn.Module):
    
    def __init__(self, n_vocab, n_speaker, n_tags, 
                 n_embed_text, n_embed_speaker, n_embed_tags,
                 n_hidden_enc, n_layers, n_hidden_dec, dropout):
        
        super().__init__()
        
        self.n_layers = n_layers
        self.n_hidden_enc = n_hidden_enc
        
        self.text_embedding = nn.Embedding(n_vocab, n_embed_text)
        self.tag_embedding = nn.Embedding(n_tags, n_embed_tags)
        self.speaker_embedding = nn.Embedding(n_speaker, n_embed_speaker)
        
        self.bdGRU = nn.GRU(n_embed_text + n_embed_speaker + n_embed_tags, 
                            n_hidden_enc, 
                            n_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout)
        
        self.fc = nn.Linear(n_hidden_enc, n_hidden_dec)
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, s, t, h):
        ''' x, s, t: [b, seq_len] '''
        
        embedded = torch.cat((self.text_embedding(x), self.speaker_embedding(s)), dim=2)  
        embedded = torch.cat((embedded, self.tag_embedding(t)), dim=2) #[b, seq_len, n_embed_text + _tags + _speaker]
        
        last_layer_enc, last_h_enc = self.bdGRU(embedded, h)
        
        last_layer_enc = self.dropout(last_layer_enc) #[b, seq_len, 2*n_hidden_enc]
        last_h_enc = self.dropout(last_h_enc.permute(1, 0, 2))         #[b, 2*n_layer, n_hidden_enc]

        # this is to be decoder's 1st hidden state
        for i in range(0, -self.n_layers, -2):
            if i==0:
                last_h_enc_sum = last_h_enc[:, i-1, :] + last_h_enc[:, i-2, :] #[b, n_hidden_enc]
                last_h_enc_sum = last_h_enc_sum.unsqueeze(1)
            else:
                last_h_enc_sum = torch.cat((last_h_enc_sum, 
                                            (last_h_enc[:, i-1, :] + last_h_enc[:, i-2, :]).unsqueeze(1)), dim=1)
        

        last_h_enc_sum = self.fc(last_h_enc_sum) #[b, n_layers, n_hidden_dec]

        return last_layer_enc, last_h_enc_sum   #[b, seq_len, 2*n_hidden_enc], [b, n_layer, n_hidden_enc]
        
    
    def init_hidden(self, batch_size):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight = next(self.parameters()).data
    
        h = weight.new(2*self.n_layers, batch_size, self.n_hidden_enc).zero_().to(device)
        
        return h
