import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class HindiVAE(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2, latent_dim=128):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_var = nn.Linear(d_model, latent_dim)
        
        self.fc_z_to_decoder = nn.Linear(latent_dim, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick:
        z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, tgt_pad_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        enc_output = self.transformer_encoder(src_emb)
        pooled_output = torch.mean(enc_output, dim=1)
        mu = self.fc_mu(pooled_output)
        logvar = self.fc_var(pooled_output)
        z = self.reparameterize(mu, logvar)
        z_projected = self.fc_z_to_decoder(z).unsqueeze(1)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.transformer_decoder(tgt_emb, z_projected, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        prediction = self.output_layer(output)
        
        return prediction, mu, logvar