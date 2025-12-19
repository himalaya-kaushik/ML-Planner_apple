import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionPrior(nn.Module):
    """
    The 'Planner' Model.
    Input: Noisy Latent Vector + English Text + Time Step
    Output: Predicted Noise (which we subtract to get the clean vector)
    """
    def __init__(self, d_model=256, latent_dim=128, num_layers=6, vocab_size=32000):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.english_emb = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.english_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.net = nn.ModuleList([])
        
        for _ in range(num_layers):
            self.net.append(nn.Sequential(
                nn.Linear(latent_dim + d_model + d_model, d_model * 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, latent_dim),
            ))
            
    def forward(self, x_noisy, t, english_ids, english_mask=None):
        """
        x_noisy: The latent vector with noise added
        t: The time step (e.g., 500)
        english_ids: The English sentence tokens
        """
        
        t_emb = self.time_mlp(t) 
        
        eng_emb = self.english_emb(english_ids) * math.sqrt(self.d_model)
        eng_encoded = self.english_encoder(eng_emb) 
        
        eng_context = torch.mean(eng_encoded, dim=1) 
        h = x_noisy
        
        for layer in self.net:
            
            model_input = torch.cat([h, eng_context, t_emb], dim=-1)
            
            h = h + layer(model_input)
            
        return h 