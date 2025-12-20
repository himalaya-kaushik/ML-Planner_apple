import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

from src.models.vae import HindiVAE
from src.models.diffusion import DiffusionPrior

# CONFIG
BATCH_SIZE = 64 
LR = 3e-4
EPOCHS = 10
MAX_LEN = 128
LATENT_DIM = 128
VOCAB_SIZE = 32000
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
VAE_CHECKPOINT = "checkpoints/vae_epoch_3.pth" # We will need this later

class TranslationDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.src = data['src'] 
        self.tgt = data['tgt']
        
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return {
            "english": self.src[idx].long(),
            "hindi": self.tgt[idx].long()
        }

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=DEVICE):
        self.num_timesteps = num_timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # \bar{\alpha}
        
    def add_noise(self, x_start, t):
        """
        Forward Process: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon
        """
        noise = torch.randn_like(x_start)
        
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])[:, None]
        
        x_noisy = sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
        return x_noisy, noise

def train():
    print(f" Training Diffusion on {DEVICE}...")
    
    # Load Data
    if not os.path.exists("data/processed_pairs.pt"):
        raise FileNotFoundError("Run preprocess.py first!")
    ds = TranslationDataset("data/processed_pairs.pt")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # Load VAE (Teacher) - MUST BE PRE-TRAINED
    print("🧊 Loading VAE Teacher...")
    vae = HindiVAE(vocab_size=VOCAB_SIZE).to(DEVICE)
    try:
        vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=DEVICE))
        print(" VAE Loaded!")
    except:
        print(" WARNING: VAE Checkpoint not found. Model will output garbage latents.")
        print(" (This is fine for testing code logic, but bad for actual training)")
    
    vae.eval() # Freeze VAE
    for p in vae.parameters():
        p.requires_grad = False
        
    diffusion = DiffusionPrior(vocab_size=VOCAB_SIZE).to(DEVICE)
    optimizer = optim.AdamW(diffusion.parameters(), lr=LR)
    scheduler = NoiseScheduler(device=DEVICE)
    
    diffusion.train()
    
    for epoch in range(EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            english = batch['english'].to(DEVICE)
            hindi = batch['hindi'].to(DEVICE)
            
            with torch.no_grad():
                _, mu, _ = vae(hindi, hindi) 
                x_0 = mu.detach() 
                
            t = torch.randint(0, scheduler.num_timesteps, (english.shape[0],), device=DEVICE)
            
            x_noisy, noise = scheduler.add_noise(x_0, t)
            
            noise_pred = diffusion(x_noisy, t, english)
            
            loss = nn.MSELoss()(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())
            
        torch.save(diffusion.state_dict(), f"checkpoints/diffusion_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()