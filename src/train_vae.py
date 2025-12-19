import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from contextlib import nullcontext

from src.models.vae import HindiVAE
# from src.dataset import HindiStreamDataset
from src.dataset import HindiFastDataset

BATCH_SIZE = 32        
LR = 1e-4              
EPOCHS = 3    
ACCUM_STEPS = 4        
MAX_LEN = 128
VOCAB_SIZE = 32000
SAVE_DIR = "checkpoints"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(" Using Apple MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(" Using NVIDIA CUDA")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

os.makedirs(SAVE_DIR, exist_ok=True)

def loss_function(recon_logits, target, mu, logvar, step, total_steps, cycle_len=2000):
    recon_loss = nn.CrossEntropyLoss(ignore_index=0)(
        recon_logits.view(-1, VOCAB_SIZE), 
        target.view(-1)
    )

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / target.size(0)

    rel_step = step % cycle_len
    beta = min(1.0, rel_step / (cycle_len * 0.5))
    
    total_loss = recon_loss + (beta * kl_loss)
    
    return total_loss, recon_loss.item(), kl_loss.item(), beta

def train():
    print(f" Training on {DEVICE}...")
    
    # 1. Load Data (Preprocessed)
    if os.path.exists("data/processed_tensors.pt"):
        dataset = HindiFastDataset("data/processed_tensors.pt")
    else:
        raise FileNotFoundError("Run src/preprocess.py first!")
    
    # FIX: pin_memory=False for MPS to stop warning
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=False)
    
    # 2. Init Model
    model = HindiVAE(vocab_size=VOCAB_SIZE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # Mixed Precision Setup
    use_amp = (DEVICE.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    global_step = 0
    model.train()
    
    # Initialize gradients once
    optimizer.zero_grad()
    
    for epoch in range(EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(loop):
            src = batch['input_ids'].to(DEVICE)
            tgt_input = src 
            tgt_label = src
            
            # Context Manager
            ctx = torch.cuda.amp.autocast() if use_amp else nullcontext()
            
            with ctx:
                recon_logits, mu, logvar = model(src, tgt_input)
                loss, recon, kl, beta = loss_function(
                    recon_logits, tgt_label, mu, logvar, global_step, 0
                )
                
                # NORMALIZE LOSS because we are accumulating
                loss = loss / ACCUM_STEPS 

            # Backward Pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # --- GRADIENT ACCUMULATION STEP ---
            # Only update weights every 4 steps
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Clear gradients for next accumulation cycle
                optimizer.zero_grad()
                global_step += 1
            
            # Multiply loss back by ACCUM_STEPS just for logging purposes
            epoch_loss += (loss.item() * ACCUM_STEPS)
            
            # Log
            loop.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}", beta=f"{beta:.2f}")
            
            if global_step % 1000 == 0:
                torch.save(model.state_dict(), f"{SAVE_DIR}/vae_step_{global_step}.pth")

        print(f"✅ Epoch {epoch+1} Completed.")
        torch.save(model.state_dict(), f"{SAVE_DIR}/vae_epoch_{epoch+1}.pth")
    print(f" Training on {DEVICE}...")
    
    # dataset = HindiStreamDataset("data/hindi_tokenizer.json", max_length=MAX_LEN)
    if os.path.exists("data/processed_tensors.pt"):
        dataset = HindiFastDataset("data/processed_tensors.pt")
    else:
        raise FileNotFoundError("Run src/preprocess.py first!")
        
    # loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2,       # Uses 2 CPU cores to load data in background
    pin_memory=True,     # Faster transfer to GPU/MPS
    prefetch_factor=2    # Pre-loads 2 batches ahead of time
)
    
    model = HindiVAE(vocab_size=VOCAB_SIZE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    use_amp = (DEVICE.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    global_step = 0
    model.train()
    
    for epoch in range(EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(loop):
            src = batch['input_ids'].to(DEVICE)
            
            tgt_input = src 
            tgt_label = src
            ctx = torch.cuda.amp.autocast() if use_amp else nullcontext()
            
            with ctx:
                recon_logits, mu, logvar = model(src, tgt_input)
                loss, recon, kl, beta = loss_function(
                    recon_logits, tgt_label, mu, logvar, global_step, 0
                )
            optimizer.zero_grad()
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            global_step += 1
            epoch_loss += loss.item()
            
            loop.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon:.2f}", kl=f"{kl:.2f}", beta=f"{beta:.2f}")
            
            if global_step % 1000 == 0:
                torch.save(model.state_dict(), f"{SAVE_DIR}/vae_step_{global_step}.pth")

        # End of Epoch
        print(f" Epoch {epoch+1} Completed. Avg Loss: {epoch_loss / len(loop):.4f}")
        torch.save(model.state_dict(), f"{SAVE_DIR}/vae_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()