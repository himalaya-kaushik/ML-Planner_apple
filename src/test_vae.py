import torch
from src.models.vae import HindiVAE

# 1. Setup Dummy Data
batch_size = 2
seq_len = 10
vocab_size = 32000
model = HindiVAE(vocab_size=vocab_size)

# Create fake input (Batch of 2 sentences, 10 words long)
src = torch.randint(0, vocab_size, (batch_size, seq_len))
tgt = torch.randint(0, vocab_size, (batch_size, seq_len))

# 2. Forward Pass
print("Running VAE Forward Pass...")
try:
    # We ignore masks for this simple test
    prediction, mu, logvar = model(src, tgt)
    
    print(f"Input Shape: {src.shape}")
    print(f"Prediction Shape: {prediction.shape}") # Should be [2, 10, 32000]
    print(f"Mu Shape: {mu.shape}")                 # Should be [2, 128]
    print("✅ VAE Architecture is Valid")
except Exception as e:
    print(f"❌ VAE Failed: {e}")