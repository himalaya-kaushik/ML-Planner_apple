import torch
from src.models.diffusion import DiffusionPrior

# Setup Dummy Data
batch_size = 4
latent_dim = 128
d_model = 256
vocab_size = 32000

model = DiffusionPrior(d_model, latent_dim, vocab_size=vocab_size)

# Fake Inputs
noisy_vector = torch.randn(batch_size, latent_dim)  # A random vector
time_step = torch.tensor([10, 50, 100, 900])        # Random time steps
english_text = torch.randint(0, vocab_size, (batch_size, 20)) # Random English sentence

print("Running Diffusion Forward Pass...")
try:
    noise_pred = model(noisy_vector, time_step, english_text)
    print(f"Output Shape: {noise_pred.shape}") # Should be [4, 128]
    print("✅ Diffusion Model works!")
except Exception as e:
    print(f"❌ Failed: {e}")