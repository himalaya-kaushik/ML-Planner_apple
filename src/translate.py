import torch
import torch.nn.functional as F
from src.models.vae import HindiVAE
from src.models.diffusion import DiffusionPrior
from tokenizers import Tokenizer

# --- CONFIG ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
VAE_PATH = "checkpoints/vae_epoch_3.pth"
DIFFUSION_PATH = "checkpoints/diffusion_epoch_10.pth"
TOKENIZER_PATH = "data/hindi_tokenizer.json"
MAX_LEN = 64 # Reduced for speed

class Translator:
    def __init__(self):
        print(f" Loading Tokenizer...")
        self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        
        print(f" Loading VAE...")
        self.vae = HindiVAE(vocab_size=32000).to(DEVICE)
        self.vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
        self.vae.eval()
        
        print(f" Loading Diffusion...")
        self.diffusion = DiffusionPrior(vocab_size=32000).to(DEVICE)
        self.diffusion.load_state_dict(torch.load(DIFFUSION_PATH, map_location=DEVICE))
        self.diffusion.eval()
        
        self.num_timesteps = 1000
        self.betas = torch.linspace(0.0001, 0.02, self.num_timesteps).to(DEVICE)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(DEVICE), self.alphas_cumprod[:-1]])
        
    def p_sample(self, model, x, t, english_ids):
        with torch.no_grad():
            noise_pred = model(x, t, english_ids)
        
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t])
        sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alphas[t])
        
        mean = sqrt_recip_alpha_t * (x - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t)
        
        if t == 0:
            return mean
        else:
            posterior_variance_t = beta_t * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def translate(self, english_text):
        clean_en = english_text.lower().strip()
        en_ids = self.tokenizer.encode(clean_en).ids
        en_tensor = torch.tensor([en_ids], dtype=torch.long).to(DEVICE)
        
        latents = torch.randn((1, 128)).to(DEVICE)
        pbar_range = range(self.num_timesteps - 1, -1, -1)
        
        print(f" Dreaming '{english_text}'...")
        for i in pbar_range:
            t_tensor = torch.tensor([i], device=DEVICE)
            latents = self.p_sample(self.diffusion, latents, t_tensor, en_tensor)
            
        print(f"   (Latent Stats: Mean={latents.mean():.4f}, Std={latents.std():.4f})")

        print("  Speaking...")
        memory = self.vae.fc_z_to_decoder(latents).unsqueeze(1)
        
        # Start with a random token instead of 0 to shake things up, 
        # or stick to 0 but rely on temperature. Let's try 0 first.
        curr_tokens = torch.tensor([[0]], device=DEVICE)
        
        generated_ids = []
        for _ in range(MAX_LEN):
            tgt_emb = self.vae.embedding(curr_tokens) * torch.sqrt(torch.tensor(float(self.vae.d_model)))
            tgt_emb = self.vae.pos_encoder(tgt_emb)
            
            out = self.vae.transformer_decoder(tgt_emb, memory)
            logits = self.vae.output_layer(out) 
            next_token_logits = logits[:, -1, :] # [1, vocab]

            temperature = 0.7 
            next_token_logits = next_token_logits / temperature
            
            for id in generated_ids:
                next_token_logits[0, id] /= 1.5 
            
            probs = F.softmax(next_token_logits, dim=-1)
            top_k = 50
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            # Sample from the filtered distribution
            ix = torch.multinomial(top_k_probs, 1) # Pick 1 index from top k
            next_token = top_k_indices[0, ix[0]].item()
            
            if next_token == 0: 
                break
                
            generated_ids.append(next_token)
            curr_tokens = torch.cat([curr_tokens, torch.tensor([[next_token]], device=DEVICE)], dim=1)
            
        return self.tokenizer.decode(generated_ids)

if __name__ == "__main__":
    translator = Translator()
    
    sentences = [
        "The weather is good",
        "Machine learning is complex"
    ]
    
    for s in sentences:
        print("-" * 30)
        try:
            translation = translator.translate(s)
            print(f"🇺🇸 En: {s}")
            print(f"🇮🇳 Hi: {translation}")
        except Exception as e:
            print(f" Error: {e}")