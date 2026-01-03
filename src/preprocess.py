# import torch
# import os
# from datasets import load_dataset
# from tokenizers import Tokenizer
# from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
# from indicnlp.tokenize import indic_tokenize
# from tqdm import tqdm

# # CONFIG
# MAX_LEN = 128
# NUM_SAMPLES = 200000  # We will just take 200k for now to keep it fast
# SAVE_PATH = "data/processed_tensors.pt"
# TOKENIZER_PATH = "data/hindi_tokenizer.json"

# def preprocess():
#     print("⏳ Loading Tokenizer & Normalizer...")
#     tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
#     tokenizer.enable_padding(length=MAX_LEN, pad_id=0, pad_token="[PAD]")
#     tokenizer.enable_truncation(max_length=MAX_LEN)
    
#     factory = IndicNormalizerFactory()
#     normalizer = factory.get_normalizer("hi")
    
#     print("🌊 Loading Dataset Stream...")
#     dataset = load_dataset("cfilt/iitb-english-hindi", split="train", streaming=True)
    
#     input_ids_list = []
    
#     print(f"⚙️ Processing {NUM_SAMPLES} samples...")
#     count = 0
#     for example in tqdm(dataset, total=NUM_SAMPLES):
#         if count >= NUM_SAMPLES:
#             break
            
#         # 1. Get Text
#         tgt_raw = example['translation']['hi']
        
#         # 2. Indic NLP Clean
#         try:
#             norm = normalizer.normalize(tgt_raw)
#             tokens = indic_tokenize.trivial_tokenize(norm)
#             clean_text = " ".join(tokens)
            
#             # 3. BPE Tokenize
#             enc = tokenizer.encode(clean_text)
            
#             # 4. Save ID Only (We don't need masks for now, padding is 0)
#             input_ids_list.append(torch.tensor(enc.ids, dtype=torch.int32)) # int32 saves RAM
#             count += 1
#         except:
#             continue

#     print("📦 Stacking Tensors...")
#     # Stack into one big tensor [N, 128]
#     all_input_ids = torch.stack(input_ids_list)
    
#     print(f"Saving to {SAVE_PATH}...")
#     torch.save(all_input_ids, SAVE_PATH)
#     print(" Done! Data is ready for fast training.")

# if __name__ == "__main__":
#     preprocess()


# src/preprocess.py (Update)
import torch
import os
from datasets import load_dataset
from tokenizers import Tokenizer
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
from tqdm import tqdm

# CONFIG
MAX_LEN = 128
NUM_SAMPLES = 200000 
SAVE_PATH = "data/processed_pairs.pt"  # <--- Changed name
TOKENIZER_PATH = "data/hindi_tokenizer.json"

def preprocess():
    print("⏳ Loading Tokenizer & Normalizer...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.enable_padding(length=MAX_LEN, pad_id=0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=MAX_LEN)
    
    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer("hi")
    
    print("🌊 Loading Dataset Stream...")
    dataset = load_dataset("cfilt/iitb-english-hindi", split="train", streaming=True)
    
    data_pairs = [] # Store tuples (english_ids, hindi_ids)
    
    print(f" Processing {NUM_SAMPLES} pairs...")
    count = 0
    for example in tqdm(dataset, total=NUM_SAMPLES):
        if count >= NUM_SAMPLES:
            break
            
        try:
            # 1. Hindi (Target)
            hi_raw = example['translation']['hi']
            hi_norm = normalizer.normalize(hi_raw)
            hi_tokens = indic_tokenize.trivial_tokenize(hi_norm)
            hi_clean = " ".join(hi_tokens)
            hi_enc = tokenizer.encode(hi_clean).ids
            
            # 2. English (Source)
            # We treat English simpler: just lower case + BPE
            en_raw = example['translation']['en']
            en_clean = en_raw.lower().strip()
            en_enc = tokenizer.encode(en_clean).ids # Using same tokenizer for speed
            
            # Save as Tensor Int32 to save RAM
            data_pairs.append([
                torch.tensor(en_enc, dtype=torch.int32),
                torch.tensor(hi_enc, dtype=torch.int32)
            ])
            count += 1
        except:
            continue

    print("📦 Stacking Tensors...")
    # Separate into two big tensors
    src_tensors = torch.stack([x[0] for x in data_pairs])
    tgt_tensors = torch.stack([x[1] for x in data_pairs])
    
    print(f" Saving to {SAVE_PATH}...")
    torch.save({"src": src_tensors, "tgt": tgt_tensors}, SAVE_PATH)
    print(" Done! English-Hindi Pairs Ready.")

if __name__ == "__main__":
    preprocess()