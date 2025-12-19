# import torch
# from torch.utils.data import IterableDataset
# from datasets import load_dataset
# from tokenizers import Tokenizer
# from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
# from indicnlp.tokenize import indic_tokenize

# class HindiStreamDataset(IterableDataset):
#     def __init__(self, tokenizer_path, max_length=128):
#         self.tokenizer = Tokenizer.from_file(tokenizer_path)
#         self.tokenizer.enable_padding(length=max_length, pad_id=0, pad_token="[PAD]")
#         self.tokenizer.enable_truncation(max_length=max_length)
        
#         factory = IndicNormalizerFactory()
#         self.normalizer = factory.get_normalizer("hi")
        
#         self.dataset = load_dataset("cfilt/iitb-english-hindi", split="train", streaming=True)

#     def process_text(self, text):
#         norm = self.normalizer.normalize(text)
#         tokens = indic_tokenize.trivial_tokenize(norm)
#         return " ".join(tokens)

#     def __iter__(self):
#         for example in self.dataset:
#             src_txt = example['translation']['en'] # You can add English cleaning here later
            
#             tgt_raw = example['translation']['hi']
#             tgt_clean = self.process_text(tgt_raw)
#             tgt_enc = self.tokenizer.encode(tgt_clean)
            
#             yield {
#                 "input_ids": torch.tensor(tgt_enc.ids, dtype=torch.long),
#                 "attention_mask": torch.tensor(tgt_enc.attention_mask, dtype=torch.long)
#             }

# if __name__ == "__main__":
#     ds = HindiStreamDataset("data/hindi_tokenizer.json")
#     loader = torch.utils.data.DataLoader(ds, batch_size=2)
#     print("Testing pipeline...")
#     batch = next(iter(loader))
#     print(f"Shape: {batch['input_ids'].shape}")
#     print("Pipeline Ready")


import torch
from torch.utils.data import Dataset

class HindiFastDataset(Dataset):
    def __init__(self, tensor_path):
        """
        Loads pre-processed tensors directly into RAM.
        """
        print(f"📂 Loading data from {tensor_path}...")
        self.data = torch.load(tensor_path)
        print(f"✅ Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data is already padded and tokenized!
        # Just return it as Long (int64) which PyTorch expects
        return {
            "input_ids": self.data[idx].long() 
        }