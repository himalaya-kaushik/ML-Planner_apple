import os
import sys
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize

def train_hindi_tokenizer():
    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer("hi")
    print("Loading Samantar dataset...")
    dataset = load_dataset("cfilt/iitb-english-hindi", split="train", streaming=True)
    
    def batch_iterator(batch_size=1000):
        batch = []
        for i, example in enumerate(dataset):
            hindi_text = example['translation']['hi']
            normalized_text = normalizer.normalize(hindi_text)
            tokenized_text = indic_tokenize.trivial_tokenize(normalized_text)
            batch.append(" ".join(tokenized_text))
            if len(batch) == batch_size:
                yield batch
                batch = []
            
            if i > 200000: 
                break
        if batch:
            yield batch
            
    print("using ByteLevelBPETokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    if not os.path.exists("data"):
        os.makedirs("data")
    tokenizer.save("data/hindi_tokenizer.json")
    print(" Hybrid Indic-BPE Tokenizer Saved!")
    
if __name__ == "__main__":
    train_hindi_tokenizer()