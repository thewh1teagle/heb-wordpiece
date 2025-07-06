"""
wget https://huggingface.co/datasets/thewh1teagle/phonikud-data/resolve/main/knesset_nikud_v6.txt.7z
7z x knesset_nikud_v6.txt.7z
uv run main.py --input-path ./knesset_nikud_v6.txt 
"""

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers, processors
import json
import argparse


def train_tokenizer(input_path, tokenizer_path, vocab_size=512):
    # Define special tokens here
    # Note: WordPiece may automatically add [BLANK] for alignment tasks
    special_tokens = [
        "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BLANK]"
    ]

    # Use WordPiece model (like BERT)
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    
    # Use NFD normalization to separate base characters from diacritics
    # This allows the tokenizer to learn patterns with/without niqqud separately
    tokenizer.normalizer = normalizers.NFD()
    
    # Use character-level splitting for TTS
    # Each character/diacritic should be a separate token for phonetic alignment
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")

    # Use WordPiece trainer with larger vocab for Hebrew + niqqud
    trainer = trainers.WordPieceTrainer(
        special_tokens=special_tokens,
        vocab_size=vocab_size,
        min_frequency=2,
        continuing_subword_prefix="##",  # BERT-style subword prefix
        show_progress=True
    )

    # Train the tokenizer
    tokenizer.train([input_path], trainer)

    # Add post-processor for automatic [CLS] and [SEP] wrapping (BERT-style)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )

    # Save tokenizer
    tokenizer.save(tokenizer_path)

    # Set language metadata
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    tokenizer_json['model']['language'] = "he"
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=4)

    print(f"WordPiece tokenizer saved to {tokenizer_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: {special_tokens}")
    print("Features: Character-level tokenization, NFD normalization, automatic [CLS]/[SEP] wrapping")
    
    # Show vocabulary preview - Hebrew subwords
    vocab = tokenizer.get_vocab()
    print("\nVocabulary preview (showing Hebrew patterns learned):")
    
    hebrew_tokens = []
    subword_tokens = []
    
    for token, token_id in vocab.items():
        if token in special_tokens:
            continue
        elif any('\u0590' <= c <= '\u05FF' for c in token):  # Hebrew characters
            if token.startswith('##'):
                subword_tokens.append(f"'{token}' (ID: {token_id}) - Hebrew subword")
            else:
                hebrew_tokens.append(f"'{token}' (ID: {token_id}) - Hebrew token")
    
    print("\nHebrew word-level tokens:")
    for token in hebrew_tokens[:10]:
        print(f"  {token}")
    
    print("\nHebrew subword tokens (with ## prefix):")
    for token in subword_tokens[:10]:
        print(f"  {token}")
    
    # Test with Hebrew samples
    test_samples = [
        "שָׁלוֹם עוֹלָם",
        "זֶה טֶקְסְט בְּעִבְרִית", 
        "הַכְּנֶסֶת שֶׁל יִשְׂרָאֵל",
        "הַיְלָדִים מְשַׂחֲקִים בַּגַּן"
    ]
    
    print("\nTesting WordPiece tokenization:")
    for sample in test_samples:
        encoding = tokenizer.encode(sample)
        print(f"\nText: {sample}")
        print(f"Tokens: {encoding.tokens}")
        print(f"Token count: {len(encoding.tokens)}")
        
        # Show token boundaries more clearly
        reconstructed = tokenizer.decode(encoding.ids)
        print(f"Reconstructed: {reconstructed}")
        
        # Show the automatic [CLS] and [SEP] wrapping
        print(f"Note: Automatic [CLS] and [SEP] wrapping is now enabled")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WordPiece tokenizer for Hebrew PL-BERT")
    parser.add_argument("--input-path", 
                        default="dataset_files/parsed_concatenated_dataset.txt",
                        help="path to processed text")
    parser.add_argument("--tokenizer-path",
                        default="tokenizer.json",
                        help="where to save the tokenizer JSON")
    parser.add_argument("--vocab-size", type=int, default=512,
                        help="size of the vocabulary")
    args = parser.parse_args()

    train_tokenizer(
        args.input_path,
        args.tokenizer_path,
        vocab_size=args.vocab_size
    )