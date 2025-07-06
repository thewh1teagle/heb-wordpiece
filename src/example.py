#!/usr/bin/env python3
"""
Test script to see how ## prefix works in WordPiece tokenization
"""

from tokenizers import Tokenizer

def test_tokenizer(tokenizer_path="tokenizer.json"):
    # Load the trained tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print("=== Testing ## prefix behavior ===\n")
    
    # Test samples - mix of Hebrew and English
    test_samples = [
        "שלום",           # Simple Hebrew word
        "שלום עולם",      # Two Hebrew words  
        "הילדים",         # Longer Hebrew word
        "hello",          # Simple English word
        "running",        # English word that might split
        "שלום hello",     # Mixed Hebrew-English
    ]
    
    for sample in test_samples:
        print(f"Input: '{sample}'")
        
        # Encode without post-processing to see raw tokens
        encoding = tokenizer.encode(sample, add_special_tokens=False)
        print(f"Tokens: {encoding.tokens}")
        print(f"IDs: {encoding.ids}")
        
        # Show which tokens have ## prefix
        has_prefix = [token.startswith("##") for token in encoding.tokens]
        print(f"Has ## prefix: {has_prefix}")
        
        # Reconstruct
        reconstructed = tokenizer.decode(encoding.ids)
        print(f"Reconstructed: '{reconstructed}'")
        
        print("-" * 50)

    print("\n=== Understanding ## prefix ===")
    print("• No ##: Token starts a new word")
    print("• ##: Token continues the previous word")
    print("• This helps the model understand word boundaries")
    
    # Test with post-processing (automatic [CLS]/[SEP])
    print("\n=== With post-processing ([CLS]/[SEP]) ===")
    sample = "שלום עולם"
    encoding = tokenizer.encode(sample, add_special_tokens=True)
    print(f"Input: '{sample}'")
    print(f"Tokens: {encoding.tokens}")
    print(f"Note: [CLS] and [SEP] are automatically added!")

if __name__ == "__main__":
    test_tokenizer() 