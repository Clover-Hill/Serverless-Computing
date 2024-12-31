from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

dataset = load_dataset("/data1/jqcao/datasets/wikitext", "wikitext-103-v1", split="train")["text"]

def save_words(dataset, filename, max_words=5_000_000):
    # Initialize a counter for words and a list to store words
    word_count = 0
    all_words = []

    # Process each text entry
    for text in tqdm(dataset, desc="Processing text"):
        # Skip empty lines
        if not text.strip():
            continue
            
        # Split into words and count
        words = text.split()
        
        # Add words if we're still under max words
        if word_count + len(words) <= max_words:
            all_words.extend(words)
            word_count += len(words)
        else:
            # Add partial words to reach exactly max words
            remaining = max_words - word_count
            if remaining > 0:
                all_words.extend(words[:remaining])
            break

    # print distinct number of words
    print(f"Distinct number of words: {len(set(all_words))}")
    
    # Write to file
    # with open(filename, "w", encoding="utf-8") as f:
    #     f.write("\n".join(all_words))

    print(f"Saved {max_words} words to {filename}")

def save_tokens(dataset, filename, max_words=5_000_000):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data1/jqcao/gpt2")
    
    # First get text up to max_words
    word_count = 0
    selected_text = []

    # Process each text entry to get words
    for text in tqdm(dataset, desc="Processing text"):
        # Skip empty lines
        if not text.strip():
            continue
            
        # Split into words and count
        words = text.split()
        
        # Add text if we're still under max words
        if word_count + len(words) <= max_words:
            selected_text.append(text)
            word_count += len(words)
        else:
            # Add partial text to reach exactly max words
            remaining = max_words - word_count
            if remaining > 0:
                words = words[:remaining]
                selected_text.append(" ".join(words))
            break

    # Now tokenize the selected text
    full_text = "\n".join(selected_text)
    tokens = tokenizer(full_text)["input_ids"]
    token_strings = tokenizer.convert_ids_to_tokens(tokens)
    
    # print distinct number of tokens
    print(f"Distinct number of tokens: {len(set(tokens))}")

    # Write tokens to file
    # with open(filename, "w", encoding="utf-8") as f:
    #     f.write("\n".join(str(token) for token in tokens))

    print(f"Saved {len(tokens)} tokens from {max_words} words to {filename}")

save_words(dataset, "dataset/wikitext-train-10M-words.txt", max_words=100000)

save_tokens(dataset, "dataset/wikitext-train-100K-tokens.txt", max_words=100000)