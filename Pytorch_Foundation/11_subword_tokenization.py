from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "unfriendly, unbelievable, and hyperactive!"
tokens = tokenizer.tokenize(text)

input_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"Original Text: '{text}'\n")
print(f"Subword Tokens: {tokens}")
print(f"Token IDs:      {input_ids}")

encoded_ids = tokenizer.encode(text)

decoded_text = tokenizer.decode(encoded_ids)

print("\nEncoded IDs (with special tokens):", encoded_ids)
print("Decoded Text Back:", decoded_text)