import tiktoken

tokencal = tiktoken.encoding_for_model("gpt-4o-mini")
encoded = tokencal.encode("Today we are going to test the tokenization and decoding process using tiktoken library.")

print(f"Encoded tokens: {encoded}")
print(f"Number of encoded tokens: {len(encoded)}")

decodetext = tokencal.decode([9216, 2560])
print(f"Decoded text: {decodetext}")
