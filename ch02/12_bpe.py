import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

print(tokenizer.special_tokens_set)

print(tokenizer.encode(text, allowed_special='all'))

strings = tokenizer.decode(integers)

print(strings)