from simple_tokenizer_v2 import SimpleTokenizerV2
from make_vocab_v2 import make_vocab_v2

vocab = make_vocab_v2()

tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))
print(text)

encodedText = tokenizer.encode(text)
print(encodedText)

decodedText = tokenizer.decode(encodedText)
print(decodedText)