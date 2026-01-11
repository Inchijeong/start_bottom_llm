from simple_tokenizer_v1 import SimpleTokenizerV1
from make_vocab import make_vocab

vocab = make_vocab()

tokenizer = SimpleTokenizerV1(vocab)

text = "Hello, do you like tea. Is this-- a test?"

tokenizer.encode(text)