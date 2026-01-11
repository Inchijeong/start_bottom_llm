from simple_tokenizer_v1 import SimpleTokenizerV1
from make_vocab import make_vocab

vocab = make_vocab()

tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know,"
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

text2 = tokenizer.decode(ids)
print(text2)