bos_token = '<s>'
eos_token = '</s>'
pad_token = '<pad>'
unk_token = '<unk>'

class TokenEncoder:
  def __init__(self, seq_len):
    self.seq_len = seq_len
    self.token2index = {}
    self.token2count = {}
    self.index2token = {0: bos_token, 1: eos_token, 2: pad_token, 3: unk_token}

    self.bos_token = bos_token
    self.eos_token = eos_token
    self.pad_token = pad_token
    self.unk_token = unk_token

    self.bos_token_id = 0
    self.eos_token_id = 1
    self.pad_token_id = 2
    self.unk_token_id = 3    

    self.n_tokens = 4  # Count bos, eos, and pad tokens

  def add_sequence(self, sequence):
    for token in sequence:
        self.add_token(token)

  def add_token(self, token):
    if token not in self.token2index:
      self.token2index[token] = self.n_tokens
      self.token2count[token] = 1
      self.index2token[self.n_tokens] = token
      self.n_tokens += 1
    else:
      self.token2count[token] += 1
  
  def get_token_index(self, token):
    if token not in self.token2index:
      return self.unk_token_id
    return self.token2index[token]
  
  def decode_sequence(self, encoded_sequence):
    sequence = []
    for token in encoded_sequence:
      sequence.append(self.index2token[token])
    return sequence

def build_io_token_encodings(
  input_space,
  output_space,
  pairs
):
  for (input_seq, output_seq, _) in pairs:
    input_space.add_sequence(input_seq)
    output_space.add_sequence(output_seq)