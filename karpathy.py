import tensorflow as tf

import numpy as np
import os
import sys

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

def create_model(vocab):
    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None)

    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    # Length of the vocabulary in StringLookup Layer
    vocab_size = len(ids_from_chars.get_vocabulary())

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    return (ids_from_chars,
            chars_from_ids,
            MyModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                rnn_units=rnn_units))

def train(origin,name):
    path_to_file = tf.keras.utils.get_file(origin=origin)
    text = open(path_to_file, 'r').read()

    vocab = sorted(set(text))

    ids_from_chars,chars_from_ids,model = create_model(vocab)

    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    seq_length = 100
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text
        

    dataset = sequences.map(split_input_target)

    # Batch size
    BATCH_SIZE = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)

    EPOCHS = 20
    model.fit(dataset, epochs=EPOCHS, callbacks=[])
    
    model.save_weights(name+'/model')
    with open(name+"/vocab","w") as f:
        f.write(str(vocab))

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

def use(name):
    with open(name+"/vocab","r") as f:
        vocab = eval(f.read())

    ids_from_chars,chars_from_ids,model = create_model(vocab)
    model.load_weights(name+"/model")

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    states = None
    next_char = tf.constant(["ROMEO:"])
    result = [next_char]
    while True:
        prompt = input("prompt> ")
        if prompt:
            next_char = tf.constant([prompt])
            result = [next_char]
            states = None

        while True:
          next_char, states = one_step_model.generate_one_step(next_char, states=states)
          result.append(next_char)
          if next_char[0].numpy() == b'\n':
              break

        print(tf.strings.join(result)[0].numpy().decode('utf-8'),sep="")
        result = []

if len(sys.argv) > 3 and sys.argv[1] == "train":
    train(sys.argv[2],sys.argv[3])
elif len(sys.argv) > 2 and sys.argv[1] == "use":
    use(sys.argv[2])
else:
    print("Example uses")
    print("   python3 karpathy.py train https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt shakespeare")
    print("   python3 karpathy.py use   shakespeare")
