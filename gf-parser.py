import pgf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

MAX_SENTENCE = 23
MAX_ABSTRACT = 77

def flatten(expr,pad):
    funs = ["[START]"]
    def flatten(expr,arity):
        if type(expr) == pgf.ExprApp:
            flatten(expr.fun,arity+1)
            flatten(expr.arg,0)
        elif type(expr) == pgf.ExprLit:
            funs.append(str(expr.val))
        else:
            funs.append(expr.name)
    flatten(expr,0)
    funs.append("[END]")
    while len(funs) < pad:
        funs.append("[PAD]")
    return funs

def tokenize(line,pad):
    words = ["[START]"]

    start = 0
    while start < len(line):
        while start < len(line) and line[start].isspace():
            start = start+1

        end = start
        if end < len(line):
            if line[end].isalpha():
                while end < len(line) and line[end].isalpha():
                    end = end + 1
            elif line[end].isdigit():
                while end < len(line) and line[end].isdigit():
                    end = end + 1
            else:
                end = end + 1

            words.append(line[start:end].lower())

        start = end

    words.append("[END]")
    while len(words) < pad:
        words.append("[PAD]")

    return words

def train():
    with open("../GF/gf-wordnet/examples.txt") as f:
        i = iter(f)
        abs_vocab = set()
        cnc_vocab = set()
        corpus    = []
        try:
            while True:
                line = next(i)
                if line[:4] == "abs:":
                    abs = flatten(pgf.readExpr(line[5:]),MAX_ABSTRACT)
                    cnc = tokenize(next(i)[5:-1],MAX_SENTENCE)
                    for fun in abs:
                        abs_vocab.add(fun)
                    for w in cnc:
                        cnc_vocab.add(w)
                    corpus.append((abs,cnc))
        except StopIteration:
            pass

        ids_from_abs = tf.keras.layers.StringLookup(
            vocabulary=list(abs_vocab), mask_token=None)

        ids_from_cnc = tf.keras.layers.StringLookup(
            vocabulary=list(cnc_vocab), mask_token=None)

        BUFFER_SIZE = 1000
        BATCH_SIZE = 64

        dataset = tf.data.Dataset.from_tensor_slices((([ids_from_cnc(cnc) for abs,cnc in corpus],[ids_from_abs(abs)[:-1] for abs,cnc in corpus]),[ids_from_abs(abs)[1:] for abs,cnc in corpus]))
        dataset = dataset.shuffle(BUFFER_SIZE) \
                         .batch(BATCH_SIZE) \
                         .prefetch(buffer_size=tf.data.AUTOTUNE)

    def positional_encoding(length, depth):
      depth = depth/2

      positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
      depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

      angle_rates = 1 / (10000**depths)         # (1, depth)
      angle_rads = positions * angle_rates      # (pos, depth)

      pos_encoding = np.concatenate(
          [np.sin(angle_rads), np.cos(angle_rads)],
          axis=-1) 

      return tf.cast(pos_encoding, dtype=tf.float32)

    class PositionalEmbedding(tf.keras.layers.Layer):
      def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

      def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

      def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

    class BaseAttention(tf.keras.layers.Layer):
      def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    class CrossAttention(BaseAttention):
      def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

    class GlobalSelfAttention(BaseAttention):
      def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

    class CausalSelfAttention(BaseAttention):
      def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

    class FeedForward(tf.keras.layers.Layer):
      def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),
          tf.keras.layers.Dense(d_model),
          tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

      def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x

    class EncoderLayer(tf.keras.layers.Layer):
      def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

      def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

    class Encoder(tf.keras.layers.Layer):
      def __init__(self, *, num_layers, d_model, num_heads,
                   dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

      def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
          x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


    class DecoderLayer(tf.keras.layers.Layer):
      def __init__(self,
                   *,
                   d_model,
                   num_heads,
                   dff,
                   dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

      def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x

    class Decoder(tf.keras.layers.Layer):
      def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
                   dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

      def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
          x  = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


    class Transformer(tf.keras.Model):
      def __init__(self, *, num_layers, d_model, num_heads, dff,
                   input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
        self.last_attn_scores = None

      def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x  = inputs
        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        self.last_attn_scores = self.decoder.last_attn_scores

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        # Return the final output and the attention weights.
        return logits

    class Parser(tf.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer

        @tf.function(input_signature=[tf.TensorSpec(shape=[None,MAX_SENTENCE], dtype=tf.int64),tf.TensorSpec(shape=[None,MAX_ABSTRACT-1], dtype=tf.int64)])
        def __call__(self, sentence, c):
            logits = self.transformer((sentence, c), training=False)
            return logits, self.transformer.last_attn_scores

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=len(ids_from_cnc.get_vocabulary()),
        target_vocab_size=len(ids_from_abs.get_vocabulary()),
        dropout_rate=dropout_rate)

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
      def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

      def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    def masked_loss(label, pred):
      mask = label != 0
      loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
      loss = loss_object(label, pred)

      mask = tf.cast(mask, dtype=loss.dtype)
      loss *= mask

      loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
      return loss


    def masked_accuracy(label, pred):
      pred = tf.argmax(pred, axis=2)
      label = tf.cast(label, pred.dtype)
      match = label == pred

      mask = label != 0

      match = match & mask

      match = tf.cast(match, dtype=tf.float32)
      mask = tf.cast(mask, dtype=tf.float32)
      return tf.reduce_sum(match)/tf.reduce_sum(mask)

    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])

    transformer.fit(dataset,
                    epochs=40)

    tf.saved_model.save(Parser(transformer), export_dir='parser')

    with open("parser/assets/abs_vocab","w") as f:
        for word in abs_vocab:
            f.write(word+"\n")
    with open("parser/assets/cnc_vocab","w") as f:
        for word in cnc_vocab:
            f.write(word+"\n")

def plot_attention_head(in_tokens, translated_tokens, attention):
  # The model didn't generate `<START>` in the output. Skip it.
  in_tokens = [t for t in in_tokens if t != "[PAD]"]
  translated_tokens = [t.numpy().decode('utf-8') for t in translated_tokens[1:] if t != b"[PAD]"]

  ax = plt.gca()
  ax.matshow(attention[:len(translated_tokens),:len(in_tokens)])
  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(translated_tokens)))
  ax.set_xticklabels(in_tokens, rotation=90)
  ax.set_yticklabels(translated_tokens)

def plot_attention_weights(in_tokens, translated_tokens, attention_heads):
  fig = plt.figure(figsize=(16, 8))

  for h, head in enumerate(attention_heads[0]):
    ax = fig.add_subplot(2, 4, h+1)

    plot_attention_head(in_tokens, translated_tokens, head)

    ax.set_xlabel(f'Head {h+1}')

  plt.tight_layout()
  plt.show()

def use():
    parser = tf.saved_model.load('parser')
    with open("parser/assets/abs_vocab","r") as f:
        abs_vocab = []
        for line in f:
            abs_vocab.append(line.strip())

        ids_from_abs = tf.keras.layers.StringLookup(
            vocabulary=list(abs_vocab), mask_token=None)

        abs_from_ids = tf.keras.layers.StringLookup(
            vocabulary=ids_from_abs.get_vocabulary(), invert=True, mask_token=None)

    with open("parser/assets/cnc_vocab","r") as f:
        cnc_vocab = []
        for line in f:
            cnc_vocab.append(line.strip())
        ids_from_cnc = tf.keras.layers.StringLookup(
            vocabulary=cnc_vocab, mask_token=None)

    while True:
        s = input("> ")
        sentence = ids_from_cnc([tokenize(s,MAX_SENTENCE)])

        output = list(ids_from_abs(["[START]"]+["[PAD]"]*75).numpy())

        for i in range(75):
            c = tf.constant([output],dtype=tf.int64)
            predictions,attn_scores = parser(sentence, c)

            # Select the last token from the `seq_len` dimension.
            predictions = predictions[:, i:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)[0][0].numpy()

            output[i+1] = predicted_id
            
        print(abs_from_ids(output))
        
        plot_attention_weights(tokenize(s,MAX_SENTENCE), abs_from_ids(output), attn_scores)

train()
#use()
