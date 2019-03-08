from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.optimizers import RMSprop, Adamax
import keras
import keras.utils
import tensorflow as tf
from keras import metrics

# This is a generic attentional seq2seq model.  Guide by Collin to help students understand
# how to implement the basic idea and what attention is.

# I write this guide with much thanks to:
# https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html
# https://arxiv.org/abs/1508.04025

class AttentionGRUModel:
    def __init__(self, config):
        
        # override default tdatlen
        config['tdatlen'] = 50
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.datlen = config['tdatlen']
        self.comlen = config['comlen']
        
        self.embdims = 100
        self.recdims = 256

        self.config['num_input'] = 2
        self.config['num_output'] = 1

    def create_model(self):

        # The first two lines here are the input.  We assume a fixed-size input, padded to 
        # datlen and comlen.  In principle, we could avoid padding if we carefully controlled
        # the batches during training, so all equal-sized inputs go in during the same batches.
        # But that is a lot of trouble for not much benefit.
        
        dat_input = Input(shape=(self.datlen,))
        com_input = Input(shape=(self.comlen,))

        # This is a fairly-common encoder structure.  It is just an embedding space followed by
        # a uni-directional GRU.  We have to disable masking here, since not all following layers
        # support masking.  Hopefully the network will learn to ignore zeros anyway.
        
        ee = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)(dat_input)

        # The regular GRU can be swapped for the CuDNNGRU if desired.  The CuDNNGRU seems to be
        # around 50% faster in this model, but we are constained to GPU training /and testing/.

        # The return_state flag is necessary so that we get the hidden state of the encoder, to pass
        # to the decoder later.  The return_sequences flag is necessary because we want to get the
        # state /at every cell/ instead just the final state.  We need the state at every cell for the
        # attention mechanism later.
        
        # The embedding will output a shape of (batch_size, tdatvocabsize, embdims).  What this means
        # is that for every batch, each word in the sequence has one vector of length embdims.  For
        # example, (300, 100, 100) means that for each of 300 examples in a batch, there are 100 words.
        # and each word is represented by a 100 length embedding vector.
        
        #enc = GRU(self.recdims, return_state=True, return_sequences=True, activation='tanh', unroll=True)
        enc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        encout, state_h = enc(ee)
        
        # Tensor encout would normally have shape (batch_size, recdims), a recdims-length vector
        # representation of every input in the batch.  However, since we have return_sequences enabled,
        # encout has the shape (batch_size, tdatvocabsize, recdims), which is the recdims-length
        # vector at every time-step.  That is, the recdims-length vector at every word in the sequence.
        # So we see the status of the output vector as it changes with each word in the sequence.
        # We also have return_state enabled, which just means that we get state_h, the recdims vector
        # from the last cell.  This is a GRU, so this state_h is the same as the output vector, but we
        # get it here anyway for convenience, to use as the initial state in the decoder.
        
        # The decoder is basically the same as the encoder, except the GRU does not need return_state
        # enabled.  The embedding will output (batch_size, comvocabsize, embdims).  The GRU will
        # output (batch_size, comvocabsize, recdims).
        
        # I suppose we could speed things up by not setting the initial_state of the decoder to have
        # the output state of the encoder.  Right now, the GPU will have to wait until the encoder is
        # done before starting the decoder.  Not sure how much this would affect quality.
        
        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        #dec = GRU(self.recdims, return_sequences=True, activation='tanh', unroll=True)
        dec = CuDNNGRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=state_h)
        
        # Ok now things become more interesting.  This is the start of the attention mechanism.
        # In the first of these two lines, we take the dot product of the decoder and encoder
        # output.  Remember that the output shape of decout is, e.g., (batch_size, 13, 256) and
        # encout is (batch_size, 100, 256).
        
        # The axis 2 of decout is 256 long.  The axis 2 of encout is also 256 long.  So by computing
        # the dot product along the 2 axis in both, we get a tensor of shape (batch_size, 13, 100).
        
        # For one example in the batch, we get decout of (13, 256) and encout (100, 256).
        #
        #      1 2 ... 256          1 2 ... 256         1 2 ... 100
        #      ___________          __________          __________
        #   1 |v1------->        1 |v3------->       1 |a b
        #   2 |v2------->    *   2 |v4------->   =   2 |c d
        #  .. |                 .. |                .. |
        #  13 |                100 |                13 |
        #
        # Where a is the dot product of vectors v1 and v3, and b is the dot product of v1 and v4, etc.
        # a = v1 . v3
        # b = v1 . v4
        # c = v2 . v3
        # d = v2 . v4
        # This may look a little different than the dot product we all did back in school, but the 
        # behavior is dictated by the axes given in the dot() function parameters.
        # Another way to think of it is if we transpose the second matrix prior to computing the
        # product.  Then it behaves more like we expect.
        
        # In any case, the result is that each of the 13 positions in the decoder sequence is now
        # represented by a 100-length vector.  Each value in the 100-length vector reflects the 
        # similarity between the element in the decoder sequence and the element in the encoder
        # sequence.  I.e. 'b' above reflects how similar element 1 in the output/decoder sequence
        # is similar to element 2 in the input/encoder sequence.  This is the heart of how attention
        # works.  The 100-length vector for each of the 13 input positions represents how much
        # that a given input position is similar (should "pay attention to") a given position in
        # the output vector.
        
        # The second line applies a softmax to each of the 13 (100-length) vectors.  The effect is
        # to exaggerate the 'most similar' things, so that 'more attention' will be paid to the 
        # more-similar input vectors.
        
        # Note that the dot product here is not normalized, so it is not necessarily equivalent to
        # cosine similarity.  Also, the application of softmax is not required, but recommended.

        attn = dot([decout, encout], axes=[2, 2])
        attn = Activation('softmax')(attn)

        # To be 100% clear, the output shape of attn is (batch_size, comvocabsize, tdatvocabsize).
        
        # But what do we do with the attention vectors, now that we have them?  Answer is that we
        # need to scale the encoder vectors by the attention vectors.  This is how we 'pay 
        # attention' to particular areas of input for specific outputs.  The following line
        # takes attn, with shape (batch_size, 13, 100), and takes the dot product with
        # encout (batch_size, 100, 256).  Remember that the encoder has tdatvocabsize, 100 in this
        # example, elements since it takes a sequence of 100 words.  Axis 1 of this tensor means
        # 'for each element of the input sequence'.
        
        # The multiplication this time, for each sample in the batch, is:
        #
        #     attn (axis 2)        encout (axis 1)         context
        #      1 2 ... 100          1 2 ... 100         1 2 ... 256
        #      ___________          __________          __________
        #   1 |v1------->        1 |v3------->       1 |a b
        #   2 |v2------->    *   2 |v4------->   =   2 |c d
        #  .. |                 .. |                .. |
        #  13 |                256 |                13 |
        #
        # The result is a context /matrix/ that has one context /vector/ for each element in
        # the output sequence.  This is different than the vanilla sequence to sequence
        # approach, which has only one context vector used for every output.
        #
        # Each output word has its own customized context vector.  The context vector is
        # created from the most attended-to part of the encoder sequence.

        context = dot([attn, encout], axes=[2,1])
        
        # But... we still don't have the decoder sequence information.  This is important
        # because when we train, we send each word one at a time.  So instead of sending:
        # [ 'cat', 'on', 'the', 'table' ] => [ 'chat', 'sur', 'la', 'table' ]
        # we send:
        # [ 'cat', 'on', 'the', 'table' ] => [ 'chat', 0, 0, 0 ] + [ 'sur' ]
        # [ 'cat', 'on', 'the', 'table' ] => [ 'chat', 'sur', 0, 0 ] + [ 'la' ]
        # [ 'cat', 'on', 'the', 'table' ] => [ 'chat', 'sur', 'la', 0 ] + [ 'table' ]
        # (plus start and end sentence tokens)
        
        # In other words, the model gets to look at the previous words in the sentence in
        # addition to the words in the encoder sequence.  It does not have the burden of
        # predicting the entire output sequence all at once.
        
        # But somehow we have to get the decoder information into the final dense prediction
        # layers along with the context matrix.  So we just concatenate them.  Technically,
        # what we have here is a context matrix with shape (batch_size, 13, 256) and a
        # decout with shape (batch_size, 13, 256).  The default axis is -1, which means the
        # last part of the shape (the 256 one in this case).  All this does is create 
        # a tensor of shape (batch_size, 13, 512)... one 512-length vector for each of the
        # 13 input elements instead of two 256-length vectors.
        
        context = concatenate([context, decout])
        
        # To be clear, context's shape is now (batch_size, comvocabsize, recdims*2).
        
        # Now we are ready to actually predict something.  Using TimeDistributed here gives
        # us one dense layer per vector in the context matrix.  So, we end up with one
        # recdims-length vector for every element in the decoder sequence.  For example,
        # one 256-length vector for each of the 13 positions in the decoder sequence.
        
        # A way to think of it is, one predictor for each of the 13 decoder positions.
        # The hope is that the network will learn which of the predictors to use, based
        # on which position the network is trying to predict.
        
        out = TimeDistributed(Dense(self.recdims, activation="tanh"))(context)

        # out's shape is now (batch_size, 13, 256)
        
        # But... we are trying to output a single word, the next word in the sequence.
        # So ultimately we need a single output vector of length comsvocabsize.
        # For example, this could be a single 3000-length vector in which every element
        # represents one word in the vocab.
        
        # To get that, we first flatten the (13, 256) matrix into a single (3328) vector.
        # Then, we use a dense output layer of length comsvocabsize, and apply softmax.

        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[dat_input, com_input], outputs=out)
        
        # I do not imagine that the multigpu model will help much here, because the layers
        # depend a lot on the output of other layers in a sequential pattern, and
        # the overhead of moving everything back and forth between the GPUs is likely to
        # soak up any advantage we get from parallelizing the arithmetic.

        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.config, model
