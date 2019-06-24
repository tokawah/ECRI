from keras import Input, Model
from keras.layers import Embedding, Dense, Reshape, Dropout, GlobalMaxPooling1D, Conv1D
from keras.layers import Multiply, Add, Subtract, Concatenate


def get_encoder(emb):
    encoded = []
    for region_size, num_filters in [(3, 200), (4, 200), (5, 200)]:
        conv = Conv1D(
            filters=num_filters,
            kernel_size=region_size,
            padding='same',
        )(emb)
        gate = Conv1D(
            filters=num_filters,
            kernel_size=region_size,
            activation='sigmoid',
            padding='same',
        )(emb)
        glu = Subtract()([conv, emb])
        glu = Multiply()([glu, gate])
        glu = Add()([emb, glu])
        pooled = GlobalMaxPooling1D()(glu)
        encoded.append(pooled)
    concat = Concatenate()(encoded)
    return concat


def get_model(emb_dim, sentence_len, word_vocab_size):
    # content encoding
    word_input = Input(shape=(sentence_len,), dtype='int32', name='input')
    word_emb = Embedding(
        input_dim=word_vocab_size, output_dim=emb_dim,
        trainable=True, name='word_embeddings'
    )(word_input)
    content_encoded = get_encoder(word_emb)
    content_encoded = Dense(units=200, activation='relu', name='doc_emb')(content_encoded)
    # explicit content-rating interaction
    rat_input = Input(shape=(1,), name='rat_input')
    rat_emb = Embedding(
        input_dim=5, output_dim=emb_dim,
        trainable=True, name='rat_embeddings'
    )(rat_input)
    rat_emb = Reshape((200,))(rat_emb)
    rat_ratio = Dense(units=emb_dim, activation='sigmoid')(content_encoded)
    rat_needed = Multiply()([rat_ratio, rat_emb])
    content_encoded = Add()([content_encoded, rat_needed])
    # output
    dropped = Dropout(0.5)(content_encoded)
    help_predictions = Dense(units=1, activation='sigmoid', name='help')(dropped)
    model = Model([word_input, rat_input], help_predictions)
    return model