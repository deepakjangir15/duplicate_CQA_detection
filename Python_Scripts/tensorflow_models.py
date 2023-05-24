import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, TextVectorization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import random

data = pd.read_pickle('dataset/checkpoint6.pkl')

max_vocab_length = 50000
max_length = 20

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode='int',
                                    output_sequence_length=max_length)

q1_train = data["question1"]
q2_train = data["question2"]

# Use train_test_split to split training data into training and validation sets
train_q1, val_q1, train_q2, val_q2, train_labels, val_labels = train_test_split(q1_train.to_numpy(),
                                                                                q2_train.to_numpy(),
                                                                                data["is_duplicate"].to_numpy(
),
    test_size=0.15,  # dedicate 15% of samples to validation set
    random_state=101)  # random state for reproducibility\

# Combining the questions into a tf.data dataset

train_questions_data = tf.data.Dataset.from_tensor_slices((train_q1, train_q2))
train_labels_data = tf.data.Dataset.from_tensor_slices(train_labels)
train_questions_dataset = tf.data.Dataset.zip(
    (train_questions_data, train_labels_data))

train_questions_dataset = train_questions_dataset.batch(
    32).prefetch(tf.data.AUTOTUNE)


val_questions_data = tf.data.Dataset.from_tensor_slices((val_q1, val_q2))
val_labels_data = tf.data.Dataset.from_tensor_slices(val_labels)
val_questions_dataset = tf.data.Dataset.zip(
    (val_questions_data, val_labels_data))

val_questions_dataset = val_questions_dataset.batch(
    32).prefetch(tf.data.AUTOTUNE)

max_vocab_length = 50000
max_length = 20

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode='int',
                                    output_sequence_length=max_length)

# Fit the text vectorizer to the training text
text_vectorizer.adapt(train_questions_data)

embedding = layers.Embedding(input_dim=max_vocab_length,
                             output_dim=128,
                             embeddings_initializer='uniform',
                             input_length=max_length,
                             name='embedding1')

# -----> Model 1 - Simple Dense Model

token_inputs_q1 = layers.Input(
    shape=[], dtype=tf.string, name='token_input_q1')
token_embeddings_q1 = text_vectorizer(token_inputs_q1)
x = embedding(token_embeddings_q1)
x = layers.GlobalAveragePooling1D()(x)
token_output_q1 = layers.Dense(128, activation='relu')(x)
token_model_q1 = tf.keras.Model(
    inputs=token_inputs_q1, outputs=token_output_q1)

token_inputs_q2 = layers.Input(
    shape=[], dtype=tf.string, name='token_input_q2')
token_embeddings_q2 = text_vectorizer(token_inputs_q2)
x = embedding(token_embeddings_q2)
x = layers.GlobalAveragePooling1D()(x)
token_output_q2 = layers.Dense(128, activation='relu')(x)
token_model_q2 = tf.keras.Model(
    inputs=token_inputs_q2, outputs=token_output_q2)
token_questions_concat = layers.Concatenate(name='token_questions_cat')([token_model_q1.output,
                                                                         token_model_q2.output])

combined_dropout = layers.Dropout(0.5)(token_questions_concat)
combined_dense = layers.Dense(64, activation='relu')(combined_dropout)
final_dropout = layers.Dropout(0.5)(combined_dense)
output_layer = layers.Dense(1, activation='sigmoid')(final_dropout)

model_1 = tf.keras.Model(inputs=[token_model_q1.input, token_model_q2.input],
                         outputs=output_layer,
                         name='model_1_token')
model_1.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

model_1_history = model_1.fit(train_questions_dataset,
                              epochs=5,
                              validation_data=val_questions_dataset,
                              )

# -----> Model 2 - LSTM

model_2_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer='uniform',
                                     input_length=max_length,
                                     name='embedding_2')


token_inputs_q1 = layers.Input(
    shape=[], dtype=tf.string, name='token_input_q1')
token_embeddings_q1 = text_vectorizer(token_inputs_q1)
x = model_2_embedding(token_embeddings_q1)
x = layers.LSTM(64)(x)
token_output_q1 = layers.Dense(128, activation='relu')(x)
token_model_q1 = tf.keras.Model(
    inputs=token_inputs_q1, outputs=token_output_q1)

token_inputs_q2 = layers.Input(
    shape=[], dtype=tf.string, name='token_input_q2')
token_embeddings_q2 = text_vectorizer(token_inputs_q2)
x = model_2_embedding(token_embeddings_q2)
x = layers.LSTM(64)(x)
token_output_q2 = layers.Dense(128, activation='relu')(x)
token_model_q2 = tf.keras.Model(
    inputs=token_inputs_q2, outputs=token_output_q2)

token_questions_concat = layers.Concatenate(name='token_questions_cat')([token_model_q1.output,
                                                                         token_model_q2.output])

combined_dropout = layers.Dropout(0.5)(token_questions_concat)
combined_dense = layers.Dense(64, activation='relu')(combined_dropout)
final_dropout = layers.Dropout(0.5)(combined_dense)
output_layer = layers.Dense(1, activation='sigmoid')(final_dropout)

model_2 = tf.keras.Model(inputs=[token_model_q1.input, token_model_q2.input],
                         outputs=output_layer,
                         name='model_2_token')

model_2.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

model_2_history = model_2.fit(train_questions_dataset,
                              epochs=5,
                              validation_data=val_questions_dataset,
                              )

# -----> Model 3 - GRU

model_3_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer='uniform',
                                     input_length=max_length,
                                     name='embedding_3')


token_inputs_q1 = layers.Input(
    shape=[], dtype=tf.string, name='token_input_q1')
token_embeddings_q1 = text_vectorizer(token_inputs_q1)
x = model_3_embedding(token_embeddings_q1)
x = layers.GRU(64)(x)
token_output_q1 = layers.Dense(128, activation='relu')(x)
token_model_q1 = tf.keras.Model(
    inputs=token_inputs_q1, outputs=token_output_q1)

token_inputs_q2 = layers.Input(
    shape=[], dtype=tf.string, name='token_input_q2')
token_embeddings_q2 = text_vectorizer(token_inputs_q2)
x = model_3_embedding(token_embeddings_q2)
x = layers.GRU(64)(x)
token_output_q2 = layers.Dense(128, activation='relu')(x)
token_model_q2 = tf.keras.Model(
    inputs=token_inputs_q2, outputs=token_output_q2)

token_questions_concat = layers.Concatenate(name='token_questions_cat')([token_model_q1.output,
                                                                         token_model_q2.output])

combined_dropout = layers.Dropout(0.5)(token_questions_concat)
combined_dense = layers.Dense(64, activation='relu')(combined_dropout)
final_dropout = layers.Dropout(0.5)(combined_dense)
output_layer = layers.Dense(1, activation='sigmoid')(final_dropout)

model_3 = tf.keras.Model(inputs=[token_model_q1.input, token_model_q2.input],
                         outputs=output_layer,
                         name='model_3_token')

model_3.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

model_3_history = model_3.fit(train_questions_dataset,
                              epochs=5,
                              validation_data=val_questions_dataset,
                              )
# -----> Model 4 - Bidirectional LSTM

model_4_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer='uniform',
                                     input_length=max_length,
                                     name='embedding_4')


token_inputs_q1 = layers.Input(
    shape=[], dtype=tf.string, name='token_input_q1')
token_embeddings_q1 = text_vectorizer(token_inputs_q1)
x = model_4_embedding(token_embeddings_q1)
x = layers.Bidirectional(layers.LSTM(64))(x)
token_output_q1 = layers.Dense(128, activation='relu')(x)
token_model_q1 = tf.keras.Model(
    inputs=token_inputs_q1, outputs=token_output_q1)

token_inputs_q2 = layers.Input(
    shape=[], dtype=tf.string, name='token_input_q2')
token_embeddings_q2 = text_vectorizer(token_inputs_q2)
x = model_4_embedding(token_embeddings_q2)
x = layers.Bidirectional(layers.LSTM(64))(x)
token_output_q2 = layers.Dense(128, activation='relu')(x)
token_model_q2 = tf.keras.Model(
    inputs=token_inputs_q2, outputs=token_output_q2)

token_questions_concat = layers.Concatenate(name='token_questions_cat')([token_model_q1.output,
                                                                         token_model_q2.output])

combined_dropout = layers.Dropout(0.5)(token_questions_concat)
combined_dense = layers.Dense(64, activation='relu')(combined_dropout)
final_dropout = layers.Dropout(0.5)(combined_dense)
output_layer = layers.Dense(1, activation='sigmoid')(final_dropout)

model_4 = tf.keras.Model(inputs=[token_model_q1.input, token_model_q2.input],
                         outputs=output_layer,
                         name='model_4_token')

model_4.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

model_4_history = model_4.fit(train_questions_dataset,
                              epochs=5,
                              validation_data=val_questions_dataset,
                              )


# -----> Model 5 - Conv1D
model_5_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer='uniform',
                                     input_length=max_length,
                                     name='embedding_5')


token_inputs_q1 = layers.Input(
    shape=[], dtype=tf.string, name='token_input_q1')
token_embeddings_q1 = text_vectorizer(token_inputs_q1)
x = model_5_embedding(token_embeddings_q1)
x = layers.Conv1D(filters=32, kernel_size=5, activation='relu')(x)
x = layers.GlobalMaxPool1D()(x)
token_output_q1 = layers.Dense(128, activation='relu')(x)
token_model_q1 = tf.keras.Model(
    inputs=token_inputs_q1, outputs=token_output_q1)

token_inputs_q2 = layers.Input(
    shape=[], dtype=tf.string, name='token_input_q2')
token_embeddings_q2 = text_vectorizer(token_inputs_q2)
x = model_5_embedding(token_embeddings_q2)
x = layers.Conv1D(filters=32, kernel_size=5, activation='relu')(x)
x = layers.GlobalMaxPool1D()(x)
token_output_q2 = layers.Dense(128, activation='relu')(x)
token_model_q2 = tf.keras.Model(
    inputs=token_inputs_q2, outputs=token_output_q2)

token_questions_concat = layers.Concatenate(name='token_questions_cat')([token_model_q1.output,
                                                                         token_model_q2.output])

combined_dropout = layers.Dropout(0.5)(token_questions_concat)
combined_dense = layers.Dense(64, activation='relu')(combined_dropout)
final_dropout = layers.Dropout(0.5)(combined_dense)
output_layer = layers.Dense(1, activation='sigmoid')(final_dropout)

model_5 = tf.keras.Model(inputs=[token_model_q1.input, token_model_q2.input],
                         outputs=output_layer,
                         name='model_5_token')

model_5.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

model_5_history = model_5.fit(train_questions_dataset,
                              epochs=5,
                              validation_data=val_questions_dataset,
                              )
# -----> Model 6 -TF hub Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

question_encoder_layer = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',
                                        input_shape=[],
                                        dtype=tf.string,
                                        trainable=False,
                                        name='USE')

token_inputs_q1 = layers.Input(
    shape=[], dtype=tf.string, name='token_input_q1')
token_embeddings_q1 = question_encoder_layer(token_inputs_q1)
token_output_q1 = layers.Dense(128, activation='relu')(token_embeddings_q1)
token_model_q1 = tf.keras.Model(inputs=token_inputs_q1,
                                outputs=token_output_q1)

token_inputs_q2 = layers.Input(
    shape=[], dtype=tf.string, name='token_input_q2')
token_embeddings_q2 = question_encoder_layer(token_inputs_q2)
token_output_q2 = layers.Dense(128, activation='relu')(token_embeddings_q2)
token_model_q2 = tf.keras.Model(inputs=token_inputs_q2,
                                outputs=token_output_q2)

token_questions_concat = layers.Concatenate(name='token_questions_cat')([token_model_q1.output,
                                                                         token_model_q2.output])

combined_dropout = layers.Dropout(0.5)(token_questions_concat)
combined_dense = layers.Dense(64, activation='relu')(combined_dropout)
final_dropout = layers.Dropout(0.5)(combined_dense)
output_layer = layers.Dense(1, activation='sigmoid')(final_dropout)

model_6 = tf.keras.Model(inputs=[token_model_q1.input, token_model_q2.input],
                         outputs=output_layer,
                         name='model_6_token')

model_6.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

model_6_history = model_6.fit(train_questions_dataset,
                              epochs=5,
                              validation_data=val_questions_dataset,
                              )
