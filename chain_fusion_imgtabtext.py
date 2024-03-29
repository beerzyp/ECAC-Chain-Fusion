import numpy as np
import pandas as pd
import tensorflow as tf
import os
import keras
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, Dense, Input, Dropout, concatenate
from keras.applications.vgg16 import VGG16
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from data_generator import TextMultimodalDataGenerator
from math import ceil
import datetime

# If using Nvidia gpu and running into memory issues
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
#tf.TF_ENABLE_GPU_GARBAGE_COLLECTION=False

BATCH_SIZE = 32
IMAGE_SIZE = 224
DROPOUT_PROB = 0.2
DATASET_PATH = "data/"
LOG_PATH = "log/"
TABULAR_COLS = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'usage']
log_name = LOG_PATH + str(datetime.datetime.today().strftime("%Y%m%d%H%M%S")) + ".txt"

# Read CSV file
#df = pd.read_csv(DATASET_PATH + "prepared_data.csv", error_bad_lines=False)
df = pd.read_csv(DATASET_PATH + "balanced_sorted.csv", nrows=100, error_bad_lines=False)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df['usage'] = df['usage'].astype('str')

images = df['image']
tabular = pd.get_dummies(df[TABULAR_COLS])
labels = pd.get_dummies(df['season'])

NUM_CLASSES = len(labels.columns)
dummy_tabular_cols = tabular.columns
dummy_labels_cols = labels.columns

# Text pre-processing
vocab_filename = 'data/vocab.txt'
file = open(vocab_filename, 'r')
vocab = file.read()
file.close()
vocab = vocab.split()
vocab = set(vocab)
sentences = df['productDisplayName'].astype('str').values.tolist()
usage = pd.get_dummies(df['season'])
usage = usage.values.tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
text = pd.DataFrame(tokenizer.texts_to_matrix(sentences, mode='tfidf'))
dummy_text_cols = text.columns

data = pd.concat([ images, tabular, text, labels ], axis=1)

train, test = train_test_split(
    data,
    random_state=42,
    shuffle=True,
    stratify=labels
)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

train_images = train['image']
train_tabular = train[dummy_tabular_cols]
train_text = train[dummy_text_cols]
train_labels = train[dummy_labels_cols]

training_generator = TextMultimodalDataGenerator(
    train_images,
    train_tabular,
    train_text,
    train_labels,
    batch_size=BATCH_SIZE,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    directory=DATASET_PATH + "images/"
)

test_images = test['image']
test_tabular = test[dummy_tabular_cols]
test_text = test[dummy_text_cols]
test_labels = test[dummy_labels_cols]

test_generator = TextMultimodalDataGenerator(
    test_images,
    test_tabular,
    test_text,
    test_labels,
    batch_size=BATCH_SIZE,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    directory=DATASET_PATH + "images/"
)


# Build image model
base_model1 = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

for layer in base_model1.layers:
    layer.trainable = False

x1 = base_model1.output
x1 = GlobalAveragePooling2D()(x1)
x1 = Dropout(DROPOUT_PROB)(x1)

# Add simple input layer for tabular data
input_tab = Input(batch_shape=(None, len(train_tabular.columns)))

# CHAIN FUSION WITH TABULAR
x2 = concatenate([x1, input_tab])
x2 = Dense(x2.shape[1], activation='relu')(x2)
x2 = Dropout(DROPOUT_PROB)(x2)

# Build text model
n_words = text.shape[1]
input_text = Input(batch_shape=(None, len(train_text.columns)))

x = concatenate([x2, input_text])

# The same as in the tabular data
x = Sequential()(x)
x = Dense(x.shape[1], input_shape=(n_words,), activation='relu')(x) #12
x = Dropout(DROPOUT_PROB)(x)
x = Dense(ceil(x.shape[1]/2), activation='relu')(x) #8
x = Dropout(DROPOUT_PROB)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=[base_model1.input, input_tab, input_text], outputs=predictions) # Inputs go into two different layers
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

log_file = open(log_name, 'w')
log_file.write('VGG->Tabular Chain Fusion \n')
summary = model.summary(print_fn=lambda x: log_file.write(x + '\n'))
log_file.close()
print(summary)

callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2,
        verbose=1)
]

history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch=ceil(0.75 * (df.size / BATCH_SIZE)),

    validation_data=test_generator,
    validation_steps=ceil(0.25 * (df.size / BATCH_SIZE)),

    epochs=10,
    callbacks=callbacks,
    verbose=1
)

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(log_name, mode='a', header=True)
model.save('weights/chain_fusion.h5')
log_file.close()