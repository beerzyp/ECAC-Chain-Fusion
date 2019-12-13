import numpy as np
import pandas as pd
import os
import keras
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, Dense, Input, Dropout, concatenate
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from data_generator import MultimodalDataGenerator
from math import ceil

BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 5
DROPOUT_PROB = 0.2
DATASET_PATH = "data/"
TABULAR_COLS = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year']

# Read CSV file
df = pd.read_csv(DATASET_PATH + "styles.csv", nrows=200, error_bad_lines=True)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df['usage'] = df['usage'].astype('str')

images = df['image']
tabular = pd.get_dummies(df[TABULAR_COLS])
labels = pd.get_dummies(df['usage'])

dummy_tabular_cols = tabular.columns
dummy_labels_cols = labels.columns

data = pd.concat([ images, tabular, labels ], axis=1)

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
train_labels = train[dummy_labels_cols]

training_generator = MultimodalDataGenerator(
    train_images,
    train_tabular,
    train_labels,
    batch_size=BATCH_SIZE,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    directory=DATASET_PATH + "images/"
)

test_images = test['image']
test_tabular = test[dummy_tabular_cols]
test_labels = test[dummy_labels_cols]

test_generator = MultimodalDataGenerator(
    test_images,
    test_tabular,
    test_labels,
    batch_size=BATCH_SIZE,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    directory=DATASET_PATH + "images/"
)


# Build image model
base_model1 = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dropout_prob=DROPOUT_PROB)

for layer in base_model1.layers:
    layer.trainable = False

x1 = base_model1.output
x1 = GlobalAveragePooling2D()(x1)
x1 = Dropout(DROPOUT_PROB)(x1)

# Add simple input layer for tabular data
base_model2 = Input(batch_shape=(None, len(train_tabular.columns)))

# CHAIN FUSION
x = concatenate([x1, base_model2])

# The same as in the tabular data
x = Sequential()(x)
x = Dense(12, activation='relu')(x)
x = Dropout(DROPOUT_PROB)(x)
x = Dense(8, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=[base_model1.input, base_model2], outputs=predictions) # Inputs go into two different layers
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit_generator(
    generator=training_generator,
    steps_per_epoch=ceil(0.75 * (df.size / BATCH_SIZE)),

    validation_data=test_generator,
    validation_steps=ceil(0.25 * (df.size / BATCH_SIZE)),

    epochs=1,
    verbose=1
)

model.save('weights/chain_fusion.h5')