import numpy as np
import pandas as pd
import os

IMAGES_PATH = 'data/images/'
TABULAR_PATH = 'data/styles.csv'
SAVE_PATH = 'data/prepared_data.csv'

df = pd.read_csv(TABULAR_PATH, nrows=None, error_bad_lines=False)   # error_bad_lines=False drops instances with too many columns
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)

df = df.loc[df['image'].isin(os.listdir(IMAGES_PATH))]  # keep rows that have an image in the IMAGES_PATH
df = df.drop('year', axis=1)

df.to_csv(SAVE_PATH, index=False)