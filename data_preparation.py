import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to dataset directories
DATASET_DIR = 'chest_xray'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')
val_generator = val_datagen.flow_from_directory(VAL_DIR,
                                                target_size=(150, 150),
                                                batch_size=32,
                                                class_mode='binary')
