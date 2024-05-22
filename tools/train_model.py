import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.applications import MobileNetV2
from sklearn.metrics import confusion_matrix, roc_curve, auc  

#export TF_CPP_MIN_LOG_LEVEL=2
#export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/nomyeet/miniconda3/pkgs/cuda-nvcc-12.3.107-0
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
def lr_schedule(epoch, lr):
    if epoch > 30 and (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
parent_folder = 'spectrogram_data'
label_folder = 'apnlabel_data'

all_data = []  # Initialize the list to store image paths
all_labels = []

input_folders = [os.path.join(parent_folder, subfolder) for subfolder in os.listdir(
    parent_folder) if os.path.isdir(os.path.join(parent_folder, subfolder))]
csv_paths = [os.path.join(label_folder, f'{subfolder}apn.csv') for subfolder in os.listdir(
    parent_folder) if os.path.isdir(os.path.join(parent_folder, subfolder))]

for input_folder, csv_path in zip(input_folders, csv_paths):
    labels_df = pd.read_csv(csv_path)
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            row_number = int(filename.split('_')[1].split('.')[0])
            if row_number <= len(labels_df):
                label = labels_df.iloc[row_number - 1, 2]
                class_mapping = {'A': 0, 'N': 1}
                all_labels.append(class_mapping[label])
                all_data.append(img_path)  # Add image path to all_data list

all_labels = np.array(all_labels)
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
xtrain, xtest, ytrain, ytest = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
) 

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': xtrain, 'class': ytrain.astype(str)}),
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': xtest, 'class': ytest.astype(str)}),
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    shuffle=False
)

y_labels = np.array(train_generator.classes)
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
base_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lr_scheduler = LearningRateScheduler(lr_schedule)
callback1 = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(train_generator, epochs=100, validation_data=test_generator, 
    callbacks=[callback1, lr_scheduler])

Y_pred = model.predict(test_generator)
Y_pred_classes = (Y_pred > 0.5).astype(int)
Y_true = ytest

model.save(os.path.join("model.h5"))
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
C = confusion_matrix(Y_true, Y_pred_classes, labels=(1,0))
TP, TN, FP, FN = C[0,0], C[1,1], C[1,0], C[0,1]

print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.plot()
plt.show()
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(Y_true, Y_pred)

roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.plot()
plt.show()
