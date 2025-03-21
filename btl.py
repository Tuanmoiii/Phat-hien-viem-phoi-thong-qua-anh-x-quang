import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Định nghĩa đường dẫn đến dataset
DATASET_PATH = "./chest_xray/"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory(VAL_PATH, target_size=(224, 224), batch_size=32, class_mode='binary')

def build_model(base_model):
    base_model.trainable = True  # Mở khóa mô hình
    for layer in base_model.layers[:-20]:  # Chỉ fine-tune phần cuối
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    return model

# Chọn mô hình nền tảng
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = build_model(base_model)

# Compile mô hình
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình với EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop]
)

# Lưu mô hình
model.save("xray_pneumonia_model.h5")

# Grad-CAM

def grad_cam(model, img_array, layer_name):
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Hiển thị Grad-CAM cho tất cả ảnh trong thư mục test
def display_gradcam_for_all_images(test_folder, model, layer_name='conv5_block16_concat'):
    for class_name in os.listdir(test_folder):
        class_path = os.path.join(test_folder, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img, (224, 224))
                img_array = np.expand_dims(img_resized, axis=0) / 255.0
                
                heatmap = grad_cam(model, img_array, layer_name)
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
                
                pred = model.predict(img_array)[0][0]
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.title("Original Image")
                plt.axis("off")
                
                plt.subplot(1, 2, 2)
                plt.imshow(superimposed_img)
                plt.title(f"Grad-CAM - Prediction: {pred:.2%} Pneumonia")
                plt.axis("off")
                plt.show()

# Áp dụng Grad-CAM cho tất cả ảnh trong thư mục test
display_gradcam_for_all_images(TEST_PATH, model)
