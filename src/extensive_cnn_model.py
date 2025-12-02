
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_METAL_ENABLED"] = "0"


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices([], 'GPU')
except:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except:
    pass

IMG_SIZE = 150
DATA_PATH = "data/CombinedDataset/train"
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def load_data_manual(path):
    X, y = [], []
    class_names = sorted(os.listdir(path))
    print("Classes:", class_names)

    for label, cls in enumerate(class_names):
        cls_path = os.path.join(path, cls)
        for img_name in os.listdir(cls_path):

            img_path = os.path.join(cls_path, img_name)

            if not os.path.isfile(img_path):
                continue

            if not img_name.lower().endswith(ALLOWED_EXTENSIONS):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y), class_names

data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
])


def build_extensive_cnn(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_aug(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Dense Head
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

# --------------------------------------------------------------
# MAIN TRAINING LOOP
# --------------------------------------------------------------
def main():
    print("Loading dataset manually...")
    X, y, class_names = load_data_manual(DATA_PATH)
    print("Loaded:", X.shape, y.shape)

    # Train/Validation Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(16)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(16)

    print(f"Building model with {len(class_names)} classes")
    model = build_extensive_cnn(len(class_names))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Model compiled.")
    model.summary()

    print("\nStarting CPU Training...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=40
    )

    os.makedirs("results", exist_ok=True)
    model.save("results/extensive_cnn_model.h5")
    print("\nModel saved to results/extensive_cnn_model.h5")

# --------------------------------------------------------------
if __name__ == "__main__":
    main()
