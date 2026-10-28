import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# RASPBERRY PI SETTINGS
# ============================================================

TRAIN_DIR = "/home/pi4/waste_classifier/dataset/train"
TEST_DIR = "/home/pi4/waste_classifier/dataset/test"

IMG_SIZE = 128
BATCH_SIZE = 2
EPOCHS = 25

# ============================================================
# CHECK DATASET
# ============================================================

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def count_images(folder):
    total = 0

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                total += 1

    return total


print("\n" + "=" * 60)
print("DATASET CHECK")
print("=" * 60)

for dataset_name, dataset_path in [
    ("TRAIN", TRAIN_DIR),
    ("TEST", TEST_DIR)
]:

    print(f"\n{dataset_name} DATASET:")

    if not os.path.exists(dataset_path):
        print(f"ERROR: Folder not found -> {dataset_path}")
        exit()

    for class_name in sorted(os.listdir(dataset_path)):

        class_path = os.path.join(
            dataset_path,
            class_name
        )

        if os.path.isdir(class_path):

            image_count = count_images(
                class_path
            )

            print(
                f"{class_name}: "
                f"{image_count} images"
            )


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING TRAINING DATASET")
print("=" * 60)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING TEST DATASET")
print("=" * 60)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = train_ds.class_names

NUM_CLASSES = len(class_names)

print("\n" + "=" * 60)
print("CLASSES FOUND")
print("=" * 60)

for i, class_name in enumerate(class_names):

    print(
        f"Class {i} -> {class_name}"
    )


print(
    f"\nTotal Classes: {NUM_CLASSES}"
)


# ============================================================
# DATA PREPROCESSING
# LOW MEMORY SETTINGS FOR RASPBERRY PI
# ============================================================

def preprocess(image, label):

    image = tf.cast(
        image,
        tf.float32
    )

    image = image / 255.0

    return image, label


train_ds = train_ds.map(
    preprocess,
    num_parallel_calls=1
)

test_ds = test_ds.map(
    preprocess,
    num_parallel_calls=1
)

# Keep memory usage low
train_ds = train_ds.prefetch(1)

test_ds = test_ds.prefetch(1)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.08
    ),

    layers.RandomZoom(
        0.08
    ),

])


# ============================================================
# BUILD LIGHTWEIGHT CNN MODEL
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    # Data augmentation
    data_augmentation,


    # CONVOLUTION BLOCK 1

    layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # CONVOLUTION BLOCK 2

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # CONVOLUTION BLOCK 3

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # PREVENT OVERFITTING

    layers.Dropout(
        0.25
    ),


    # REDUCE PARAMETERS

    layers.GlobalAveragePooling2D(),


    # DENSE LAYER

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dropout(
        0.30
    ),


    # OUTPUT LAYER
    # Automatically supports 3 classes

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    # Stop training if validation loss stops improving

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True,

        verbose=1

    ),


    # Save best model

    tf.keras.callbacks.ModelCheckpoint(

        "best_medical_biomedical_normal_model.keras",

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1

    ),

]


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60)


history = model.fit(

    train_ds,

    validation_data=test_ds,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1

)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(

    "medical_biomedical_normal_classifier.keras"

)


print("\nFinal model saved successfully!")


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n" + "=" * 60)
print("TESTING MODEL")
print("=" * 60)


test_loss, test_accuracy = model.evaluate(

    test_ds,

    verbose=1

)


print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")


y_true = []

y_pred = []


for images, labels in test_ds:

    predictions = model.predict(

        images,

        verbose=0

    )

    predicted_classes = np.argmax(

        predictions,

        axis=1

    )

    y_true.extend(

        labels.numpy()

    )

    y_pred.extend(

        predicted_classes

    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)


print(

    classification_report(

        y_true,

        y_pred,

        labels=list(
            range(NUM_CLASSES)
        ),

        target_names=class_names,

        zero_division=0

    )

)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=list(
        range(NUM_CLASSES)
    )

)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    cm
)


plt.title(
    "Confusion Matrix"
)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


plt.xticks(

    range(NUM_CLASSES),

    class_names,

    rotation=30

)


plt.yticks(

    range(NUM_CLASSES),

    class_names

)


for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(

            j,

            i,

            str(
                cm[i][j]
            ),

            ha="center",

            va="center"

        )


plt.colorbar()

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.close()


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure()

plt.plot(

    history.history["accuracy"],

    label="Training Accuracy"

)

plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"

)


plt.title(
    "Training and Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "accuracy_graph.png"
)

plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure()

plt.plot(

    history.history["loss"],

    label="Training Loss"

)

plt.plot(

    history.history["val_loss"],

    label="Validation Loss"

)


plt.title(
    "Training and Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "loss_graph.png"
)

plt.close()


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

with open(

    "class_names.txt",

    "w"

) as file:

    for i, class_name in enumerate(class_names):

        file.write(

            f"{i}:{class_name}\n"

        )


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)


print("\nGenerated Files:")

print(
    "1. best_medical_biomedical_normal_model.keras"
)

print(
    "2. medical_biomedical_normal_classifier.keras"
)

print(
    "3. confusion_matrix.png"
)

print(
    "4. accuracy_graph.png"
)

print(
    "5. loss_graph.png"
)

print(
    "6. class_names.txt"
)


print("\nFinal Class Mapping:")

for i, class_name in enumerate(class_names):

    print(
        f"Class {i} -> {class_name}"
    )