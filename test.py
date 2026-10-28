import cv2
import numpy as np
import tensorflow as tf

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "medical_biomedical_normal_classifier.keras"
CLASS_FILE = "class_names.txt"

IMG_SIZE = 128

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ============================================================
# LOAD CLASS NAMES
# ============================================================

class_names = []

with open(CLASS_FILE, "r") as file:

    for line in file:

        line = line.strip()

        if ":" in line:

            index, class_name = line.split(":", 1)

            class_names.append(class_name)

print("\nClasses:")

for i, name in enumerate(class_names):

    print(f"{i} -> {name}")

# ============================================================
# OPEN USB CAMERA
# ============================================================

print("\nOpening USB camera...")

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():

    print("ERROR: Cannot open USB camera")

    exit()

print("Camera started successfully!")
print("Press Q to quit.")

# ============================================================
# REAL-TIME CLASSIFICATION
# ============================================================

while True:

    ret, frame = camera.read()

    if not ret:

        print("ERROR: Cannot read camera frame")

        break

    # Copy frame for display
    display_frame = frame.copy()

    # ========================================================
    # PREPROCESS IMAGE
    # ========================================================

    image = cv2.resize(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    # OpenCV uses BGR, model expects RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Convert to float
    image = image.astype("float32")

    # Normalize
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    class_index = np.argmax(prediction)

    confidence = prediction[class_index] * 100

    predicted_class = class_names[class_index]

    # ========================================================
    # RESULT TEXT
    # ========================================================

    # Replace underscore with space
    display_name = predicted_class.replace(
        "_",
        " "
    )

    text = (
        f"{display_name}: "
        f"{confidence:.1f}%"
    )

    # ========================================================
    # DISPLAY MAIN RESULT
    # ========================================================

    cv2.putText(
        display_frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ========================================================
    # DISPLAY ALL CLASS PROBABILITIES
    # ========================================================

    y_position = 80

    for i, class_name in enumerate(class_names):

        probability = prediction[i] * 100

        display_class = class_name.replace(
            "_",
            " "
        )

        probability_text = (
            f"{display_class}: "
            f"{probability:.1f}%"
        )

        cv2.putText(
            display_frame,
            probability_text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        y_position += 35

    # ========================================================
    # SHOW CAMERA FEED
    # ========================================================

    cv2.imshow(
        "Waste Classification - Raspberry Pi",
        display_frame
    )

    # Press Q to exit

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLOSE CAMERA
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("\nCamera closed.")