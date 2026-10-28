import cv2
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO
from time import sleep


# ============================================================
#                     USER SETTINGS
# ============================================================

# ---------------- MODEL SETTINGS ----------------

MODEL_PATH = "medical_biomedical_normal_classifier.keras"
CLASS_FILE = "class_names.txt"

IMG_SIZE = 128


# ---------------- CAMERA SETTINGS ----------------

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ---------------- GPIO SETTINGS ----------------

# Servo 1 = Gate Open / Close
SERVO_GATE_PIN = 17

# Servo 2 = Waste Direction Selector
SERVO_SELECTOR_PIN = 27


# ============================================================
#               SERVO 1 - GATE ANGLES
# ============================================================
# Keep your already tested angles here

GATE_CLOSED_ANGLE = 100
GATE_OPEN_ANGLE = 0


# ============================================================
#             SERVO 2 - SELECTOR ANGLES
# ============================================================
# Keep your already tested angles here

BIOMEDICAL_ANGLE = 50
MEDICAL_ANGLE = 150
NORMAL_ANGLE = 100


# ============================================================
#                    TIMING SETTINGS
# ============================================================

SERVO_MOVE_DELAY = 0.7

# Extra waiting time after Servo 2 reaches position
SELECTOR_SETTLE_TIME = 1.0

# Time for waste to fall
GATE_OPEN_TIME = 2.0

# Extra waiting time after Servo 2 returns to normal
RETURN_SETTLE_TIME = 1.0


# ============================================================
#                 DETECTION SETTINGS
# ============================================================

# Minimum prediction confidence
CONFIDENCE_THRESHOLD = 85

# Same class must be detected this many consecutive times
# before the servo operation starts
REQUIRED_CONSECUTIVE_DETECTIONS = 5

# Number of normal frames required before allowing
# another waste object
NORMAL_FRAMES_TO_RESET = 10


# ============================================================
#                    GPIO SETUP
# ============================================================

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(SERVO_GATE_PIN, GPIO.OUT)
GPIO.setup(SERVO_SELECTOR_PIN, GPIO.OUT)


# Standard servo PWM frequency = 50 Hz

gate_servo = GPIO.PWM(SERVO_GATE_PIN, 50)
selector_servo = GPIO.PWM(SERVO_SELECTOR_PIN, 50)

gate_servo.start(0)
selector_servo.start(0)


# ============================================================
#                 SERVO MOVEMENT FUNCTION
# ============================================================

def move_servo(servo, angle):
    """
    Move a standard servo between 0 and 180 degrees.
    """

    # Safety limit
    angle = max(0, min(180, angle))

    # Convert angle to PWM duty cycle
    duty = 2.5 + (angle / 180.0) * 10.0

    servo.ChangeDutyCycle(duty)

    sleep(SERVO_MOVE_DELAY)

    # Stop PWM to reduce jitter
    servo.ChangeDutyCycle(0)


# ============================================================
#              INITIAL SERVO POSITIONS
# ============================================================

print("\nSetting initial servo positions...")

# Servo 1 ALWAYS starts closed
move_servo(
    gate_servo,
    GATE_CLOSED_ANGLE
)

# Servo 2 starts at normal/default position
move_servo(
    selector_servo,
    NORMAL_ANGLE
)

print("Initial positions set successfully.")


# ============================================================
#                       LOAD MODEL
# ============================================================

print("\nLoading AI model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
#                    LOAD CLASS NAMES
# ============================================================

class_names = []

with open(CLASS_FILE, "r") as file:

    for line in file:

        line = line.strip()

        if ":" in line:

            index, class_name = line.split(":", 1)

            class_names.append(
                class_name.strip()
            )


print("\nClass Mapping:")

for i, name in enumerate(class_names):

    print(
        f"Class {i} -> {name}"
    )


# ============================================================
#                       OPEN CAMERA
# ============================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


if not camera.isOpened():

    print(
        "ERROR: Cannot open USB camera"
    )

    gate_servo.stop()
    selector_servo.stop()

    GPIO.cleanup()

    exit()


print("\nCamera started successfully.")
print("Press Q to quit.")


# ============================================================
#               GET SERVO 2 SELECTOR ANGLE
# ============================================================

def get_selector_angle(detected_class):

    if detected_class == "biomedical-waste":

        return BIOMEDICAL_ANGLE


    elif detected_class == "medical_waste":

        return MEDICAL_ANGLE


    else:

        return NORMAL_ANGLE


# ============================================================
#                 WASTE SORTING FUNCTION
# ============================================================

def perform_sorting(detected_class):

    print("\n")
    print("=" * 60)
    print("STARTING SORTING PROCESS")
    print("Detected Class:", detected_class)
    print("=" * 60)


    # ========================================================
    # SAFETY CHECK
    # ========================================================
    # Normal must NEVER open Servo 1

    if detected_class not in [
        "biomedical-waste",
        "medical_waste"
    ]:

        print("Normal detected.")
        print("Gate remains CLOSED.")

        return


    # ========================================================
    # STEP 1: MOVE SERVO 2 FIRST
    # ========================================================

    selector_angle = get_selector_angle(
        detected_class
    )


    print(
        f"Servo 2 moving to {selector_angle} degrees"
    )


    move_servo(
        selector_servo,
        selector_angle
    )


    print(
        "Waiting for Servo 2 to reach position..."
    )


    sleep(
        SELECTOR_SETTLE_TIME
    )


    # ========================================================
    # STEP 2: OPEN SERVO 1
    # ========================================================

    print(
        "Servo 1 opening gate..."
    )


    move_servo(
        gate_servo,
        GATE_OPEN_ANGLE
    )


    # ========================================================
    # STEP 3: WAIT FOR WASTE TO FALL
    # ========================================================

    print(
        f"Gate open for {GATE_OPEN_TIME} seconds"
    )


    sleep(
        GATE_OPEN_TIME
    )


    # ========================================================
    # STEP 4: CLOSE SERVO 1
    # ========================================================

    print(
        "Servo 1 closing gate..."
    )


    move_servo(
        gate_servo,
        GATE_CLOSED_ANGLE
    )


    # ========================================================
    # STEP 5: RETURN SERVO 2 TO NORMAL POSITION
    # ========================================================

    print(
        f"Servo 2 returning to normal position "
        f"({NORMAL_ANGLE} degrees)..."
    )


    move_servo(
        selector_servo,
        NORMAL_ANGLE
    )


    print(
        "Waiting for Servo 2 to settle..."
    )


    sleep(
        RETURN_SETTLE_TIME
    )


    print(
        "Sorting completed successfully."
    )

    print("=" * 60)


# ============================================================
#                     SYSTEM STATES
# ============================================================

STATE_DETECTING = "DETECTING"

STATE_WAIT_FOR_NORMAL = "WAIT_FOR_NORMAL"


system_state = STATE_DETECTING

last_class = None

consecutive_count = 0

normal_frame_count = 0


# ============================================================
#                       MAIN LOOP
# ============================================================

try:

    while True:


        # ====================================================
        # READ CAMERA FRAME
        # ====================================================

        ret, frame = camera.read()


        if not ret:

            print(
                "ERROR: Camera frame not received"
            )

            break


        display_frame = frame.copy()


        # ====================================================
        # PREPROCESS IMAGE
        # ====================================================

        image = cv2.resize(
            frame,
            (IMG_SIZE, IMG_SIZE)
        )


        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        image = image.astype(
            "float32"
        )


        image = image / 255.0


        image = np.expand_dims(
            image,
            axis=0
        )


        # ====================================================
        # AI PREDICTION
        # ====================================================

        prediction = model.predict(
            image,
            verbose=0
        )[0]


        class_index = np.argmax(
            prediction
        )


        confidence = (
            prediction[class_index] * 100
        )


        predicted_class = class_names[
            class_index
        ]


        # ====================================================
        # DISPLAY MAIN RESULT
        # ====================================================

        result_text = (
            f"{predicted_class}: "
            f"{confidence:.1f}%"
        )


        cv2.putText(
            display_frame,
            result_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ====================================================
        # DISPLAY ALL CLASS PROBABILITIES
        # ====================================================

        y = 80


        for i, class_name in enumerate(class_names):

            probability = (
                prediction[i] * 100
            )


            probability_text = (
                f"{class_name}: "
                f"{probability:.1f}%"
            )


            cv2.putText(
                display_frame,
                probability_text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


            y += 30


        # ====================================================
        # STATE 1: DETECTING
        # ====================================================

        if system_state == STATE_DETECTING:


            status_text = "STATUS: DETECTING"


            # Only these two classes can activate sorting

            valid_waste = (
                predicted_class in [
                    "biomedical-waste",
                    "medical_waste"
                ]
            )


            # ------------------------------------------------
            # BIOMEDICAL OR MEDICAL DETECTED
            # ------------------------------------------------

            if (
                confidence >= CONFIDENCE_THRESHOLD
                and valid_waste
            ):


                # Check repeated detection

                if predicted_class == last_class:

                    consecutive_count += 1


                else:

                    last_class = predicted_class

                    consecutive_count = 1


                stable_text = (
                    f"Stable: "
                    f"{consecutive_count}/"
                    f"{REQUIRED_CONSECUTIVE_DETECTIONS}"
                )


                cv2.putText(
                    display_frame,
                    stable_text,
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )


                # ============================================
                # WASTE CONFIRMED
                # ============================================

                if (
                    consecutive_count >=
                    REQUIRED_CONSECUTIVE_DETECTIONS
                ):


                    print(
                        "\nWASTE CONFIRMED"
                    )


                    print(
                        "Detected:",
                        predicted_class
                    )


                    # Servo 2 moves first
                    # Servo 1 opens only after Servo 2 settles
                    # Servo 1 closes
                    # Servo 2 returns to NORMAL

                    perform_sorting(
                        predicted_class
                    )


                    # Now wait for object to disappear

                    system_state = (
                        STATE_WAIT_FOR_NORMAL
                    )


                    consecutive_count = 0

                    last_class = None

                    normal_frame_count = 0


            # ------------------------------------------------
            # NORMAL OR LOW CONFIDENCE
            # ------------------------------------------------

            else:


                # Reset detection counter

                consecutive_count = 0

                last_class = None


                # IMPORTANT:
                # Servo 1 remains CLOSED.
                # No gate operation happens here.


        # ====================================================
        # STATE 2: WAIT FOR NORMAL SCENE
        # ====================================================

        elif system_state == STATE_WAIT_FOR_NORMAL:


            status_text = (
                "STATUS: WAITING FOR NEXT OBJECT"
            )


            # Wait until the camera sees normal again

            if (
                predicted_class == "normal"
                and confidence >= CONFIDENCE_THRESHOLD
            ):

                normal_frame_count += 1


            else:

                normal_frame_count = 0


            ready_text = (
                f"Normal Check: "
                f"{normal_frame_count}/"
                f"{NORMAL_FRAMES_TO_RESET}"
            )


            cv2.putText(
                display_frame,
                ready_text,
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )


            # =================================================
            # READY FOR NEXT OBJECT
            # =================================================

            if (
                normal_frame_count >=
                NORMAL_FRAMES_TO_RESET
            ):


                print(
                    "\nNormal scene confirmed."
                )


                print(
                    "Ready for next waste."
                )


                # Servo 1 stays CLOSED
                # Servo 2 is already at NORMAL position

                system_state = STATE_DETECTING

                normal_frame_count = 0

                consecutive_count = 0

                last_class = None


        # ====================================================
        # DISPLAY SYSTEM STATUS
        # ====================================================

        cv2.putText(
            display_frame,
            status_text,
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "AI Waste Segregation System",
            display_frame
        )


        # ====================================================
        # PRESS Q TO EXIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


# ============================================================
#                       CLEANUP
# ============================================================

finally:


    print(
        "\nStopping system..."
    )


    # Safety: close gate

    move_servo(
        gate_servo,
        GATE_CLOSED_ANGLE
    )


    # Safety: return selector to normal

    move_servo(
        selector_servo,
        NORMAL_ANGLE
    )


    gate_servo.stop()

    selector_servo.stop()


    GPIO.cleanup()


    camera.release()

    cv2.destroyAllWindows()


    print(
        "System stopped safely."
    )

