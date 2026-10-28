import RPi.GPIO as GPIO
from time import sleep

SERVO1_PIN = 17
SERVO2_PIN = 27

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(SERVO1_PIN, GPIO.OUT)
GPIO.setup(SERVO2_PIN, GPIO.OUT)

servo1 = GPIO.PWM(SERVO1_PIN, 50)
servo2 = GPIO.PWM(SERVO2_PIN, 50)

servo1.start(0)
servo2.start(0)


def move_servo(servo, angle):

    # Convert 0-180 degree to PWM duty cycle
    duty = 2.5 + (angle / 180.0) * 10.0

    servo.ChangeDutyCycle(duty)

    sleep(0.7)

    # Stop PWM after movement
    servo.ChangeDutyCycle(0)



print("Servo 1 -> 0 degrees")
move_servo(servo1, 100)

sleep(1)

servo1.stop()

