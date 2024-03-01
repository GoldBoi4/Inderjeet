import cv2
import mediapipe as mp
import pyttsx3
import time
from tensorflow.keras.models import load_model
import numpy as np

class YogaTrainer:
    def __init__(self, model_path):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 140)
        self.model = load_model(model_path)
        self.correct_pose_threshold = 0.6
        self.yoga_steps = {
            "Mountain Pose": {
                "instructions": [
                    "Let's Learn Mountain Pose.",
                    "Stand tall with feet together.",
                    "Arms at sides, palms facing forward.",
                    "Relax your shoulders.",
                    "Lengthen your spine.",
                    "Breathe deeply and hold for a few seconds."
                ]
            },
            "Downward-Facing Dog": {
                "instructions": [
                    "Let's Start on your hands and knees.",
                    "Lift your hips up and back, forming an inverted V-shape with your body.",
                    "Keep your arms and legs straight.",
                    "Press your palms and heels into the ground.",
                    "Breathe deeply and hold for a few seconds."
                ]
            },
            "Tree Pose": {
                "instructions": [
                    "Let's Learn Tree Pose!",
                    "Stand on one leg and place the sole of the other foot against the inner thigh or calf of the standing leg.",
                    "Engage your core for balance.",
                    "Bring your palms together at your chest or extend them overhead.",
                    "Focus on a point in front of you to help with balance.",
                    "Breathe deeply and hold for a few seconds."
                ]
            },
            "Warrior Pose": {
                "instructions": [
                    "Step one foot back and bend your front knee, keeping your back leg straight.",
                    "Raise your arms overhead or extend them out to the sides.",
                    "Keep your torso upright and facing forward.",
                    "Engage your core and breathe deeply.",
                    "Hold for a few seconds and then switch sides."
                ]
            }
        }

    def detect_pose(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(image_rgb)
        return self.results.pose_landmarks

    def draw_pose(self, image):
        if self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

    def classify_pose(self, pose_landmarks):
        
        prediction = np.random.randint(0, 2)

        return prediction

    def teach_yoga_step(self, step_name):
        instructions = self.yoga_steps.get(step_name, {}).get("instructions", [])
        if instructions:
            for instruction in instructions:
                self.engine.say(instruction)
                self.engine.runAndWait()
                time.sleep(3)
        else:
            self.engine.say("Yoga step not found.")
            self.engine.runAndWait()

def main():
    cap = cv2.VideoCapture(0)
    model_path = "model.h5"  # Path to your pre-trained model
    yoga_trainer = YogaTrainer(model_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose_landmarks = yoga_trainer.detect_pose(frame)
        yoga_trainer.draw_pose(frame)

        cv2.putText(frame, "Press 't' to teach yoga step", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Yoga Trainer", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('t'):
            cv2.putText(frame, "Enter yoga step name: ", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Yoga Trainer", frame)
            step_name = input("Enter yoga step name: ")
            prediction = yoga_trainer.classify_pose(pose_landmarks)
            if prediction == 1:
                yoga_trainer.teach_yoga_step(step_name)
            else:
                yoga_trainer.engine.say("Incorrect pose detected. Please adjust your pose.")
                yoga_trainer.engine.runAndWait()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
