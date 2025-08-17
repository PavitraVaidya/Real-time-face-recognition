import cv2
import os
import pandas as pd
import numpy as np
import shutil

# Paths
faces_folder = "faces"
details_file = "details.csv"
os.makedirs(faces_folder, exist_ok=True)

# Load Haar Cascade
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)

# LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load existing data
if os.path.exists(details_file):
    details_df = pd.read_csv(details_file)
else:
    details_df = pd.DataFrame(columns=["id", "name", "age", "address", "phone"])

# --- Register a new person ---
def register_person():
    global details_df
    person_id = len(details_df) + 1  
    name = input("Enter name: ")
    age = input("Enter age: ")
    address = input("Enter address: ")
    phone = input("Enter phone number: ")

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Could not access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            file_path = os.path.join(faces_folder, f"{person_id}_{count}.jpg")
            cv2.imwrite(file_path, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Register - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save details
    details_df.loc[len(details_df)] = [person_id, name, age, address, phone]
    details_df.to_csv(details_file, index=False)
    print(f"✅ Registered {name} with ID {person_id}")

def train_model():
    global details_df

    # Reload CSV before training
    if os.path.exists(details_file):
        details_df = pd.read_csv(details_file)
    else:
        print("⚠ No registered people found! Register first.")
        return

    # Get all valid image files from faces folder
    image_paths = [os.path.join(faces_folder, f) for f in os.listdir(faces_folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_paths:
        print("⚠ No images found! Register people first.")
        return

    face_samples = []
    ids = []

    for image_path in image_paths:
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            print(f"⚠ Could not read image: {image_path}")
            continue

        id_str = os.path.basename(image_path).split("_")[0]
        ids.append(int(id_str))
        face_samples.append(gray_img)

    if not face_samples:
        print("⚠ No valid face images found! Cannot train.")
        return

    recognizer.train(face_samples, np.array(ids))
    recognizer.save("trainer.yml")
    print("✅ Model trained and saved as trainer.yml")

def recognize_person():
    global details_df

    if not os.path.exists("trainer.yml"):
        print("⚠ Train the model first by registering people.")
        return

    # Reload details CSV in case it was deleted/modified
    if os.path.exists(details_file):
        details_df = pd.read_csv(details_file)
    else:
        details_df = pd.DataFrame(columns=["id", "name", "age", "address", "phone"])

    recognizer.read("trainer.yml")
    # ... rest of the function ...

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Could not access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id_pred, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            # --- Debug print ---
            print(f"Predicted ID: {id_pred}")

            # Adjust threshold if needed
            threshold = 100
            if confidence < threshold:
                matched = details_df[details_df["id"] == id_pred]

                if not matched.empty:
                    person = matched.iloc[0]

                    # Console output
                    print("\n--- Person Recognized ---")
                    print(f"ID: {person['id']}")
                    print(f"Name: {person['name']}")
                    print(f"Age: {person['age']}")
                    print(f"Address: {person['address']}")
                    print(f"Phone: {person['phone']}")
                  

                    # Webcam display
                    cv2.putText(frame, f"{person['name']} ({person['age']} yrs)", 
                                (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Addr: {person['address']}", 
                                (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Ph: {person['phone']}", 
                                (x+5, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Unknown", (x+5, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Unknown", (x+5, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Recognition - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def delete_all_data():
    global details_df

    # Delete faces folder
    if os.path.exists(faces_folder):
        shutil.rmtree(faces_folder, ignore_errors=True)
    os.makedirs(faces_folder, exist_ok=True)

    # Delete trained model
    if os.path.exists("trainer.yml"):
        os.remove("trainer.yml")

    # Delete CSV file
    if os.path.exists(details_file):
        os.remove(details_file)

    # Reset dataframe in memory
    details_df = pd.DataFrame(columns=["id", "name", "age", "address", "phone"])

    print("🗑 All stored information deleted successfully.")

# --- Main menu ---
while True:
    print("\n1. Register new person")
    print("2. Train model")
    print("3. Recognize person")
    print("4. Delete all stored data")
    print("5. Exit")
    choice = input("Enter choice: ")

    if choice == "1":
        register_person()
    elif choice == "2":
        train_model()
    elif choice == "3":
        recognize_person()
    elif choice == "4":
        delete_all_data()
    elif choice == "5":
        break
    else:
        print("Invalid choice!")