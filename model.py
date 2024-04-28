from tensorflow.keras.models import model_from_json
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np

model = model_from_json(open("model.json", "r").read())
model.load_weights('model.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to preprocess an image for the model (adjust based on your model's requirements)
def preprocess_image(image):
    resized_image = cv2.resize(image, (48, 48))  # Resize to model input size
    image_pixels = img_to_array(resized_image)  # Convert to NumPy array
    image_pixels = np.expand_dims(image_pixels, axis=0)  # Add a new dimension
    image_pixels /= 255.0  # Normalize to 0-1 range
    return image_pixels

# Function to detect emotions (replace with your model's specific logic)
def predict_emotion(model, image_pixels):
    predictions = model.predict(image_pixels)  # Make predictions
    max_index = np.argmax(predictions[0])  # Find class with highest probability
    emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    emotion_prediction = emotion_detection[max_index]
    return emotion_prediction

# Load the image
image = cv2.imread('happy1.jpg')

# Check if image is loaded successfully
if image is None:
    print("Error: Image not found. Please check the image path.")
    exit()


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# Loop through detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
    roi_gray = gray_image[y - 5:y + h + 5, x - 5:x + w + 5]
    roi_image = preprocess_image(roi_gray)  # Preprocess the ROI

    emotion = predict_emotion(model, roi_image)  # Predict emotion

    # Display prediction on the frame
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2
    lable_color = (10, 10, 255)
    textX = int((image.shape[1] - len(emotion) * 13) / 2)  # Adjust text position based on emotion length
    textY = y - 10
    cv2.putText(image, emotion, (textX, textY), FONT, FONT_SCALE, lable_color, FONT_THICKNESS)

# Display the processed image
cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()