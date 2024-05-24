import time
import picamera
import zeroshot

# Initialize the ZeroShot Classifier
model_id = "b2f15922-e634-4d58-884f-d3583f4a280d"
classifier = zeroshot.Classifier(model_id)

# Offline model
classifier = zeroshot.Classifier("./meter-maid-model.json")

preprocess_fn = zeroshot.create_preprocess_fn()

# Desired class names
desired_classes = ["meter maid", "meter maidcar", "meter maid car"]

# Function to capture image from Raspberry Pi Camera
def capture_image():
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.start_preview()
        time.sleep(2)  # Camera warm-up time
        image_path = '/tmp/meter_maid.jpg'
        camera.capture(image_path)
    return image_path

def detect_meter_maid(image_path):
    image = zeroshot.numpy_from_file(image_path)
    prediction = classifier.predict_proba(preprocess_fn(image))
    classes = classifier.classes

    # Find the class with the highest probability
    max_prob_index = prediction.argmax()
    class_name = classes[max_prob_index]
    confidence = prediction[max_prob_index]

    return class_name, confidence

def main():
    while True:
        image_path = capture_image()
        class_name, confidence = detect_meter_maid(image_path)

        if class_name in desired_classes:
            print(f'Meter maid detected: {class_name} with confidence {confidence:.2%}')
        else:
            print(f'No meter maid detected. Detected: {class_name} with confidence {confidence:.2%}')

        # Pause for a while before capturing the next image
        time.sleep(5)

if __name__ == "__main__":
    main()
