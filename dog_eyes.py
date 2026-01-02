import cv2
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def load_and_detect_eyes(num_images=10):
    """Load dog images and detect eyes using improved parameters"""
    dataset = tfds.load('oxford_iiit_pet:4.0.0', split='train')
    dogs = dataset.filter(lambda x: x['species'] == 1)
    
    eye_pairs = []
    for dog in tfds.as_numpy(dogs.take(num_images)):
        img = cv2.cvtColor(dog['image'], cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (400, 400))  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eye_rects = eyes.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=8,  
            minSize=(30, 30), 
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Process if exactly 2 eyes found
        if len(eye_rects) == 2:
            eye_rects = sorted(eye_rects, key=lambda x: x[0])
            eyes_cropped = []
            for (x,y,w,h) in eye_rects:
                eye = img[y:y+h, x:x+w]
                eye = cv2.resize(eye, (100, 100))
                eyes_cropped.append(eye)
            
            # Combine eyes horizontally
            eye_pair = np.hstack(eyes_cropped)
            eye_pairs.append(eye_pair)
    
    return eye_pairs

def display_results(eye_pairs):
    """Display detected eye pairs with error handling"""
    if not eye_pairs:
        print("\nERROR: No eyes detected. Possible solutions:")
        print("1. Try more images (some dogs may have closed eyes)")
        print("2. Use better detector (MTCNN/DLib animal model)")
        print("3. Manually verify 'haarcascade_eye.xml' exists at:", cv2.data.haarcascades)
        return
    
    plt.figure(figsize=(15, 5))
    for i, pair in enumerate(eye_pairs[:3]):
        plt.subplot(1, 3, i+1)
        plt.imshow(cv2.cvtColor(pair, cv2.COLOR_BGR2RGB))
        plt.title(f"Eye Pair {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Execution
print("Processing dog images...")
eye_pairs = load_and_detect_eyes(num_images=20)
display_results(eye_pairs)
