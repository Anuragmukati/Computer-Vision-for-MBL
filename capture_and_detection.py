# Import necessary libraries
import cv2  # OpenCV for image and video processing
import os   # Operating system-related functions
from ultralytics import YOLO  # YOLOv8 model for object detection
import numpy as np  # NumPy for numerical operations

# Load YOLOv8 model
yolo_model = YOLO('yolov8x.pt')

# Define display width and height for the video feed
dispW = 640
dispH = 480

# 'flip' is used to flip the video feed; 2 means flipping both horizontally and vertically
flip = 2

# Open the default camera (camera index 0)
cam = cv2.VideoCapture(0)

# Create a directory to save screenshots if it doesn't already exist
screenshot_path = "screenshots"
if not os.path.exists(screenshot_path):
    os.makedirs(screenshot_path)

# Initialize a boolean variable 'paused' to control pausing the video feed
paused = False

# Enter into an infinite loop for continuously capturing and processing the video frames
while True:
    # Read a frame from the camera
    ret, frame = cam.read()

    # Display the live video feed if not paused
    if not paused:
        cv2.imshow('Live Feed', frame)

    # Wait for a key press for 1 millisecond
    key = cv2.waitKey(1)

    # Check if the 'q' key is pressed to exit the loop and close the program
    if key == ord('q'):
        break

    # Check if the 'c' key is pressed to capture a screenshot
    elif key == ord('c'):
        # Generate a unique filename for the screenshot
        screenshot_filename = os.path.join(screenshot_path, f"screenshot_{len(os.listdir(screenshot_path))}.png")
        
        # Save the screenshot as an image file
        cv2.imwrite(screenshot_filename, frame)
        
        # Print a message indicating the filename where the screenshot is saved
        print(f"Screenshot saved as {screenshot_filename}")
        
        # Set 'paused' to True to process the screenshot with YOLOv8
        paused = True

        if paused:
            # Load the screenshot as an image
            screenshot = cv2.imread(screenshot_filename)
            
            # Perform object detection on the screenshot using YOLOv8
            result = yolo_model(source=screenshot, classes=0, save=True)

    # Check if the 'p' key is pressed to toggle pause/unpause the video feed
    elif key == ord('p'):
        paused = not paused  # Toggle pause

# Release the camera resource
cam.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
