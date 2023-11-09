# Import the required libraries
import cv2
import numpy as np

# Define camera calibration parameters (you need to calibrate your camera beforehand)
focal_length = 1000  # Focal length of the camera in pixels
baseline = 1.0  # Baseline distance between the cameras in arbitrary units

# Define a list of image paths to process (add the paths to your images)
image_paths = ['IMG_20230323_120115.jpg']

def detect_persons(image_path):
    """
    Detect persons in an image using a pre-trained MobileNet SSD model.

    Parameters:
    - image_path (str): Path to the input image.

    Returns:
    - person_coordinates (list of tuples): Bounding box coordinates of detected persons.
    """

    # Load the pre-trained MobileNet SSD model
    model = cv2.dnn.readNetFromCaffe(
        'pi-object-detection-master/MobileNetSSD_deploy.prototxt.txt',
        'pi-object-detection-master/MobileNetSSD_deploy.caffemodel'
    )

    # Load the input image
    image = cv2.imread(image_path)

    # Get the dimensions of the image
    h, w = image.shape[:2]

    # Preprocess the image for the model
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    # Set the input to the model
    model.setInput(blob)

    # Perform object detection
    detections = model.forward()

    person_count = 0
    person_coordinates = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        # If the detected object is a person and confidence is high enough
        if class_id == 15 and confidence > 0.5:
            person_count += 1

            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype(int)

            # Store the bounding box coordinates
            person_coordinates.append((startX, startY, endX, endY))

    return person_coordinates

list_1 = []

# Process each image in the list
for image_path in image_paths:
    # Detect persons in the image and get their bounding box coordinates
    person_coordinates = detect_persons(image_path)

    # Calculate distances for each detected person
    for coords in person_coordinates:
        x_min, y_min, x_max, y_max = coords

        # Calculate image coordinates of person's center
        person_center_x = (x_min + x_max) / 2
        person_center_y = (y_min + y_max) / 2

        # Calculate angles (in radians) using camera properties
        angle_x = np.arctan((person_center_x - image.shape[1] / 2) / focal_length)
        angle_y = np.arctan((person_center_y - image.shape[0] / 2) / focal_length)

        # Calculate distance using triangulation
        distance = baseline / np.tan(angle_x)  # You can use angle_y if desired

        # Ensure the distance is positive
        distance = abs(distance)

        # Append the calculated distance to the list
        list_1.append(distance)

# Sort the list of distances in ascending order
list_1.sort()

def find_median(lst):
    """
    Calculate the median of a list of values.

    Parameters:
    - lst (list of float): List of values.

    Returns:
    - median (float): The median value.
    """
    # Sort the list in ascending order
    lst.sort()

    # Check if the list has an odd or even number of elements
    n = len(lst)
    if n % 2 == 1:
        # If odd, return the middle element
        return lst[n // 2]
    else:
        # If even, return the average of the two middle elements
        middle1 = lst[(n - 1) // 2]
        middle2 = lst[n // 2]
        return (middle1 + middle2) / 2

# Calculate the median of the list of distances
final_result = find_median(list_1)

# Print the approximate range in meters
print('Approximate range is', round(final_result + 0.5, 1), 'meters')
