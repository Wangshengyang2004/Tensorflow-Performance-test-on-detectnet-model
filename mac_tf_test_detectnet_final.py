import cv2
import tensorflow as tf
import time

# Load the TensorFlow model
model = tf.saved_model.load('/Users/simonwsy/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model')

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Set the video resolution to 720p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Start the timer
    start_time = time.time()

    # Capture frame-by-frame
    ret, img = cap.read()

    # Add an extra dimension to match the input shape
    input_tensor = tf.expand_dims(img, 0)

    # Perform object detection
    detections = model(input_tensor)

    # Draw bounding boxes and labels
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detection_classes = detections['detection_classes'].astype(int)
    detection_scores = detections['detection_scores']

    for i in range(num_detections):
        score = detection_scores[i]
        bbox = detections['detection_boxes'][i]

        if score > 0.5:
            # Calculate bounding box coordinates
            left, top, right, bottom = bbox[1] * img.shape[1], bbox[0] * img.shape[0], bbox[3] * img.shape[1], bbox[2] * img.shape[0]

            # Calculate the center and the radius of the circle
            centerX, centerY = int((left + right) / 2), int((top + bottom) / 2)
            radius = int(0.25 * max(right - left, bottom - top))  # Adjust the size of the circle here

            # Draw a circle around the detected object
            cv2.circle(img, (centerX, centerY), radius, (0, 0, 255), 2)

            # Draw the label
            label = f"{detection_classes[i]}: {score}"
            cv2.putText(img, label, (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Calculate FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the image
    cv2.imshow('Object Detection', img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()