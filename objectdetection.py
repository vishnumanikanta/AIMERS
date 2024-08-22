import cv2
import ultralytics
from ultralytics import YOLO

# Load the pretrained YOLOv8n model
model = YOLO("path for pt file")

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize variables for object tracking
tracker = None
initBB = None

# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.5

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # If we are not tracking any object, perform detection
    if initBB is None:
        # Run inference on the frame
        results = model(frame)

        # Process results and initialize tracker
        for result in results:
            for box in result.boxes:
                # Extract the confidence score
                confidence = box.conf

                # Check if the detection meets the confidence threshold
                if confidence > CONFIDENCE_THRESHOLD:
                    # Extract the coordinates and convert them to integers
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())

                    # Print out the coordinates of the detected bounding box
                    print("Detected bounding box:", x_min, y_min, x_max, y_max)

                    # Initialize the tracker with the first bounding box
                    initBB = (x_min, y_min, x_max - x_min, y_max - y_min)
                    tracker = cv2.TrackerCSRT_create()  # You can choose other trackers as well
                    tracker.init(frame, initBB)
                    break  # We assume only one object to track

    # If we are already tracking an object, update the tracker
    if initBB is not None:
        success, box = tracker.update(frame)
        if success:
            # Extract the coordinates from the box
            x, y, w, h = map(int, box)

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Optionally, draw the label
            label = f"Tracking x :{x} y {y} w: {w} h: {h}"
            #print(f"x: {x}, y: {y}, w: {w}, h: {h}")

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # If tracking fails, reset the tracker
            initBB = None

    # Display the resulting frame
    cv2.imshow("Webcam", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
