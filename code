import cv2
import numpy as np

def detect_traffic_lights(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for traffic lights
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    green_lower = np.array([40, 100, 100])
    green_upper = np.array([80, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    
    # Create masks for each color
    red_mask = cv2.inRange(hsv, red_lower, red_upper) | cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # Find contours for each color
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours and annotate
    for contour in red_contours:
        cv2.drawContours(frame, [contour], -1, (0, 0, 255), 3)
        cv2.putText(frame, "Red", (contour[0][0][0], contour[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    for contour in green_contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
        cv2.putText(frame, "Green", (contour[0][0][0], contour[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    for contour in yellow_contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 255), 3)
        cv2.putText(frame, "Yellow", (contour[0][0][0], contour[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def main():
    cap = cv2.VideoCapture(0)  # Change to 'video.mp4' for file input

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = detect_traffic_lights(frame)
        cv2.imshow('Traffic Light Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
