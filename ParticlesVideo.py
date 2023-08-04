import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture("sevenup.m4v")
cv2.namedWindow("Video")

# Define parameters
object_width = 100
object_height = 100
canny_threshold1 = 100
canny_threshold2 = 100

# Initializations
object_x = 0
object_y = 0

# Standard deviation
sigma = 10

# Initialize particles
N = 100
particles = np.zeros((N, 2))

# Likelihood decay factor
a = 2

# Likelihood function
def calculateObservationLikelihood(i0, j0, i, j):
    d = np.sqrt((i - i0) ** 2 + (j - j0) ** 2)
    f = np.exp(-a * d)
    return f

# Loop over the video
while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Fit a rectangle to the largest contour
    if max_contour is not None:
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Update the object position
        object_x, object_y, object_width, object_height = cv2.boundingRect(max_contour)

        # Draw the rectangle
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    # Place particles based on likelihood
    if max_contour is not None:
        # Calculate likelihood for each particle
        likelihoods = []
        for i in range(N):
            x = int(particles[i, 0])
            y = int(particles[i, 1])
            likelihood = calculateObservationLikelihood(object_x, object_y, x, y)
            likelihoods.append(likelihood)

        # Normalize likelihoods
        likelihoods = np.array(likelihoods)
        likelihoods /= np.sum(likelihoods)

        # Resample particles based on likelihoods
        indices = np.random.choice(np.arange(N), size=N, p=likelihoods)
        particles = particles[indices]

    # Update particles
    for i in range(N):
        dx, dy = np.random.normal(0, sigma, 2)
        particles[i, 0] += dx
        particles[i, 1] += dy

    # Draw particles
    for particle in particles:
        cv2.circle(frame, (int(particle[0]), int(particle[1])), 3, (255, 0, 0), -1)

    # Display
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()