import cv2
import numpy as np

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
# Provide the theta for orientation
particles = np.zeros((N, 3))  
# Likelihood decay factor
a = 2

# Likelihood function
def calculateObservationLikelihood(i0, j0, theta0, i, j, theta):
    d = np.sqrt((i - i0) ** 2 + (j - j0) ** 2)
    dtheta = abs(theta - theta0)
    f = np.exp(-a * d) * np.exp(-a * dtheta)
    return f

# Cycle over the frames
CYCLES = 291
for k in range(2, CYCLES+1):
    if k <= 9:
        pic = f"000{k}.jpg"
    elif k <= 99:
        pic = f"00{k}.jpg"
    else:
        pic = f"0{k}.jpg"

    # Read the frame
    frame = cv2.imread(pic)
    if frame is None:
        print(f"Cannot read frame: {pic}")
        continue

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
        _, _, angle = rect

        # Draw the rectangle
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    # Place particles based on likelihood
    if max_contour is not None:
        # Calculate likelihood for each particle
        likelihoods = []
        for i in range(N):
            x = int(particles[i, 0])
            y = int(particles[i, 1])
            theta = particles[i, 2]
            likelihood = calculateObservationLikelihood(object_x, object_y, angle, x, y, theta)
            likelihoods.append(likelihood)

        # Normalize likelihoods
        likelihoods = np.array(likelihoods)
        if np.sum(likelihoods) != 0:
            likelihoods /= np.sum(likelihoods)
        else:
            likelihoods = np.full((N,), 1/N)

        # If sum isn't exactly 1
        if not np.isclose(np.sum(likelihoods), 1.0):
            likelihoods /= np.sum(likelihoods)

        # Resample particles based on likelihoods
        indices = np.random.choice(np.arange(N), size=N, p=likelihoods)
        particles = particles[indices]

    # Update particles
    for i in range(N):
        dx, dy = np.random.normal(0, sigma, 2)
        # Orientation
        dtheta = np.random.normal(0, sigma)
        particles[i, 0] += dx
        particles[i, 1] += dy
        particles[i, 2] += dtheta

    # Draw particles and the orientation of them 
    for particle in particles:
        x, y, theta = particle
        cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
        end_x = int(x + 10 * np.cos(theta))
        end_y = int(y + 10 * np.sin(theta))
        cv2.line(frame, (int(x), int(y)), (end_x, end_y), (0, 0, 255), 2)

    # Display
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()