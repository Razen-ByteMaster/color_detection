import argparse
import numpy as np
import pandas as pd
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Initialize argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args["image"]


# Load image and handle resizing
def load_image(img_path, max_dim=800):
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError("Could not load image from path:", img_path)

    h, w = original_img.shape[:2]
    scaling_factor = max_dim / max(h, w)

    if scaling_factor < 1:
        resized_img = cv2.resize(
            original_img,
            None,
            fx=scaling_factor,
            fy=scaling_factor,
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized_img = original_img.copy()

    return original_img, resized_img, scaling_factor


original_img, display_img, scale_factor = load_image(img_path)
clicked = False

# Load and preprocess color data
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv("colors.csv", names=index, header=None)

# Prepare RGB data for PCA
colors_rgb = csv[["R", "G", "B"]].values
scaler = StandardScaler().fit(colors_rgb)
scaled_rgb = scaler.transform(colors_rgb)

# Apply PCA (keeping all components)
pca = PCA(n_components=3)
pca_features = pca.fit_transform(scaled_rgb)
csv["PCA1"] = pca_features[:, 0]
csv["PCA2"] = pca_features[:, 1]
csv["PCA3"] = pca_features[:, 2]


# --- CORRECTED FUNCTION ORDER STARTS HERE ---
def getColorName(R, G, B):
    """Find closest color using PCA-transformed space"""
    input_rgb = np.array([[R, G, B]])
    scaled_input = scaler.transform(input_rgb)
    pca_input = pca.transform(scaled_input)[0]

    distances = np.linalg.norm(pca_features - pca_input, axis=1)
    closest_idx = np.argmin(distances)
    return csv.loc[closest_idx, "color_name"]


def calculate_accuracy():
    correct = 0
    total = len(csv)

    for i in range(total):
        R = csv.loc[i, "R"]
        G = csv.loc[i, "G"]
        B = csv.loc[i, "B"]

        predicted = getColorName(R, G, B)
        actual = csv.loc[i, "color_name"]

        if predicted == actual:
            correct += 1

    return (correct / total) * 100


# Print accuracy score at startup
print(f"Color Matching Accuracy: {calculate_accuracy():.2f}%")
# --- CORRECTED FUNCTION ORDER ENDS HERE ---


# Mouse callback function with coordinate scaling
def draw_function(event, x, y, flags, param):
    global b, g, r, xpos, ypos, clicked
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicked = True
        # Scale coordinates back to original image
        orig_x = int(x / scale_factor)
        orig_y = int(y / scale_factor)

        # Ensure coordinates stay within bounds
        orig_x = min(max(orig_x, 0), original_img.shape[1] - 1)
        orig_y = min(max(orig_y, 0), original_img.shape[0] - 1)

        b, g, r = original_img[orig_y, orig_x]
        b, g, r = int(b), int(g), int(r)
        xpos, ypos = x, y


# Set up GUI
cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_function)

# Main loop
while True:
    cv2.imshow("image", display_img)
    if clicked:
        # Create background rectangle on display image
        cv2.rectangle(display_img, (20, 20), (750, 60), (b, g, r), -1)

        # Generate text
        text = f"{getColorName(r, g, b)} R={r} G={g} B={b}"

        # Choose text color based on brightness
        text_color = (0, 0, 0) if (r + g + b) >= 600 else (255, 255, 255)

        # Put text on display image
        cv2.putText(
            display_img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2
        )

        clicked = False

    # Break loop on ESC key
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
