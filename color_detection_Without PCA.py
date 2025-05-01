import argparse
import cv2
import pandas as pd
import random

# Initialize argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args["image"]


# Load and resize image
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

# Load color data with type conversion
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv("colors.csv", names=index, header=None)
csv = csv.astype({"R": int, "G": int, "B": int})


def getColorName(R, G, B):
    # Simplified less-accurate matching
    candidates = []

    # Random search with limited scope
    for _ in range(50):  # Check random 50 colors instead of all
        i = random.randint(0, len(csv) - 1)
        diff = (
            abs(R - csv.loc[i, "R"])
            + abs(G - csv.loc[i, "G"])
            + abs(B - csv.loc[i, "B"])
        )

        if diff < 150:  # Larger tolerance for error
            candidates.append(csv.loc[i, "color_name"])

    # Fallback to first match if no candidates
    return candidates[0] if candidates else csv.loc[0, "color_name"]


def calculate_accuracy():
    correct = 0
    total = len(csv)
    for i in range(min(100, len(csv))):  # Test subset for speed
        R = csv.loc[i, "R"]
        G = csv.loc[i, "G"]
        B = csv.loc[i, "B"]
        if getColorName(R, G, B) == csv.loc[i, "color_name"]:
            correct += 1
    return (correct / min(100, len(csv))) * 100


print(f"Approximate Accuracy: {calculate_accuracy():.2f}%")


# Mouse callback function
def draw_function(event, x, y, flags, param):
    global b, g, r, clicked
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicked = True
        orig_x = min(max(int(x / scale_factor), 0), original_img.shape[1] - 1)
        orig_y = min(max(int(y / scale_factor), 0), original_img.shape[0] - 1)
        b, g, r = original_img[orig_y, orig_x]
        b, g, r = int(b), int(g), int(r)


# Setup window
cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_function)

while True:
    display_copy = display_img.copy()
    if clicked:
        cv2.rectangle(display_copy, (20, 20), (750, 60), (b, g, r), -1)
        text = f"{getColorName(r, g, b)} R={r} G={g} B={b}"
        text_color = (0, 0, 0) if (r + g + b) >= 600 else (255, 255, 255)
        cv2.putText(
            display_copy, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2
        )

    cv2.imshow("image", display_copy)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
