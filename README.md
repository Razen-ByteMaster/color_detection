# Color Detection with PCA using OpenCV and scikit-learn

Detect and name the color of any pixel in an image by projecting known colors into a PCA space. Double-click on the displayed image to view its nearest color name, RGB values, and the model’s matching accuracy.

---

## 📌 Features
- **Image loading & resizing**: Reads large images and scales them for display  
- **Color dataset**: Loads a CSV of common colors (RGB + names)  
- **Data preprocessing**: Standardizes RGB values and applies PCA  
- **Color matching**: Finds the closest color via Euclidean distance in PCA space  
- **Interactive overlay**: Displays color name & RGB values on double-click  
- **Accuracy reporting**: Computes overall color-matching accuracy  

---

## 🛠️ Dependencies
- Python 3.x  
- OpenCV (`cv2`)  
- NumPy  
- Pandas  
- scikit-learn  

Install via:
```bash
pip install opencv-python numpy pandas scikit-learn
```

### Prepare files  
Place the following in the same directory:  
- `color_pca_detector.py`  
- `colors.csv`  

---

### Run the script  
```bash
python color_pca_detector.py --image <path_to_image>
