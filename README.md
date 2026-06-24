# Fracture Classification – YOLO

A binary image classification project for X-ray bone images — **fractured** vs **not fractured** — using a YOLOv8 classification model (`yolov8n-cls`). Beyond basic training and inference, the project investigates how image preprocessing techniques (Gaussian blur and CLAHE) affect model robustness and accuracy, both on the validation set and on completely new images taken from the internet.

## Dataset

The **Bone Fracture Multi-Region X-ray Data** dataset was used.
🔗 *(add dataset link here)*

The data is organized into two classes:
- `fractured/`
- `not_fractured/`

split into `train` and `val` subdirectories.

## Project Structure

```
fracture_classification/
│
├── dataset/
│   ├── train/
│   │   ├── fractured/
│   │   └── not_fractured/
│   └── val/
│       ├── fractured/
│       └── not_fractured/
│
├── runs/              # auto-generated after training
│                      # (performance plots, model weights, confusion matrix)
├── models/
│   ├── no_preprocessing_model.onnx   
│   ├── CLAHE_model.onnx
│   ├── CLAHE_Gaus_model.onnx
│   ├── yolov8n-cls.pt
│   
│   
├── src/
│   ├── main.py            # training configuration and execution
│   ├── convert.py          # converts .pt -> .onnx model
│   ├── predict.py          # loads the ONNX model and runs inference on new 
│   ├── preprocess.py       # preprocesses training set og photos
├── 
├── test_pictures
│   ├── frac.jpg
│   ├── rib_fracture.png
│   ├── no_fracture.jpeg
```

## Installation

```bash
pip install ultralytics opencv-python onnx numpy
```

## Training

The model is trained via `main.py`, which uses the YOLOv8 classification architecture (`yolov8n-cls.pt`) as a pretrained starting point.

```bash
python main.py
```

Hyperparameters used:

| Parameter | Value | Notes |
|---|---|---|
| `model` | `yolov8n-cls.pt` | nano variant – sufficient capacity for a binary problem, faster training, lower overfitting risk |
| `epochs` | 20 | enough for convergence, since training is essentially fine-tuning of pretrained (ImageNet) layers |
| `imgsz` | 224 | standard input size for YOLOv8-cls |

After training, `runs/` contains performance plots, the confusion matrix, and saved weights (`best.pt`, `last.pt`). The weights from the last epoch (`last.pt`) were used for the inference stage.

## ONNX Conversion

```bash
python convert.py
```

Generates `last.onnx`, used for inference via the OpenCV DNN module -> `no_preprocessing_model.onnx`(independent of the PyTorch/Ultralytics environment).

## Inference

```bash
python predict.py 
```

Workflow in `predict.py`:
1. Load the `.onnx` model via `cv2.dnn.readNetFromONNX`.
2. Convert the input image into a blob (`cv2.dnn.blobFromImage`) – pixel normalization, resizing to 224×224, BGR → RGB channel swap.
3. Forward pass through the network (`net.forward()`), returning a 1×2 matrix of class probabilities.
4. `np.argmax()` determines the predicted class, and the result is drawn on the image along with the model's confidence percentage.

## Training Results (baseline model, no preprocessing)

- **Validation accuracy:** 99.75%
- **Confusion matrix:**
  - 335 correctly classified as *fractured*
  - 486 correctly classified as *not fractured*
  - 2 false negatives (fracture misclassified as healthy bone)

## Preprocessing Experiments

Five scenarios were tested to assess the impact of Gaussian blur and CLAHE on classification, using two external images from the internet (`frac.jpg` – wrist fracture, `rib_fracture.jpeg` – rib fracture):

| Scenario | Training data | Preprocessing (inference) | Model accuracy | frac.jpg | rib_fracture.jpeg |
|---|---|---|---|---|---|
| 1 | Original data | none | 99.75% | 98.95% | 99.70% |
| 2 | Original data | Gaussian blur | – | 92.76% | 99.95% |
| 3 | Original data | CLAHE | – | 99.02% | 99.76% |
| 4 | CLAHE-preprocessed data | CLAHE | 99.5% | 99.95% | 99.89% |
| 5 | Gaussian + CLAHE data | Gaussian + CLAHE | 99.27% | 91.82% | 99.94% |

### Conclusion

Consistently applying CLAHE — both during training and inference (**scenario 4**) — produces the best and most stable generalization to new, external images, despite a slightly lower validation accuracy compared to the baseline model. Applying a filter only at inference time, without training the model on similarly preprocessed images (scenarios 2 and 3), causes a train/inference distribution mismatch — most pronounced with Gaussian blur, which removes the fine edges needed to detect a fracture line. Combining Gaussian blur and CLAHE during training (**scenario 5**) provides no additional benefit over CLAHE alone, confirming that blurring carries a risk of losing relevant detail, regardless of being combined with other techniques.

It was also observed that the same filter does not affect both external images identically — for example, Gaussian blur worsens the result on `frac.jpg` but slightly improves the result on `rib_fracture.jpeg`. This suggests that the effect of preprocessing depends on the characteristics of the specific image (noise level, contrast, and the size/nature of the structure of interest).


