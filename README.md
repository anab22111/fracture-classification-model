# Fracture Classification – YOLO

A binary image classification project for X-ray bone images — **fractured** vs **not fractured** — using a YOLOv8 classification model (`yolov8n-cls`). Beyond basic training and inference, the project investigates how image preprocessing techniques (Gaussian blur and CLAHE) affect model robustness and accuracy, both on the validation set and on completely new images from the internet — including a healthy (non-fractured) bone image, used to evaluate model specificity.

## Dataset

The [**Bone Fracture Multi-Region X-ray Data**](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data) dataset from Kaggle was used.

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

> Python 3.x recommended (*specify the exact version used*).

## Training

The model is trained via `main.py`, which uses the YOLOv8 classification architecture (`yolov8n-cls.pt`) as a pretrained starting point.

```bash
python main.py
```

Hyperparameters used:

| Parameter | Value | Notes |
|---|---|---|
| `model` | `yolov8n-cls.pt` | nano variant – sufficient capacity for a binary problem (fractured / not fractured), no need for multi-class object detection |
| `epochs` | 20 | enough for convergence, since training is essentially fine-tuning of pretrained (ImageNet) layers |
| `imgsz` | 244 | input image size |

After training, `runs/` contains performance plots and the confusion matrix:
- **train/loss** decreases exponentially as expected (from ~0.4 down to ~0.01 by epoch 20), indicating an adequately chosen learning rate.
- **val/loss** follows the training curve's downward trend (a slight bump at epoch 2 is normal), with no divergence from the training loss — confirming no overfitting occurred.

The weights from the last epoch (`last.pt`, converted to `no_preprocessing_model.onnx`) were used for the inference stage.

## ONNX Conversion

```bash
python convert.py
```

Converting from PyTorch to ONNX allows the model to run in different environments (e.g. via the OpenCV DNN module, independent of Ultralytics/PyTorch dependencies).

## Inference

```bash
python predict.py --image path/to/image.jpg
```

Workflow in `predict.py`:
1. Load the `.onnx` model via `cv2.dnn.readNetFromONNX`.
2. Convert the input image into a blob (`cv2.dnn.blobFromImage`) – normalize pixel values to [0, 1] (divide by 255) and swap channel order (BGR → RGB, since OpenCV loads images in BGR while the model was trained on RGB images). The result is a 4D tensor (batch, channels, height, width).
3. Forward pass through the network (`net.setInput()` + `net.forward()`), returning a 1×2 matrix of class probabilities.
4. `np.argmax()` determines the predicted class, and the result (class + confidence percentage) is drawn on the image and shown via `cv2.imshow()`.

## Training Results (baseline model, no preprocessing)

- **Validation accuracy:** 99.75%
- **Confusion matrix:**
  - 335 correctly classified as *fractured*
  - 486 correctly classified as *not fractured*
  - 2 false negatives (fracture misclassified as healthy bone)

## Preprocessing Experiments

As an unavoidable initial step in every experiment, images were converted to grayscale, which serves as the basis for all further steps. Five scenarios were tested to determine the effect of Gaussian blur and CLAHE on the network's generalization ability. For scenarios 4 and 5, two new YOLOv8-cls models were fully retrained from scratch on the entire dataset, pre-transformed with the corresponding filters — this was not simply filtering at test time.

Evaluation was performed on three external internet images: `frac.jpg` (wrist fracture), `rib_fracture.jpeg` (rib fracture), and **`no_fracture.jpg` (healthy, non-fractured bone — used to test model specificity)**.

| Scenario | Training data | Preprocessing | Model accuracy | frac.jpg | rib_fracture.jpeg | no_fracture.jpg |
|---|---|---|---|---|---|---|
| 1 | Original data | none | 99.75% | 98.95% | 99.70% | 99.96% |
| 2 | Original data | Gaussian blur (inference only) | – | 92.76% | 99.95% | 99.96% |
| 3 | Original data | CLAHE (inference only) | – | 99.02% | 99.76% | **65.44%** |
| 4 | CLAHE-preprocessed data | CLAHE | 99.5% | 99.95% | 99.89% | 99.76% |
| 5 | Gaussian + CLAHE data | Gaussian + CLAHE | 99.27% | 91.82% | 99.94% | 99.90% |

### Observations per scenario

- **Scenario 2 (Gaussian blur, inference only):** Gaussian blur (3×3 kernel) smooths out the texture needed to detect a fracture in `frac.jpg` (drop to 92.76%), while slightly improving the result on `rib_fracture.jpeg` by removing soft-tissue background noise.
- **Scenario 3 (CLAHE, inference only) – critical finding:** Although CLAHE slightly improves detection on fractured images, on the healthy image (`no_fracture.jpg`) the model's confidence that no fracture is present drops drastically to only **65.44%**. Since the model was never trained on CLAHE-processed images, the aggressively enhanced local texture and shadows on the healthy joint get misinterpreted as potential fracture lines, leading to an unstable prediction. This demonstrates that CLAHE must not be applied only at inference/production time without the network first being trained on similarly modified images.
- **Scenario 4 (CLAHE, training + inference):** The most stable scenario — despite 2 additional misclassifications in the confusion matrix compared to the baseline (statistically negligible on a dataset of 800+ images), the model maintains high, consistent confidence across all three external images, including the healthy bone (99.76%).
- **Scenario 5 (Gaussian + CLAHE, training + inference):** Combining both filters does not improve on CLAHE alone — Gaussian blur still smooths out the fine edges needed to detect the fracture in `frac.jpg` (91.82%), even when the model was trained on such images.

### How filter effect depends on image type

The same filter does not affect every type of X-ray identically:

- **Wrist X-ray (`frac.jpg`)** – dominated by micro-texture and sharp edges (the fracture appears as a thin line). Extremely sensitive to Gaussian blur, which removes the high-frequency detail needed for detection.
- **Rib X-ray (`rib_fracture.jpeg`)** – dominated by macro-structure and soft-tissue noise. Gaussian blur here does not destroy fracture information but instead removes background noise, so CLAHE alone keeps stable, high results across all scenarios.
- **Healthy joint (`no_fracture.jpg`)** – reveals the biggest flaw of *partial* filter application: CLAHE applied only at inference time (scenario 3) artificially enhances the internal texture of healthy bone, creating false "lines" that the untrained network misreads as a fracture. Once the model is trained on CLAHE data from the start (scenario 4), it successfully distinguishes enhanced healthy texture from an actual fracture.

## Conclusion and Optimal Model Selection

Based on the five tested scenarios, **the scenario 4 model** (CLAHE applied during both training and inference) was selected as optimal:

- Highest confidence when a fracture is actually present: 99.95% (frac.jpg), 99.89% (rib_fracture.jpeg).
- High confidence that no fracture is present in the healthy bone: 99.76% (no_fracture.jpg) — unlike scenario 3, which collapses to 65.44% on the same image type.
- In medical diagnostics, a missed fracture (false negative) carries more serious clinical consequences than mild uncertainty on a healthy sample (which is easily resolved with an additional check), so prioritizing maximum sensitivity while keeping reasonable specificity is preferred — and scenario 4 achieves this better than the alternatives.
- The baseline model and Gaussian blur (scenarios 1, 2, 5) show seemingly high percentages on the healthy image, but Gaussian blur degrades performance on real fractures (drop to as low as 91.82%), making it clinically unreliable.
- Scenario 3 was rejected because it demonstrates that CLAHE must not be applied only in production without the model first being trained on similarly modified data.