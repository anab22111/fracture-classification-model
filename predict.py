import cv2 as cv
import numpy as np

# get photo path
test_photo_path = r'C:\Users\LENOVO\Documents\ANA\Treca_godina\ADOS\fracture_classification\56-rotated3-rotated3.jpg'
# get model path
model_path = r'C:\Users\LENOVO\Documents\ANA\Treca_godina\ADOS\fracture_classification\CLAHE_model.onnx'

# load onnx model with opencv
net = cv.dnn.readNetFromONNX(model_path)

# load photo
photo = cv.imread(test_photo_path)

# make gray photo - base for all filters
gray_photo = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)

# Gaus blur
blur_photo = cv.GaussianBlur(gray_photo, (3, 3), 0)

# CLAHE filter
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
clahe_photo = clahe.apply(gray_photo)

processed_photo = clahe_photo  

# convert back to 3 channels because blobFromImage expects it 
photo_for_model = cv.cvtColor(processed_photo, cv.COLOR_GRAY2BGR)

photo_size = (256, 256)

# normalize and pack to blob
blob = cv.dnn.blobFromImage(photo_for_model, scalefactor=1.0/255.0, size=photo_size, swapRB=True, crop=False)

# send photo to model and start prediction
net.setInput(blob)
output = net.forward()   # get output

# output is matrix - 1 row, 2 columns
percentages = output[0]

index = np.argmax(percentages)
safety = percentages[index] * 100

results = ["Fractured", "Not fractured"]
result_text = f"{results[index]}: {safety:.2f}%"

# get photo size 
h, w = photo_for_model.shape[:2]

font_scale = w / 1000.0  # scale the font size for photo
font_thickness = max(1, int(w / 400)) 
text_x = int(w * 0.03)  
text_y = int(h * 0.05)  

cv.putText(photo_for_model, result_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)

max_dim = 700

if max_dim < max(h, w):
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
else:
    new_w = w
    new_h = h

# create flexible window
cv.namedWindow("Rezultat Detekcije", cv.WINDOW_NORMAL)
# set dimensins for window
cv.resizeWindow("Rezultat Detekcije", new_w, new_h)

cv.imshow("Rezultat Detekcije", photo_for_model)

print(f"Model says -> {result_text}")

cv.waitKey(0)
cv.destroyAllWindows()