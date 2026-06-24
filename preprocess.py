import cv2 as cv
import os

input_dir = r'C:\Users\LENOVO\Documents\ANA\Treca_godina\ADOS\fracture_classification\dataset'
outpu_dir = r'C:\Users\LENOVO\Documents\ANA\Treca_godina\ADOS\fracture_classification\dataset_CLAHE'

# create clahe object with standard medical parameters
# clipLimit=3.0 
# tileGridSize=(8,8) 
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

photos_num = 0

print("Start preprocessing")

# go through root folder (os.walk goes through all subfolders)
for root, folders, files in os.walk(input_dir):
    for file in files:
        # check if file is photo
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            
            # make path to original picture
            input_path = os.path.join(root, file)
            
            # make path for new folder
            realtive_path = os.path.relpath(root, input_dir)
            end_folder = os.path.join(outpu_dir, realtive_path)

            if not os.path.exists(end_folder):
                os.makedirs(end_folder)
                
            output_path = os.path.join(end_folder, file)
            
            # get image in grayscale
            photo = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
            
            if photo is not None:
                
                # filter with CLAHE
                processed_photo = clahe.apply(photo)
                
                # save new photo and add to folder
                cv.imwrite(output_path, processed_photo)
                photos_num += 1

print(f"Success! Processed {photos_num} photos and saved to folder: {outpu_dir}")

