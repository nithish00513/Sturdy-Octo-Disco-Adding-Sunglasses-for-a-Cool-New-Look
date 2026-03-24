# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!

## Program :
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load face image
faceImage = cv2.imread('Ani.jpg')
plt.imshow(faceImage[:,:,::-1])
plt.title("Face")
plt.axis('off')
```
<img width="373" height="565" alt="image" src="https://github.com/user-attachments/assets/d1e403eb-42ea-44c8-83f9-4debd6ba15bc" />

```
# Load sunglasses PNG (with transparency if available)
glassPNG = cv2.imread("sunn.png", cv2.IMREAD_UNCHANGED)
print("Image shape:", glassPNG.shape)

# Separate BGR and Alpha mask
if glassPNG.shape[2] == 4:
    glassBGR = glassPNG[:, :, :3]
    glassMask1 = glassPNG[:, :, 3]
else:
    glassBGR = glassPNG
    glassMask1 = 255 * np.ones(glassPNG.shape[:2], dtype=np.uint8)

plt.imshow(cv2.cvtColor(glassBGR, cv2.COLOR_BGR2RGB))
plt.title("Sunglasses")
plt.axis('off')
plt.show()
```
<img width="641" height="311" alt="image" src="https://github.com/user-attachments/assets/bbaf5f29-ba9a-470b-8dd6-10f08ffe70ee" />

```
plt.figure(figsize=[12,5])

plt.subplot(1,2,1)
plt.imshow(glassBGR[:,:,::-1])
plt.title('Sunglass Color channels')
plt.axis('off')

glassGray = cv2.cvtColor(glassBGR, cv2.COLOR_BGR2GRAY)
_, glassMask1 = cv2.threshold(glassGray, 240, 255, cv2.THRESH_BINARY_INV)

plt.subplot(1,2,2)
plt.imshow(glassMask1, cmap='gray')
plt.title('Sunglass Mask (generated)')
plt.axis('off')
```
<img width="1123" height="264" alt="image" src="https://github.com/user-attachments/assets/16b68772-05ff-41cb-acb7-ba8588f4901c" />

```
import mediapipe as mp

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

rgb_img = cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_img)
```
```
h, w, _ = faceImage.shape

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
        x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

        eye_width = abs(x2 - x1)
        new_w = int(eye_width * 2.0)
        new_h = int(new_w * glassBGR.shape[0] / glassBGR.shape[1])

        glass_resized = cv2.resize(glassBGR, (new_w, new_h))
        mask_resized = cv2.resize(glassMask1, (new_w, new_h))

        x = x1 - int(new_w * 0.25)
        y = y1 - int(new_h * 0.4)

        if x < 0: x = 0
        if y < 0: y = 0
        if x + new_w > w: new_w = w - x
        if y + new_h > h: new_h = h - y

        roi = faceImage[y:y+new_h, x:x+new_w]
        mask_resized = mask_resized[:new_h, :new_w]
        glass_resized = glass_resized[:new_h, :new_w]

        mask_inv = cv2.bitwise_not(mask_resized)
        mask_resized = cv2.merge([mask_resized]*3)
        mask_inv = cv2.merge([mask_inv]*3)

        bg = cv2.bitwise_and(roi, mask_inv)
        fg = cv2.bitwise_and(glass_resized, mask_resized)
        combined = cv2.add(bg, fg)

        faceImage[y:y+new_h, x:x+new_w] = combined
```
```
plt.figure(figsize=[10,10])
plt.imshow(cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))
plt.title("Face with Sunglasses (Auto Aligned)")
plt.axis("off")
plt.show()
```
<img width="519" height="704" alt="image" src="https://github.com/user-attachments/assets/6bd52c8b-0400-41ad-830d-e363227758ff" />







