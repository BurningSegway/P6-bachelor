from ttictoc import tic,toc
from PIL import Image
from ultralytics import YOLO

# Assuming your model file is in the current working directory, or provide the full path
tic()
print('### Loading checkpoint... ###')
model_path = 'YOLO/YOLO kode og data eksempel/model_- 21 march 2024 16_19.pt'
print(toc())

# Load your custom model
tic()
print('### Loading model... ###')
model = YOLO(model_path)
print(toc())

# Now, predict on an image. You can use a URL as in the example or a local file path.
# For a local file, just replace the URL with the path to your image file, like 'path/to/your/image.jpg'
tic()
print('### Predicting... ###')
results = model(['YOLO/YOLO kode og data eksempel/Image_1.jpg',
                 'YOLO/YOLO kode og data eksempel/Image_2.jpg',
                 'YOLO/YOLO kode og data eksempel/Image_3.jpg',
                 'YOLO/YOLO kode og data eksempel/Image_4.jpg',
                 'YOLO/YOLO kode og data eksempel/Image_5.jpg',
                 'YOLO/YOLO kode og data eksempel/Image_6.jpg',
                 'YOLO/YOLO kode og data eksempel/Image_7.jpg',
                 'YOLO/YOLO kode og data eksempel/Image_8.jpg',
                 ], conf=0.70)  # Predict on an example image
print(toc())

# tic()
# print('### Results: ###')
# print(results)
# print(toc())

# 100x100 px: speed: {'preprocess': 1.2660026550292969, 'inference': 49.2711067199707, 'postprocess': 1.659393310546875}]
# original:   speed: {'preprocess': 1.7788410186767578, 'inference': 53.11155319213867, 'postprocess': 1.77764892578125}]


# View results: bounding boxes
# for r in results:
#     print(r.boxes)  # print the Boxes object containing the detection bounding boxes

# View results: masks
# for r in results:
#     print(r.masks)  # print the Masks object containing the detected instance masks

# View results: class probabilities
# Probs object can be used index, get top1 and top5 indices and scores of classification.
# for r in results:
#     print(r.probs)  # print the Probs object containing the detected class probabilities

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    #r.show()

    # Save results to disk
    r.save(filename=f'results{i+1}.jpg')


# Display the results
# results.show()

# If you want to save the results to a file
# results.save(save_dir='./result.jpg')

# Access specific results like bounding boxes, scores, and class names.
# results.xyxy[0] to get predictions for the first (and typically only) image in batch format [x1, y1, x2, y2, conf, cls]
