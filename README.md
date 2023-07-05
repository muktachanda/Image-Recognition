# Image-Recognition

>#### Run on Ubuntu 22.04.2 LTS

### __Libraries Needed to Install__:
```
pip install numpy
pip install matplotlib
pip install tensorflow
pip install opencv-python
```

### Training the Model:
```
python3 imageModel.py
```

### Testing the Model:
- Add any image in jpg format into the images folder.
- Resize it to 32 x 32 pixels.
- In `imageClassify.py`, change the image path to the desired image.
```
python3 imageClassify.py
```

### About the Project:
This ML model can classify an image into the following categories:

Plane, car, bird, cat, deer, dog, frog, horse, ship, truck.

This project was my attempt to understand how convolutional neural networks are used to classify images.

In `imageModel.py`, the topic is explained to the best of my abilities alongside the code so that the concept is understandable.

>## Statistics:
>### Accuracy: 71.75%
