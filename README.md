# Opencv-Plate-Recognition

## Introduction
This is a car plate recognition project implemented by Python Opencv.

## Requirement
```
pip install opencv-python
```

## Usage
```
python main.py
```

## Explanation
There are a few main functions in this progrm, representing critical steps in this recognition algorithm, as following =><br/>
* plate_location(): Locate the exact coordinate of car plate from original image.
* extract_string(): Remove redundant portion from the plate image, left only string in the result diagram.
* spilt_char(): Cut the string diagram into single characters.
* prediction(): Using SVM model(provide by opencv)