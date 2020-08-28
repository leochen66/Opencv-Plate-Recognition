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
There are a few main functions in this progrm, representing critical steps in this recognition algorithm => <br/>
![image](https://github.com/leochen66/Opencv-Plate-Recognition/blob/master/display/origin.jpeg)
* plate_location(): Locate the exact coordinate of car plate from original image. <br/>
![image](https://github.com/leochen66/Opencv-Plate-Recognition/blob/master/display/locate.jpeg)
* extract_string(): Remove redundant portion from the plate image, left only string in the result diagram. <br/>
![image](https://github.com/leochen66/Opencv-Plate-Recognition/blob/master/display/string.png)
* spilt_char(): Cut the string diagram into single characters.
* prediction(): Using SVM model (provided by opencv) to recognize each single character image.