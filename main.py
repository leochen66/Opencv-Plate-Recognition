import cv2
import numpy as np
from predict import SVM
from numpy.linalg import norm

def show_image(imgage):
	cv2.imshow('My Image', imgage)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Find which element is cloest to target element in a list
def most_close(li, target):
	dis = abs(li[0]-target)
	position = 0
	for i in range(1,len(li)):
		if abs(li[i] - target) < dis:
			dis = abs(li[i] - target)
			position = i
	return position

def vertical_projector(image):
	(h,w) = image.shape
	record = [0 for z in range(0, w)] # initialize a list with length w

	for j in range(0,w):
		for i in range(0,h):
			if  image[i,j]==0:
				record[j]+=1
				image[i,j]=255

	# do plot
	for j  in range(0,w):
	    for i in range((h-record[j]),h):
	        image[i,j]=0
	return record

def herizontal_project(image):
	(h,w) = image.shape
	record = [0 for z in range(0, h)]

	for j in range(0,h):
		for i in range(0,w):
			if  image[j,i]==0:
				record[j]+=1
				image[j,i]=255

	# do plot
	for j  in range(0,h):  
	    for i in range(0,record[j]):   
	        image[j,i]=0
	return record

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        
        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        
        samples.append(hist)
    return np.float32(samples)

# Preprocess of prediction
def predict_preprocess(img):
	w = abs(img.shape[1] - 20)
	img = cv2.copyMakeBorder(img, 0, 0, w, w, cv2.BORDER_CONSTANT, value = [0,0,0])
	img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
	img = preprocess_hog([img])
	return img

### Following functions are the main part of plate recognition

# Locate our target plate
# input：original image
# output：plate image in gray scale
def plate_location(gray_img):
	# GaussianBlur: remove noise
	gray_img = cv2.GaussianBlur(gray_img, (9, 9), 0)

	# find ROI(Start point(165,420))
	#origin_ROI = origin_img[165:165+210, 420:420+340]
	gray_ROI = gray_img[165:165+210, 420:420+340]

	# Edge computing
	gray_edge = cv2.Sobel(gray_ROI, cv2.CV_16S, 1, 0)
	gray_edge = cv2.convertScaleAbs(gray_edge)

	# threshold + filter
	ret, gray_threshold = cv2.threshold(gray_edge,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# close computing Swell
	erodeStructx = cv2.getStructuringElement(cv2.MORPH_RECT,(15,5)) #定義矩形
	gray_close = cv2.morphologyEx(gray_threshold, cv2.MORPH_CLOSE, erodeStructx)

	# find counters
	gray_cuonters, hierarchy = cv2.findContours(gray_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(gray_ROI,gray_cuonters,13,(0,0,255),3)

	# recognize which counter belongs to plate
	ratios = []
	for counter in gray_cuonters:
		x, y, w, h = cv2.boundingRect(counter)
		# print(w,h,w/h)
		if w+h>150:
			ratios.append(w/h)
		else:
			ratios.append(0)
	pos = most_close(ratios, 2.5)
	# print('pick:', pos)
	x, y, w, h = cv2.boundingRect(gray_cuonters[pos])
	plate_x = x
	plate_y = y
	plate_height = h
	plate_width = w

	# cut the plate
	plate_view = gray_ROI[plate_y:plate_y+plate_height, plate_x:plate_x+plate_width]
	return plate_view

# Desection String
# input：plate image in gray scale
# output：String image
def extract_string(plate_view):
	# threshold
	ret, plate_thre = cv2.threshold(plate_view,74,255,cv2.THRESH_BINARY)

	# Herizontal Projection
	plate_hpro = plate_thre.copy()
	her_pro = herizontal_project(plate_hpro)
	#print('水平:', her_pro)

	# remove redundant part according to herizontal projection
	for i in range(len(her_pro)):
		if her_pro[i+1] >= her_pro[i] + 15:
			top = i+1
			break
	for i in reversed(range(len(her_pro))):
		if her_pro[i-1] >= her_pro[i] + 15:
			bottom = i-1
			break
	# print('top:', top)
	# print('bottom:', bottom)
	plate_cut = plate_thre[top-3:bottom+3, ]

	# Vertical Projection
	plate_vpro = plate_cut.copy()
	ver_pro = vertical_projector(plate_vpro)
	#print(ver_pro)

	# cut both sides without any characters
	flag = False   # set True when start having 0
	for i in range(len(ver_pro)):
		if flag:
			if ver_pro==0:
				continue
			else:
				left = i
				break
		else:
			if ver_pro[i]==0:
				flag = True
			else:
				continue
	flag = False   # set True when start having 0
	for i in reversed(range(len(ver_pro))):
		if flag:
			if ver_pro==0:
				continue
			else:
				right = i
				break
		else:
			if ver_pro[i]==0:
				flag = True
			else:
				continue
	# print('left:', left)
	# print('right', right)
	plate_cut = plate_cut[0:,left-1:right+2]

	return plate_cut

# Desect Char
# input：String image
# output：list with mutiple char images
def spilt_char(plate_cut):
	# Erosion
	kernel = np.ones((2,2),np.uint8)
	plate_erode = cv2.dilate(plate_cut,kernel)

	# Vertical Projection
	plate_vpro = plate_erode.copy()
	ver_pro = vertical_projector(plate_vpro)
	#print(ver_pro)

	# desect characters by vertical projection
	characters_index = [] # store chars coordinates
	index = 1
	for i in range(7):

		# check the number of characters(6 or 5)
		if i ==6:
			check = 0
			for j in range(index, len(ver_pro)):
				check = check + ver_pro[j]
			if check == 0:
				break

		flag = True
		left = 0
		right = 0
		while left==0 or right==0 :
			if flag:
				if ver_pro[index] ==0:
					index = index+1
					continue
				else:
					flag = False
					left = index
					index = index+1
			else:
				if ver_pro[index] !=0:
					index = index+1
					continue
				else:
					flag = True
					right = index
					index = index+1
		characters_index.append([left,right])
	#print(characters_index)

	# cut image after finding coordinates
	character_images = []
	for pos in characters_index:
		split = plate_erode[0:,pos[0]-2:pos[1]+2]
		kernel = np.ones((3,3),np.uint8)
		split = cv2.erode(split,kernel) # Swell
		thresh, split = cv2.threshold(split,127,255,cv2.THRESH_BINARY_INV) # Reverse
		character_images.append(split)

	return character_images

# Predict Characters
# input：list with mutiple char images
# output：list with mutiple char
def prediction(model, character_images):
	predict_result = []
	for sample in character_images:
		sample = predict_preprocess(sample)
		result = model.predict(sample)
		character = chr(result[0])
		predict_result.append(character)
	#print(predict_result)
	return predict_result


if __name__ == '__main__':

	IMG_PATH = 'test_data/001.jpg'
	image = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
	show_image(image)

	plate = plate_location(image)
	show_image(plate)

	extraction = extract_string(plate)
	show_image(extraction)

	characters = spilt_char(extraction)

	model = SVM(C=1, gamma=0.5)
	model.load("svm.dat")
	result = prediction(model, characters)
	print(result)