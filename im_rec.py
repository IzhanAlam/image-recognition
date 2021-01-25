import jetson.inference
import jetson.utils

import argparse

#Command Line
parser = argparse.ArgumentParser()
parser.add_argument("filename",type=str, help="filename of image to process")
parser.add_argument("--network", type=str, default="googlenet", help="Defualt architecture: GoogLeNet")
args = parser.parse_args()

#Load image into shared CPU/GPU memory
img = jetson.utils.loadImage(args.filename)

#Load recognition network
net = jetson.inference.imageNet(args.network)

#Classify image
class_idx, confidence = net.Classify(img)

#Object Description
class_desc = net.GetClassDesc(class_idx)

#Print result
print("Image is described as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))

