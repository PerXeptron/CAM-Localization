import os
import re
import time
import sys
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from DenseNet import DenseNet121

# Class Activation Map code for plotting activation Heatmaps of different 
# anomaly regions in the supplied X-Ray

class HeatmapGenerator ():
    
    def __init__ (self, pathModel, nnClassCount, imageSize):
       
        #---- Initialize the network
        model = DenseNet121(nnClassCount, True).cuda()
          
        model = torch.nn.DataParallel(model).cuda()

        modelCheckpoint = torch.load(pathModel)
        state_dict =modelCheckpoint['state_dict']
		
        #modify:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.

        #So, let's write some Regex code to fix this issue

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)

        #For visualising the final convolutional layer we will define the 
        # model to be the convolutional base(densenet121)

        self.model = model.module.densenet121.features
        self.model.eval()
        
        #Initialize the weights
        self.weights = list(self.model.parameters())[-2]

        #Initialize the image transform - resize and normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(imageSize))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)
   

    def generate (self, pathImageFile, pathOutputFile, imageSize):
        
        #Load image, transform, convert 
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)
        input = torch.autograd.Variable(imageData)
        
        self.model.cuda()
        output = self.model(input.cuda())
        
        #Generate heatmap, on class based Activation
        heatmap = None
        for i in range (0, len(self.weights)):
          map = output[0,i,:,:]
          if i == 0: heatmap = self.weights[i] * map
          else: heatmap += self.weights[i] * map
        
        #Blend the images
        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (imageSize, imageSize))
        
        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (imageSize, imageSize))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_TURBO)
              
        img = heatmap * 0.5 + imgOriginal
            
        cv2.imwrite(pathOutputFile, img)


pathInputImage = 'xray3.jpg'
pathOutputImage = 'heatmap3.jpg'
pathModel = 'C:/Users/Richeek Das/Documents/GitHub/Grad-CAM-Localization/model/densenet.pth.tar'

nnClassCount = 14

imageSize = 224

h = HeatmapGenerator(pathModel, nnClassCount, imageSize)
h.generate(pathInputImage, pathOutputImage, imageSize)