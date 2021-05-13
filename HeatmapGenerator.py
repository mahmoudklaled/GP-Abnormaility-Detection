import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from densenet import densenet121
from densenet import densenet169
from densenet import densenet201
from densenet import DenseNet

#-------------------------------------------------------------------------------- 
#---- Class to generate heatmaps (CAM)

class HeatmapGenerator ():
    
    #---- Initialize heatmap generator
    #---- pathModel - path to the trained densenet model
    #---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    #---- nnClassCount - class count, 14 for chxray-14

    model = DenseNet()
    def __init__ (self, pathModel, nnArchitecture, nnClassCount, transCrop):
       
        #---- Initialize the network
        if nnArchitecture == 'DENSE-NET-121': model = densenet121(False).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = densenet169(False).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = densenet201(False).cuda()
          
        model = torch.nn.DataParallel(model).cuda()

        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['best_model_wts'], strict=False)

        self.model = model.module.features
        self.model.eval()
        
        #---- Initialize the weights
        self.weights = list(self.model.parameters())[-2]

        #---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)
    
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, transCrop):
        
        #---- Load image, transform, convert 
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)
        
        input = torch.autograd.Variable(imageData)
        
        self.model.cuda()
        output = self.model(input.cuda())
        
        #---- Generate heatmap
        heatmap = None
        for i in range (0, len(self.weights)):
            map = output[0,i,:,:]
            if i == 0: heatmap = self.weights[i] * map
            else: heatmap += self.weights[i] * map
        
        #---- Blend original and heatmap 
        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
        img = heatmap * 0.5 + imgOriginal
            
        cv2.imwrite(pathOutputFile, img)
        
#-------------------------------------------------------------------------------- 

pathInputImage = '/content/GP/test/00011997_003.png'
pathOutputImage = '/content/GP/test/heatmap.png'
pathModel = '/content/GP/checkpoint_val0.085970_train0.122693_epoch10'

nnArchitecture = 'DENSE-NET-121'
nnClassCount = 6

transCrop = 224

h = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)
h.generate(pathInputImage, pathOutputImage, transCrop)