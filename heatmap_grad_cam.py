from grad_cam import GradCAM

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np
import argparse
import os

import scipy
import imageio
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation

import skimage
from skimage.io import *
from skimage.transform import *

os.environ["CUDA_VISIBLE_DEVICES"]=""

class DenseNet121(nn.Module):
	"""Model modified.
	The architecture of our model is the same as standard DenseNet121
	except the classifier layer which has an additional sigmoid function.
	"""
	def __init__(self, out_size):
		super(DenseNet121, self).__init__()
		self.densenet121 = torchvision.models.densenet121(pretrained=True)
		num_ftrs = self.densenet121.classifier.in_features
		self.densenet121.classifier = nn.Sequential(
		    nn.Linear(num_ftrs, out_size),
		    nn.Sigmoid()
		)

	def forward(self, x):
		x = self.densenet121(x)
		return x





ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
args = vars(ap.parse_args())

test_X = []
img = imageio.imread(args["image"])

#if img.shape != (1024,1024):
#    print(img.shape)
#    img = img[:,:,0]

img_resized = skimage.transform.resize(img,(256,256))

test_X.append((np.array(img_resized)).reshape(256,256,1))

test_X = np.array(test_X)

class ChestXrayDataSet_plot(Dataset):
	def __init__(self, input_X = test_X, transform=None):
		self.X = np.uint8(test_X*255)
		self.transform = transform

	def __getitem__(self, index):
		"""
		Args:
		    index: the index of item 
		Returns:
		    image 
		"""
		current_X = np.tile(self.X[index],3)
		image = self.transform(current_X)
		return image
	def __len__(self):
		return len(self.X)


test_dataset = ChestXrayDataSet_plot(input_X = test_X,transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ]))


model = DenseNet121(8)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("model/DenseNet121_aug4_pretrain_WeightBelow1_1_0.829766922537.pkl", map_location=torch.device('cpu')))
print("model loaded")


gcam = GradCAM(model=model, cuda=True)
input_img = Variable((test_dataset[0]).unsqueeze(0).cuda(), requires_grad=True)
probs = gcam.forward(input_img)
print(probs)


"""
    activate_classes = np.where((probs > thresholds)[0]==True)[0] # get the activated class
    for activate_class in activate_classes:
        gcam.backward(idx=activate_class)
        output = gcam.generate(target_layer="module.densenet121.features.denseblock4.denselayer16.conv.2")
        #### this output is heatmap ####
        if np.sum(np.isnan(output)) > 0:
            print("fxxx nan")
        heatmap_output.append(output)
        image_id.append(index)
        output_class.append(activate_class)
    print("test ",str(index)," finished")

print("heatmap output done")
print("total number of heatmap: ",len(heatmap_output))
"""