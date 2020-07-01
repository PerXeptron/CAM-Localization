# Class Activation Map Localization

![Generic badge](https://img.shields.io/badge/Python-3.7-green.svg) ![Generic badge](https://img.shields.io/badge/OS-Linux-red.svg) ![Generic badge](https://img.shields.io/badge/PyTorch-1.5.1-<COLOR>.svg) 
 

## Enivironment

* **OS** : Linux
* **Python** : 3.7.6
* **CPU** : i7-6700HQ ~2.60GHz
* **GPU** : GTX 1060 6GB

## Preprocessing

Transform sequence :
```
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(imageSize))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)   
```
where ```imageSize = 224```.

## Model

This **Class Activation Map** is generated using a single DenseNet-121 based model. The model pipeline is as shown below:

![DenseNet-121 Pipeline](https://github.com/PerXeptron/CAM-Localization/blob/master/model/DenseNet-121%20Pipeline.jpg)

## Algorithm and Explanation

![CNN Localization](http://cnnlocalization.csail.mit.edu/framework.jpg)

CAM algorithm used here works on similar lines of what has been shown above.

* We take in a image and pass it through the transformation pipeline and finally flatten it, to feed it to the convolutional ```densenet121 base```.

* ```output = self.model(input.cuda())``` is the output of the final ```relu``` Activation layer of DenseNet.

* ```self.weights = list(self.model.parameters())[-2]``` are the 1024 weights of the second-last layer of the model convolutional base.

* The heatmap generated will be the weighted average of the **2D Sub-Tensors** of the **4D Output Tensor** over the entire ```convolutional base output```. So, the final heatmap is  => ```heatmap += self.weights[i] * output[0,i,:,:]``` with *i* ranging from ```0 to len(self.weights)``` which is *1024*

* Finally we complete the average and make sure that its a ```float32``` image.
```
  npHeatmap = heatmap.cpu().data.numpy()
  cam = npHeatmap / np.max(npHeatmap)
```
* We have the heatmap now :smile:

## Finally Some Example Images :nerd_face:

<p float="left">
  <img src="/xrays/xray3.jpg" width="200" height="200" />
  <img src="/heatmaps/heatmap3.jpg" width="200" /> 
</p>

<p float="left">
  <img src="/xrays/xray2.jpg" width="200" height="200" />
  <img src="/heatmaps/heatmap2.jpg" width="200" /> 
</p>

<p float="left">
  <img src="/xrays/xray4.jpg" width="200" height="200" />
  <img src="/heatmaps/heatmap4.jpg" width="200" /> 
</p>


## References

* This repo [here](https://github.com/zoogzog/chexnet) helped me a lot. 

* This paper about ```Learning Deep Features for Discriminative Localization``` [here](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf) explains the algorithm.
