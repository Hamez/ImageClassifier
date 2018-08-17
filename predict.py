""" Predicts which flower an image is using trained network"""
# Imports here
import matplotlib.pyplot as plt
#
import sys
import numpy as np
import argparse
#
import torch
#
from torch import nn
from torchvision import models
#
from PIL import Image
import json
from customnetwork import  classifier_factory,  model_initializer, model_customizer
#
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    return checkpoint
#
def initialize_cat_to_name(cat_names_path):
    cat_to_name = {} # new dictionary
    with open(cat_names_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
#
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        
        Parameters:
        image_path -- path to image to test
        
        Returns:
        image -- PIL image normalized for PyTorch
    '''  
    # Process a PIL image for use in a PyTorch model
    new_size=256
    crop=224
    #
    image=Image.open(image_path)
    # scale
    image = scale_image(image, new_size)
    # crop
    image = crop_image(image, crop)
    # normalize 
    np_image = normalize_image(image)
    #
    return np_image
#
def normalize_image(image):
    """ Normalizes images rgb channels of PIL image for PyTorch
    
        Parameters: 
        image -- PIL Image
        
        Returns:
        np_image -- Normalized image as numpy array for PyTorch model consumption
    """
    means=np.array([0.485, 0.456, 0.406])
    stds=np.array([0.229, 0.224, 0.225])
    # 
    np_image=np.array(image)
    np_image=np_image/255
    np_image=(np_image - means)/stds
    # Transpose
    np_image=np_image.transpose((2,0,1))
    return np_image
    
def crop_image(image, crop):
    """ Crop's image to square of crop size
    
        Parameters:
        image -- PIL image
        crop  -- Integer pixels to crop to 
        
        Returns:
        image -- cropped PIL image       
    """
    w=image.size[0]
    midw=w/2
    h=image.size[1]
    midh=h/2
    box=(midw-(crop/2),midh-(crop/2),midw+(crop/2),midh+(crop/2))
    #
    return image.crop(box)
#
def scale_image(image, new_size):
    """ Scales image to new size retaining aspect ratio
        
        Parameters:
        image -- PIL Image from file
        new_size -- integer pixels 
        
        Returns:
        image -- Scaled image
    """
    w=image.size[0]
    h=image.size[1]
    #
    if h > w:      
        r=h/w
        image=image.resize((new_size,(int(new_size*r))))
    else:   
        r=w/h
        image=image.resize(((int(new_size*r)),new_size))
    return image
#
def predict(image_path, model, gpu = False, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
        Parameters:
        image_path -- Path to image file
        model      -- Pretrained Torch model
        gpu        -- Process on GPU 
        topk       -- Integer of number of top probabilities to display
        
        Returns:
        probs, labels -- arrays of the probabilities and numeric class labels for flower type
    '''
    device=torch.device('cpu')
    if(gpu):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #
    print('Device: %s' % device)
    np_image=process_image(image_path)
   
    tn_image=torch.from_numpy(np_image)
 
    tn_image.resize_(1,3,224,224)
  
    model.to(device)
    tn_image = tn_image.float().to(device)
  
    model.eval()
    output=model(tn_image)
    output=torch.exp(output)
    output=output.topk(topk)
    #print("Raw Output: ", output)    #
    probs=output[0][0]
    label_index=output[1][0]
    #
    probs=probs.cpu().detach().numpy()
    label_index=label_index.cpu().detach().numpy()
    #  
    # Reverse class to index dictionary to index to class
    index_to_class = {v: k for k, v in model.cp_class_to_idx.items()}
    # 
    i=0
    label_list=[]
    for lbl_idx in label_index:
        str_cat_index=str(index_to_class[label_index[i]])
        label_list.append(str_cat_index)
        # print(probs[i], label_index[i], str_cat_index)
        i=i+1
    labels=np.array(label_list)
    return probs, labels
#
def init_args():
   """ Initializes command line arguments
    
        Main
        usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                        [--gpu]
                        image_path checkpoint_path

        Flower Prediction Program

        positional arguments:
            image_path            Path to image to predict.
            checkpoint_path       Path to model checkpoint file.

        optional arguments:
            -h, --help            show this help message and exit
            --top_k TOP_K         Number of top categories to display.
            --category_names CATEGORY_NAMES
                                  Path to category name list if available.
            --gpu                 Use gpu if available.
               
   """
   parser = argparse.ArgumentParser(
       description = 'Flower Prediction Program',
   )
   parser.add_argument('image_path', action = 'store', type = str, help = 'Path to image to predict.', default='flowers/train/100/image_07893.jpg')
   parser.add_argument('checkpoint_path', action = 'store', type = str, help = 'Path to model checkpoint file.', default='checkpoint.pth')
   parser.add_argument('--top_k', action = 'store', default=5, type = int,help = 'Number of top categories to display.')
   parser.add_argument('--category_names', action = 'store', type = str, help = 'Path to custom category name list if available.', default='cat_to_name.json')
   parser.add_argument('--gpu', action = 'store_true', default = False, help = 'Use gpu if available.')
   return parser.parse_args()
#
##############MAIN########################
def main():
    model = None
    print('Predicting...') 
    args=init_args()
    # Load Checkpoint 
    # Checkpoint is model
    model = load_checkpoint(args.checkpoint_path)
    # Run prediction
    probs, labels = predict(args.image_path, model, args.gpu, topk=args.top_k)
    cat_to_name = initialize_cat_to_name(args.category_names)
    #
    print('Prediction Result: ', args.image_path)
    print('')
    print('Probability | Flower Name(class)')
    print('--------------------------------')
    for i in range(probs.size):
        print('%d%%' % ( probs[i]*100), '\t  | %s(%s)' % (cat_to_name[labels[i]], labels[i]))
   
# Call it
if __name__ == '__main__':
   main()
#