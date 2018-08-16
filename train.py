""" Trains a flower prediction model on a database of flower images
"""
# Imports here
import sys
import time
import argparse
#
import numpy as np
#
import matplotlib.pyplot as plt
#
import torch
from torch import nn
from torch import optim
from torchvision import  datasets, transforms, models
#
from customnetwork import  classifier_factory,  model_initializer, model_customizer
#
def validate_classifier(model, criterion, testloader, device='cpu'):
    """ Validates a model with trained classifier
        Returns Nothing.  Prints results
        
        Modified From Udacity AI Nanodegree Lesson 4, Section 7
        Parameters:
        model -- a torch model
        testloader -- torch dataloader with test images
        device -- Device used in training     
        Returns:
        tets_loss, accuracy
    """
    # Initialize control variables
    test_loss = 0
    accuracy = 0
    # Initialize Model
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
           
            outputs = model.forward(images)           
            test_loss += criterion(outputs, labels).item()    
            # Get probabilities from LogSoftMax
            ps = torch.exp(outputs)
            
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()          
    return test_loss, accuracy
#
#
def train_classifier(model, trainloader, testloader, epochs, device = 'cpu', learning_rate = 0.001):
    """ Trains the classifier of the model.
        Taken From Udacity AI Nanodegree Lesson 4, Section 5

        Parameters:
        model -- torch Model
        trainloader -- torch dataloader for training
        testloader --  torch dataloader for validation
        epochs -- integer, number of epochs to train
        device -- torch device to train on
        learning_rate -- Model learning rate for optimizer
        
        Returns:  model, optimizer
    """
    # Model training
    # Normalize the initial weights
    model.classifier.fc1.weight.data.normal_(std=0.01)
    model.classifier.fc2.weight.data.normal_(std=0.01)
    # Loss function for LogSoftMax
    criterion = nn.NLLLoss()
    # Initialize control variables
    steps = 0
    running_loss = 0
    print_every = 40
    #
    model.to(device)
    model.train()
    # Set up optimizer after moving model to device
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    for e in range(epochs):  
        for ii,(images, labels) in enumerate(trainloader):
            start = time.time()
            images, labels = images.to(device), labels.to(device)  
            # Train
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            #
            steps += 1
            running_loss += loss.item()
            # Output loss
            if steps % print_every == 0:
                model.eval()      
                print(f"Time per batch: {(time.time() - start)/print_every:.3f} seconds")
                test_loss, accuracy = validate_classifier(model, criterion, testloader, device)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(testloader)))
                # Reset
                model.train()
                start = time.time()
                running_loss = 0
    return model, optimizer, criterion
#
def initialize_dataloaders(data_dir):
    """ Initializes the dataloaders for train, valid, and test image sets
    
        Parameters:
        data_dir -- root directory with train, valid, and test subdirectories
                    
        Returns: -- data_loaders, image_datasets
    """
    data_dirs = {
       'train': data_dir + '/train', 
       'valid': data_dir + '/valid', 
       'test': data_dir + '/test'
    }
    # Special transforms for each set
    data_transforms = {
       'train': transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]),
       'valid':transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])]),
       'test':transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])])
    }
    # Load the datasets
    image_datasets = {
       'train': datasets.ImageFolder(data_dirs['train'],transform=data_transforms['train']), 
       'valid':  datasets.ImageFolder(data_dirs['valid'],transform=data_transforms['valid']), 
       'test': datasets.ImageFolder(data_dirs['test'],transform=data_transforms['test'])
    }
    # Initialize the dataloaders
    data_loaders = {
       'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True), 
       'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32), 
       'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }                 
    return data_loaders, image_datasets
#  
def init_args():
    """Parses and Initializes command line arguments
    
        Required: data_directory -- Root path to images to train and test on.
        Optional: --save_dir -- Path to model checkpoint folder.
                  --arch -- A supported pre-trained model architecture, 
                            eg: vgg16, densenet161, resnet18.
                            classifier input_sizes
                            vgg16: 25088
                            densenet161: 1024
                            resnet18: 512
                  --learning_rate -- A fractional rate eg 0.001.
                  --hidden_units --  The number of nodes in the classifiers hidden layer, 
                                     between 25088 and 102, eg: 5000
                  --epochs -- Number of epochs to train for. eg: 3.
                  --gpu -- Whether to use gpu for processing.
        Returns:  parser.parse_args()
    """
    
    parser = argparse.ArgumentParser(
       description = 'Flower Image Training Program',
    )
    parser.add_argument('data_directory', action = 'store', type = str, help = 'Path to images to train and test on.', default='flowers')
    parser.add_argument('--save_dir', action = 'store', type = str, help = 'Path to model checkpoint folder.', default='')
    parser.add_argument('--arch', action = 'store', type = str, help = 'A supported pre-trained model architecture, eg: vgg16, densenet161, resnet18.', default = 'vgg16')   
    parser.add_argument('--learning_rate', action = 'store', type = float, help = 'A fractional rate eg 0.001', default = 0.001)
    parser.add_argument('--hidden_units', action = 'store', type = int, help = 'The number of nodes in the classifiers hidden layer.', default = 5000)
    parser.add_argument('--epochs', action = 'store', type = int, help = 'Number of epochs to train for. eg: 3', default = 3)
    parser.add_argument('--gpu', action = 'store_true', default = False, help = 'Use gpu if available')
    return parser.parse_args()
#
def main():
    """ Main training program """
    input_sizes={
        'vgg16': 25088,
        'densenet161': 1024,
        'resnet18': 512
    }
    # input sizes 
    #model.classifier[0].in_features
    args=init_args()
    print(args)
    # Load Data
    data_loaders, image_datasets= initialize_dataloaders(args.data_directory)
    # Initialize model
    model = model_initializer(args.arch)
    # Customize the classifier
    input_size = input_sizes[args.arch]
    output_size = 102
    model.classifier=classifier_factory(input_size, args.hidden_units, output_size) 
    # Train Classifier
    # Set device
    device = torch.device('cpu')
    if(args.gpu):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Training on ', device)
    #
    model, optimizer, criterion = train_classifier(model, data_loaders['train'], data_loaders['test'], args.epochs, device, args.learning_rate)
     
    print('Saving Checkpoint')
    checkpoint={
    'epochs': args.epochs,
    'optimizer_state': optimizer.state_dict(),
    'class_to_index': image_datasets['train'].class_to_idx,
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': args.hidden_units,
    'state_dict': model.state_dict(),
    'architecture': args.arch,
    'gpu': args.gpu
    }   
    #
    checkpoint_path = args.save_dir+'/checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
# Call Main
if __name__ == '__main__':
   main()                                                                           