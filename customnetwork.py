from torch import nn
from torchvision import models
from collections import OrderedDict
#
def model_initializer(architecture = 'vgg16'):
    """ Initializes Pre Trained Model
     
        Parameters:
        architecture -- Supported pre trained architecture, vgg16, densenet161, or restnet18
        
        Returns: 
        model -- Pretrained model
    """
    model=nn.Module()
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif  architecture == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif  architecture == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        print('Unsupported model architecture: ', architecture)
        print('Please use only vgg16, densenet161, or resnet18')
        raise NameError('Unsupported model architecture')
    return model
#
def classifier_factory(input_size, hidden_size, output_size):
    """ Creates a custom one layer fc network with logsoftmax output
    
        Parameters:
        input_size -- Input nodes
        hidden_size -- Hidden nodes
        output_size -- Output nodes
        
        Returns:
        classifier -- Sequential fc model with one hidden layer and logsoftmax output
    """
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_size)),
                          ('relu1', nn.ReLU()),
                          ('dropout1',nn.Dropout(p=0.5)),       
                          ('fc2', nn.Linear(hidden_size, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                    ]))
    return classifier
#
def model_customizer(checkpoint, model):
    """ Rehydrates a model from a checkpoint
    
        Parameters: 
        checkpoint -- previously saved checkpoint
        model -- Initialized model
        
        Returns:
        model -- model with custom pre trained classifier
        model.classifier[0].in_features
    """
    classifier = classifier_factory(checkpoint['input_size'],
                                    checkpoint['hidden_size'], 
                                    checkpoint['output_size'])
    # Lock feature parameters
    for param in model.parameters():
        param.requires_grad = False
    # Replace Classifier
    model.classifier=classifier
    # Restore Model
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_index']
    return model
#