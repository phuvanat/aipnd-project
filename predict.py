import argparse

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json

#python predict.py flowers/test/28/image_05253.jpg checkpoints/checkpoint.pth --gpu

def main():
    in_arg = get_input_args()
    #check_command_line_arguments(in_arg)
    
    test_image = in_arg.input
    checkpoint_path = in_arg.checkpoint
    top_k = in_arg.top_k
    category_names = in_arg.category_names
    gpu = in_arg.gpu
        
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    test_image_path = test_image
    saved_model = load_checkpoint(checkpoint_path)
    saved_model = saved_model.to(device)
    probs, classes = predict(test_image_path, saved_model, top_k, gpu)
    class_names = [cat_to_name[name] for name in classes]

    print(list(zip(class_names, probs)))   
        
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', action="store", type = str, 
                        default = 'flowers/test/28/image_05253.jpg', help = 'Set directory path for image')
    parser.add_argument('checkpoint', action="store", type = str, 
                        default = 'checkpoints/checkpoint.pth', help = 'Set directory for checkpoint')
    parser.add_argument('--top_k', type = int, 
                        default = '3', help = 'Return top K most likely classes')
    parser.add_argument('--category_names', type = str, 
                        default = 'cat_to_name.json', help = 'Use a mapping of categories to real names')
    parser.add_argument('--gpu', action="store_true", 
                        default=False, help = 'Use GPU for training')
    return parser.parse_args()

def check_command_line_arguments(in_arg):
    print("Command Line Arguments:\n /path/to/image =", in_arg.path_to_image, "\n checkpoint =", in_arg.checkpoint, "\n top_k =", in_arg.top_k, "\n category_names =", in_arg.category_names, "\n gpu =", in_arg.gpu)
    
def load_model(arch = 'vgg19', in_features = 25088, hidden_units = 512, out_features = 102):

    if arch == 'vgg11':      
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'densenet121':    
        model = models.densenet121(pretrained=True)
    elif arch == 'densenet169':     
        model = models.densenet169(pretrained=True)
    elif arch == 'densenet161':      
        model = models.densenet161(pretrained=True)
    elif arch == 'densenet201':       
        model = models.densenet201(pretrained=True)
    else:
        raise ValueError('Can not specific arch model.')
        
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(in_features, 1000)),
                            ('relu1', nn.ReLU(True)),
                            ('dropout1', nn.Dropout(p=0.5)),
                            ('fc2', nn.Linear(1000, hidden_units)),
                            ('relu2', nn.ReLU(True)),
                            ('dropout2', nn.Dropout(p=0.5)),
                            ('fc3', nn.Linear(hidden_units, out_features)),
                            ('output', nn.LogSoftmax(dim=1))
            ]))
    
    model.classifier = classifier

    return model
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    in_features = checkpoint['in_features']
    hidden_units = checkpoint['hidden_units']
    out_features = checkpoint['out_features']
    model = load_model(arch, in_features, hidden_units, out_features)
    model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']   
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.transpose(image, (1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    with torch.no_grad():
        if gpu:
            image_input = torch.from_numpy(process_image(image_path)).unsqueeze(0).float().cuda()
        else:
            image_input = torch.from_numpy(process_image(image_path)).unsqueeze(0).float()
    output = model(image_input)
    probs, indices = torch.topk(F.softmax(output, dim=1), topk, sorted=True)
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    return [prob.item() for prob in probs[0].data], [idx_to_class[ix.item()] for ix in indices[0].data]
    
if __name__ == "__main__":
    main()