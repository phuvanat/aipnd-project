import argparse

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

#python train.py flowers --gpu

def main():
    in_arg = get_input_args()
    #check_command_line_arguments(in_arg)
    
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    save_dir = in_arg.save_dir
    arch = in_arg.arch
    hidden_units = in_arg.hidden_units
    out_features = 102
    epochs = in_arg.epochs
    print_every = 100
    learning_rate = in_arg.learning_rate
    
    data_transforms = {
    'train':transforms.Compose([transforms.Resize(256),
                                       transforms.RandomCrop(224),
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
                                                           [0.229, 0.224, 0.225])]),
}

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train':datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid':datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test':datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test':torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }


    if in_arg.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    if arch == 'vgg11':      
        model = models.vgg11(pretrained=True)
        in_features = 25088 
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        in_features = 25088 
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = 25088 
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        in_features = 25088 
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = 9216
    elif arch == 'densenet121':    
        model = models.densenet121(pretrained=True)
        in_features = 1024
    elif arch == 'densenet169':     
        model = models.densenet169(pretrained=True)
        in_features = 1664
    elif arch == 'densenet161':      
        model = models.densenet161(pretrained=True)
        in_features = 2208
    elif arch == 'densenet201':       
        model = models.densenet201(pretrained=True)
        in_features = 1920
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
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    do_deep_learning(model,dataloaders['train'],dataloaders['valid'], epochs, print_every, criterion, optimizer, device)
    check_accuracy_on_test(model,dataloaders['test'], device)
    
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'in_features': in_features,
              'out_features': out_features,
              'hidden_units': hidden_units,
              'arch': arch,
              'data_dir': data_dir,
              'save_dir': save_dir,
              'epochs': epochs,
              'learning_rate': learning_rate,
              'classifier': model.classifier,
              'optimizer': optimizer,
              'class_to_idx': model.class_to_idx,
              'optimizer_state_dict': optimizer.state_dict(),
              'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    
        
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action="store", default = 'flowers', type = str, help = 'Set directory for dataset')
    parser.add_argument('--save_dir', type = str, default = 'checkpoints', help = 'Set directory to save checkpoints')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'CNN model architecture to use for image classification')
    parser.add_argument('--learning_rate', type = float, default = '0.001', help = 'Number of learning rate')
    parser.add_argument('--hidden_units', type = int, default = '512', help = 'Number of hidden units')
    parser.add_argument('--epochs', type = int, default = '1', help = 'Number of epochs')
    parser.add_argument('--gpu', action="store_true", default=False, help = 'Use GPU for training')
    return parser.parse_args()

def check_command_line_arguments(in_arg):
    print("Command Line Arguments:\n data_dir =", in_arg.data_dir, "\n save_dir =", in_arg.save_dir, "\n arch =", in_arg.arch, "\n learning_rate =", in_arg.learning_rate,"\n hidden_units =", in_arg.hidden_units, "\n epochs =", in_arg.epochs, "\n gpu =", in_arg.gpu)

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
    
def do_deep_learning(model,trainloader,testloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    model.to(device)
    model.train()
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))
                model.eval()
                check_accuracy_on_test(model,testloader, device,mode='Valid')
                model.train()
                running_loss = 0    

def check_accuracy_on_test(model,testloader, device='cpu',mode='Test'):    
    correct = 0
    total = 0
    
    model.to(device)
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('%s Accuracy: %d %%' % (mode, (100 * correct / total)))                
    
def save_checkpoint(filepath):
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'hidden_layers': [4096,1000],
              'arch': 'vgg16',
              'classifier': model.classifier,
              'epochs': epochs,
              'learn_rate': learn_rate,
              'optimizer': optimizer,
              'class_to_idx': model.class_to_idx,
              'optimizer_state_dict': optimizer.state_dict(),
              'state_dict': model.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')
    
if __name__ == "__main__":
    main()