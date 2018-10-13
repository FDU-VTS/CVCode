# written by liwenxi
import skimage.io as io
import selectivesearch
import skimage.transform as transform
import numpy as np
import sys
import os
import torch.nn as nn
import torch
from torchvision import datasets, models, transforms


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def get_regions(img_path):
    # loading  image
    img = io.imread(img_path)

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    print(len(img_lbl))
    print(np.shape(img))
    candidates = set()

    for r in regions:
        # excluding same rectangle (withsl different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 100:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image

    for i, R  in enumerate(candidates):
        x, y ,w, h = R
        nor_img = transform.resize(img[y:y + h, x:x + w, :], (224, 224))
        io.imsave("./region/"+str(i)+'.jpg', nor_img)

def get_prediction_lists():
    pre_file = os.path.join('region')
    # Set the groups in a dictionary.
    file_groups = []
    for root,dirs,files in os.walk(pre_file):
        for i in files:
            file_groups.append(i)
    return  file_groups

def prediction(model, img):
    outputs = model(img)
    print(outputs)
    return torch.max(outputs, 1)

def main():
    if len(sys.argv) != 2:
        print("Wrong Enter!")
        return 1
    img_path = sys.argv[1]
    get_regions(img_path)
    file_groups = get_prediction_lists()
    device = torch.device("cuda")
    
    model_name = "vgg"
    num_classes = 20
    batch_size = 8
    num_epochs = 15
    input_size = 224
    feature_extract = False

    # Initialize the model for this run
    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=(0, 1, 2, 3))
    model.load_state_dict(torch.load('net_pretrained_params.pkl'))
    model.eval()
    # model = model.to(device)
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # bring images to (-1,1)
    ])
    for i, pre_image in enumerate(file_groups):
        if pre_image == ".DS_Store":
            continue
        print("name:", pre_image)
        img = io.imread("./region/"+pre_image)
        test = torch.zeros(8, 3, 224, 224)
        img = torch.from_numpy(img)
        test[0, 0, :, :] = img[:, :, 0]
        test[0, 1, :, :] = img[:, :, 1]
        test[0, 2, :, :] = img[:, :, 2]
        test = test.to(device)
        result = prediction(model, test)
        print(result[1][0], result[0][0])
    return 0


if __name__ == '__main__':
    main()



