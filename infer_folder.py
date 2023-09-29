# -*- coding: utf-8 -*-
import os
import argparse
import logging
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import cv2

from net.pspnet import PSPNet

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
parser.add_argument('--image_folder', type=str, help='Path to the folder containing images')
parser.add_argument('--save_folder', type=str, help='Path to the folder saving images')
parser.add_argument('--models-path', type=str, default='./checkpoints', help='Path for storing model snapshots')
parser.add_argument('--backend', type=str, default='densenet', help='Feature extractor')
parser.add_argument('--num-classes', type=int, default=20, help="Number of classes.")
args = parser.parse_args()


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        if not epoch == 'last':
            epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch


def get_transform():
    transform_image_list = [
        transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(transform_image_list)

def main():
    # --------------- model --------------- #
    snapshot = os.path.join(args.models_path, args.backend, 'PSPNet_last')
    net, starting_epoch = build_network(snapshot, args.backend)
    net.eval()

    # ------------ load images ------------ #
    data_transform = get_transform()
    image_folder = args.image_folder
    save_folder = args.save_folder
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    for image_file in image_files:
        if 'cloth' in image_file:
            continue
        elif 'person' in image_file:
            image_path = os.path.join(image_folder, image_file)
            img = Image.open(image_path)
            # Get the height and width of the PIL image
            image_width, image_height = img.size
            img = data_transform(img)
            img = img.cuda()

            # --------------- inference --------------- #

            with torch.no_grad():
                pred, _ = net(img.unsqueeze(dim=0))
                pred = pred.squeeze(dim=0)
                pred = pred.cpu().numpy().transpose(1, 2, 0)
                pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 256, 1))
                pred = cv2.resize(pred,(image_width,image_height))
                colormap = [(0, 0, 0),
                    (1, 0.25, 0), (0, 0.25, 0), (0.5, 0, 0.25), (1, 1, 1),
                    (1, 0.75, 0), (0, 0, 0.5), (0.5, 0.25, 0), (0.75, 0, 0.25),
                    (1, 0, 0.25), (0, 0.5, 0), (0.5, 0.5, 0), (0.25, 0, 0.5),
                    (1, 0, 0.75), (0, 0.5, 0.5), (0.25, 0.5, 0.5), (1, 0, 0),
                    (1, 0.25, 0), (0, 0.75, 0), (0.5, 0.75, 0), ]
                cmap = matplotlib.colors.ListedColormap(colormap)
                bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                fig, axes = plt.subplots(1, 1)
                ax1 = axes
                ax1.get_xaxis().set_ticks([])
                ax1.get_yaxis().set_ticks([])
                ax1.set_title('pred')
                mappable = ax1.imshow(pred, cmap=cmap, norm=norm)
                # plt.savefig(fname=os.path.join(image_folder, image_file.split('.')[0] + '_mask.png'))
                cv2.imwrite(os.path.join(save_folder, image_file.split('.')[0] + '_mask.png'),pred)

if __name__ == '__main__':
    main()
