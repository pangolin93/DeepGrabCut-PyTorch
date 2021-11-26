import cv2
import numpy as np
import os
import torch


from torch.nn.functional import upsample
from dataloaders import utils
import networks.deeplab_resnet as resnet

from glob import glob
from copy import deepcopy
from bwsss import DATA_DIR, DEVICE, OUTPUT_DIR

from matplotlib import pyplot as plt

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

def deep_mask(path_image):
    image = cv2.imread(path_image)
    img_shape = (450, 450)
    image = utils.fixed_resize(image, img_shape).astype(np.uint8)

    w = img_shape[0]
    h = img_shape[1]
    output = np.zeros((w, h, 3), np.uint8)
    output[ 100:150, 400:449,:] = 1
    output[ 100+5:150-5, 400+5:449-5,:] = 0

    # output1 = output.copy()
    # output1[75:300, 0:180, :] = 1
    # output1[75+5:300-5, 0+5:180-5, :] = 0

    # output2 = output.copy()
    # output2[75:300, 250:400, :] = 1
    # output2[75+5:300-5, 0+5:250-5, :] = 0

    #output = output1 + output2 

    thres = 0.8
 
    left = 300 # 0xFFF
    right = 0
    up = 300 # 0xFFF
    down = 0

    gpu_id = 0
    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

    #  Create the network and load the weights
    net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
    print("Initializing weights from: {}".format(os.path.join(THIS_DIR, 'models', 'deepgc_pascal_epoch-99.pth')))
    state_dict_checkpoint = torch.load(os.path.join(THIS_DIR, 'models', 'deepgc_pascal_epoch-99.pth'),
                                    map_location=lambda storage, loc: storage)

    net.load_state_dict(state_dict_checkpoint)
    net.eval()
    net.to(device)

    ######################
    tmp = (output[:, :, 0] > 0).astype(np.uint8)
    tmp_ = deepcopy(tmp)
    fill_mask = np.ones((tmp.shape[0] + 2, tmp.shape[1] + 2))
    fill_mask[1:-1, 1:-1] = tmp_
    fill_mask = fill_mask.astype(np.uint8)
    cv2.floodFill(tmp_, fill_mask, (int((left + right) / 2), int((up + down) / 2)), 5)
    tmp_ = tmp_.astype(np.int8)

    output = cv2.resize(output, img_shape)

    tmp_ = tmp_.astype(np.int8)
    tmp_[tmp_ == 5] = -1  # pixel inside bounding box
    tmp_[tmp_ == 0] = 1  # pixel on and outside bounding box

    tmp = (tmp == 0).astype(np.uint8)

    dismap = cv2.distanceTransform(tmp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)  # compute distance inside and outside bounding box
    dismap = tmp_ * dismap + 128

    dismap[dismap > 255] = 255
    dismap[dismap < 0] = 0
    dismap = dismap

    dismap = utils.fixed_resize(dismap, (450, 450)).astype(np.uint8)

    dismap = np.expand_dims(dismap, axis=-1)

    image = image[:, :, ::-1] # change to rgb
    merge_input = np.concatenate((image, dismap), axis=2).astype(np.float32)
    inputs = torch.from_numpy(merge_input.transpose((2, 0, 1))[np.newaxis, ...])

    # Run a forward pass
    inputs = inputs.to(device)
    outputs = net.forward(inputs)
    outputs = upsample(outputs, size=(450, 450), mode='bilinear', align_corners=True)
    outputs = outputs.to(torch.device('cpu'))

    prediction = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
    prediction = 1 / (1 + np.exp(-prediction))
    prediction = np.squeeze(prediction)
    prediction[prediction>thres] = 255
    prediction[prediction<=thres] = 0

    prediction = np.expand_dims(prediction, axis=-1).astype(np.uint8)
    image = image[:, :, ::-1] # change to bgr
    image_original = image.copy()
    display_mask = np.concatenate([np.zeros_like(prediction), np.zeros_like(prediction), prediction], axis=-1)
    image = cv2.addWeighted(image, 0.9, display_mask, 0.5, 0.1)

    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.savefig('image_deepgc.png')

    plt.imshow(cv2.cvtColor(display_mask,cv2.COLOR_BGR2RGB))
    plt.savefig('image_deepgc_mask.png')


    # plt.imshow(image_original,cv2.COLOR_BGR2RGB)
    # plt.savefig('image_original.png')

if __name__ == "__main__":

    idx = 149
    root_dir = os.path.join(DATA_DIR, 'weak_images')
    img_name = os.path.join(root_dir, f'{idx}.tif') #'/home/valeria/projects/bayesian-fans/bwsss/DeepGrabCut-PyTorch/ims/2007_000042.jpg'#

    deep_mask(img_name)
