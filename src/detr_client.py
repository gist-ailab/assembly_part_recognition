#!/usr/bin/python
import socket
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from DETR import build_model
import yaml
import os
import time 
import json
import torch.nn.functional as F
import torch.nn as nn
import time
import pickle
import struct
from pathlib import Path
from socket_utils import *


def initialize_is_model(device):
    model, criterion, postprocessors = build_model()
    checkpoint = torch.load(params["is_weight_path"], map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    return model, postprocessors
    
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


if __name__ == "__main__" :

    yaml_path = os.path.join(Path(__file__).parent.parent, "params", "single_azure_detr_mpaae_GIST.yaml")
    with open(yaml_path) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = params["is_gpu_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==> Loading DETR on", device, params["is_gpu_id"])
    model, postprocessors = initialize_is_model(device)
    rgb_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
                ])

    sock = socket.socket()
    sock.connect((params["tcp_ip"], params["detr_tcp_port"]))
    print("==> Connected to DETR server on {}:{}".format(params["tcp_ip"], params["detr_tcp_port"]))
    
    while True:
        rgb_img = recvall_image(sock)
        rgb = rgb_transform(rgb_img).unsqueeze(0)
        outputs = model(rgb.to(device))

        scores = outputs["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1)[0] > params["is_thresh"]
        pred_boxes = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu().detach(), (params["width"], params["height"]))
        pred_labels = []
        pred_scores = []
        for p in probas[keep]: 
            cl = p.argmax()
            pred_labels.append(cl.cpu().detach().numpy())
            pred_scores.append(p[cl].cpu().detach().numpy())
        pred_masks = outputs["pred_masks"][scores > params["is_thresh"]]
        pred_masks = F.interpolate(pred_masks.unsqueeze(0), size=(params["height"], params["width"]), mode="bilinear", align_corners=False)
        pred_masks = pred_masks.sigmoid()
        pred_masks = pred_masks.squeeze(0).cpu().detach().numpy()
        sendall_pickle(sock, pred_masks)
        sendall_pickle(sock, pred_boxes)
        sendall_pickle(sock, pred_labels)
        sendall_pickle(sock, pred_scores)

    s.close()
