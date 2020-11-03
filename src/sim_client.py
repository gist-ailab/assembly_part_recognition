#!/usr/bin/env python

from pyrep import PyRep
from os import path
from pyrep.objects import Shape
from pyrep.objects import Dummy
from assembly_simulator.simulator import Camera, Object_

from pathlib import Path
import os
import sys
import yaml
import numpy as np
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from sensor_msgs.msg import CameraInfo
from rospy_message_converter import json_message_converter
import json
import cv2
import socket
from socket_utils import *
from skimage.measure import *

def pose_to_pq(msg):
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array([msg.orientation.x, msg.orientation.y,
                  msg.orientation.z, msg.orientation.w])
    return p, q

def pose_stamped_to_pq(msg):
    return pose_to_pq(msg.pose)

def transform_to_pq(msg):
    p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = np.array([msg.rotation.x, msg.rotation.y,
                  msg.rotation.z, msg.rotation.w])
    return p, q

def transform_stamped_to_pq(msg):
    return transform_to_pq(msg.transform)

        
# 0, 38, 76, 115, 127, 

def get_ssim_with_mask(imageA, imageB, mask):

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, -1)
    imageA = imageA * mask
    imageB = imageB * mask
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, diff = compare_ssim(grayA, grayB, full=True)
    return score

def get_mse_error_with_mask(imageA, imageB, mask):
    if len(imageA.shape) == 3:
        imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    imageA = imageA * mask
    imageB = imageB * mask
    return compare_mse(imageA, imageB)
    
def get_rmse_with_mask(imageA, imageB, mask):
    if len(mask.shape) == 3:
        mask = np.squeeze(mask, -1)
    imageA = imageA * mask
    imageB = imageB * mask
    # grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    # grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    return compare_nrmse(imageA, imageB)



if __name__ == '__main__':

    yaml_path = os.path.join(Path(__file__).parent.parent, "params", "single_azure_detr_mpaae_GIST.yaml")
    with open(yaml_path) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        pe_obj_names = params["pe_class_names"]
        is_obj_names = params["is_class_names"]
    # tcp connect
    sock = socket.socket()
    sock.connect((params["tcp_ip"], params["sim_tcp_port"]))
    print("==> Connected to Simulation server on {}:{}".format(params["tcp_ip"], params["sim_tcp_port"]))

    # start pyrep
    scene_file = os.path.join(Path(__file__).parent, "assembly_simulator/assembly_env.ttt")
    pr = PyRep()
    pr.launch(scene_file=scene_file, headless=False)
    pr.start()

    # set base, object
    base = Dummy("base")
    objects = {}
    for obj_name in pe_obj_names:
        object_ = Object_(obj_name)
        object_.set_pose([0, 0, -3, 0, 0, 0, 1], relative_to=base)
        objects[obj_name] = object_
    # set camera
    fov_x = 2 * np.rad2deg(np.arctan(params["width"] / (2* params["K"][0])))
    with open(os.path.join(params["camera_map_path"]), "r") as json_file:
        json_str = json.load(json_file)
    tf_map_to_cam = json_message_converter.convert_json_to_ros_message('geometry_msgs/TransformStamped', json_str)
    pos, quat = transform_stamped_to_pq(tf_map_to_cam)     
    camera = Camera('Azure', [params["width"], params["height"]], fov_x, params["min_depth"],  params["max_depth"])
    camera.set_pose([*pos] + [*quat], relative_to=base)
    print("Loaded PyRep environment")


    while True:

        is_obj_ids = recvall_pickle(sock)
        pos_list = recvall_pickle(sock)
        quat_list = recvall_pickle(sock)
        rgb_real = recvall_image(sock) # [720, 1280, 3]
        depth_real = recvall_image(sock)[:, :, 0] # [720, 1280]
        mask_reals = []
        for i in range(len(is_obj_ids)):
            mask = recvall_image(sock)
            mask[mask < 128] = 0
            mask[mask > 128] = 1
            mask_reals.append(mask[:, :, 0])

        for i, is_obj_id in enumerate(is_obj_ids):
            obj_name = is_obj_names[is_obj_id]
            objects[obj_name].set_pose([*pos_list[i]] + [*quat_list[i]], relative_to=base)
            pr.step()

        final_pe_obj_ids = is_obj_ids
        final_pos_list = pos_list
        final_quat_list = quat_list

        # distinguish the part
        # by IoU => long vs short
        is_obj_ids = [int(x) for x in is_obj_ids]
        print(is_obj_ids, type(is_obj_ids[0]))
        print(is_obj_names.index("ikea_stefan_long"), type(is_obj_names.index("ikea_stefan_long")))
        longshort_idxes = np.where(is_obj_ids == is_obj_names.index("ikea_stefan_long"))
        print("longshort_idxes", longshort_idxes)
        
        long_error = []
        short_error = []
        for i, longshort_idx in enumerate(longshort_idxes):
            longshort_idx = int(longshort_idx)
            objects["ikea_stefan_long"].set_pose([*pos_list[longshort_idx]] + [*quat_list[longshort_idx]], relative_to=base)
            rgb_sim, depth_sim, mask_sim = camera.get_image()
            mask_common = np.bitwise_and(mask_reals[longshort_idx], mask_sim)
            error = get_mse_error_with_mask(rgb_real, rgb_sim, mask_common)
            long_error.append(error)

            objects["ikea_stefan_short"].set_pose([*pos_list[longshort_idx]] + [*quat_list[longshort_idx]], relative_to=base)
            rgb_sim, depth_sim, mask_sim = camera.get_image()
            mask_common = np.bitwise_and(mask_reals[longshort_idx], mask_sim)
            error = get_mse_error_with_mask(rgb_real, rgb_sim, mask_common)
            short_error.append(error)

        for i, longshort_idx in enumerate(longshort_idxes):
            if long_error[i] > short_error[i]:
                final_pe_obj_ids[longshort_idx] = pe_obj_names.index("ikea_stefan_short")
                print("it is short!")
            else:
                final_pe_obj_ids[longshort_idx] = pe_obj_names.index("ikea_stefan_long")
                print("it is long!")


            




        # by RGB => side_left vs side_right

        # distinguish the orientation
        # by RGB  => long, short, bottom
        # by Depth -> middle



        # 1. calculate iou / rgb / depth errors for various pose and objects
        # !TODO: support instance-wise mask jacquard error
        # for i, is_obj_id in enumerate(is_obj_ids):        
        #     mask_common = np.bitwise_and(mask_reals[i], mask_sim)
        #     rgb_error = get_mse_error_with_mask(rgb_real, rgb_sim, mask_common)
        #     depth_error = get_mse_error_with_mask(depth_real, depth_sim, mask_common)
        #     obj_name = is_obj_names[is_obj_id]
        #     print("{}: RGB={:4f}, Depth={:4f}".format(obj_name, rgb_error, depth_error))

        # # 2. get the pose with minimum error
        # # 3. add it to filter


        # cv2.imwrite('/home/demo/rgb.png', np.hstack([rgb_real, rgb_sim]))
        # cv2.imwrite('/home/demo/depth.png', np.hstack([depth_real, depth_sim]))
        # mask_real = mask_reals[0]
        # for mask in mask_reals:
        #     mask_real = np.bitwise_or(mask, mask_real) 
        # cv2.imwrite('/home/demo/seg_mask.png', 255*np.hstack([mask_real, mask_sim]))

        # rgb = Image.fromarray(np.uint8(255*rgb))
        # rgb.save('/home/demo/rgb.png')

        # depth = Image.fromarray(np.uint8(255*depth))
        # depth.save('/home/demo/depth.png')
        
        # # labeled on green color 
        # class_idx = np.unique(mask, axis=-1)
        # print("class_idx: ", class_idx)
        # mask = mask[:, :, 0]

        