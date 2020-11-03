# Assembly Furniture Recognition

## Features
- Furniture recognition toolkits for furniture assembly project (IKEA STEFAN)
- Instance segmentation using [Mask R-CNN](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)
- 6d pose estimation using [Augmented AutoEncoder (AAE)](https://openaccess.thecvf.com/content_ECCV_2018/html/Martin_Sundermeyer_Implicit_3D_Orientation_ECCV_2018_paper.html) and [Multi-Path Augmented AutoEncoder (MPAAE)](https://openaccess.thecvf.com/content_CVPR_2020/html/Sundermeyer_Multi-Path_Learning_for_Object_Pose_Estimation_Across_Domains_CVPR_2020_paper.html)

![mpaae_sample](./MPAAE_sample.png)
![recognition_demo](./recognition_demo.gif)


## To Do

- list up ROS dependencies

## Getting Started

- python 2.7 
- tensorflow == 1.14
- torch 1.3.0
- torchvision 0.4.1
- [azure_kinect_ros_driver](https://github.com/microsoft/Azure_Kinect_ROS_Driver)
- [zivid_ros_driver](https://github.com/zivid/zivid-ros)
- [open3d-ros-helper](https://github.com/SeungBack/open3d-ros-helper)
- [assembly_msgs](https://github.com/psh117/assembly_msgs)
- [assembly_camera_manager](https://github.com/SeungBack/assembly_camera_manager)
- [AugmentedAutoEncoder](https://github.com/DLR-RM/AugmentedAutoencoder/tree/multipath)
```
# setup environment using anaconda3
conda env create -f environment.yml
conda activate assembly 
# install autopose (AAE)
cd src
git clone https://github.com/SeungBack/AugmentedAutoencoder.git
cd AugmentedAutoencoder && git checkout multipath
pip install --user .
# install ros dependencies
pip install -U rosdep rosinstall_generator wstool rosinstall six vcstools
```


## Published Topics
#### `/assembly/vis_is`
- message type: `sensor_msgs/Image`
- Visualization results of instance segmentation 

#### `/assembly/vis_pe_aae` 
- message type: `sensor_msgs/Image`
- Visualization results of 6d pose estimation from AAE

#### `/assembly/markers/aae` 
- message type: `visualization_msgs/MarkerArray`
- Visualization markers for AAE

#### `/assembly/markers/icp` 
- message type: `visualization_msgs/MarkerArray`
- Visualization markers for AAE + ICP

#### `/assembly/detections/aae` 
- message type: `vision_msgs/Detection3DArray`
- 6d pose estimation results from AAE

#### `/assembly/detections/icp` 
- message type: `vision_msgs/Detection3DArray`
- 6d pose estimation results from AAE + ICP


## How to use
### Single Camera Setup 
1. Launch k4a driver
```
$ ROS_NAMESPACE=azure1 roslaunch azure_kinect_ros_driver driver.launch color_resolution:=720P depth_mode:=NFOV_2X2BINNED fps:=5  tf_prefix:=azure1_
```
2. Launch k4a manager 
```
$ roslaunch assembly_camera_manager single_azure_manager.launch target_fiducial_id:="0"
```
3. Set camera pose
```
$ rosservice call /azure1/set_camera_pose "json_file: 'map_to_azure1_rgb_camera_link_20201102-183839'"
```
4. 6d object pose estimation using MPAAE
```
$ roslaunch assembly_part_recognition single_azure_detr_mpaae.launch yaml:=single_azure_detr_mpaae_GIST
# in python 3.6
$ ass36 & python /home/demo/catkin_ws/src/assembly_part_recognition/src/detr_client.py
```
5. visualization using RVIZ
```
rosrun rviz rviz -d single_azure.rviz
```

## Multi Camera Setup 
1. launch k4a driver
```
$ ROS_NAMESPACE=azure3 roslaunch azure_kinect_ros_driver driver.launch sensor_sn:=000880594512 wired_sync_mode:=2 subordinate_delay_off_master_usec:=500 fps:=5 color_resolution:=720P depth_mode:=NFOV_2X2BINNED tf_prefix:=azure3_ rgb_point_cloud:=true

$ ROS_NAMESPACE=azure2 roslaunch azure_kinect_ros_driver driver.launch sensor_sn:=000853594412 wired_sync_mode:=2 subordinate_delay_off_master_usec:=250 fps:=5 color_resolution:=720P depth_mode:=NFOV_2X2BINNED tf_prefix:=azure2_ rgb_point_cloud:=true

$ ROS_NAMESPACE=azure1 roslaunch azure_kinect_ros_driver driver.launch sensor_sn:=000256194412 wired_sync_mode:=1 color_resolution:=720P depth_mode:=NFOV_2X2BINNED fps:=5 tf_prefix:=azure1_ rgb_point_cloud:=true
```

# Calibrate Multi K4a Network
$ roslaunch assembly_camera_manager triple_azure_manager.launch
$ rosservice call /triple_azure/extrinsic_calibration "target_fiducial_ids: [1, 3, 13]"

## Authors
* **Seunghyeok Back** [seungback](https://github.com/SeungBack)

## License
This project is licensed under the MIT License

## Acknowledgments
This work was supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by Korea goverment(MSIT) (No.2019-0-01335, Development of AI technology to generate and validate the task plan for assembling furniture in the real and virtual environment by understanding the unstructured multi-modal information from the assembly manual.