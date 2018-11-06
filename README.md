# SCV-Net - Sparse Cost Volume for Efficient Stereo Matching
Our work is build on the GC-Net.
The GC-Net is proposed in "End-to-end learning of geometry and context for deep stereo regression", by Kendall et al. in ICCV 2017.
By making the cost volume compact and proposing an efficient similarity evaluation, we achieved faster stereo matching while improving the accuracy.
Moreover, we proposed to use weight normalization instead of batch normalization.
This improved the performance at dim and noise regions.
Finally, we achieved 70% GPU memory and 60% processing time reducing, while improving the matching accuracy (3PE on the KITTI 2015 Dataset, GC-Net: 2.87% -> Ours: 2.61%).

# System requirement
----Python 3.6

--------PyTorch 0.3.0

--------torchvision 0.1.8

--------pypng 0.0.18

--------pillow 4.2.1

--------numpy 1.13.3

--------matplotlab 2.1.0

# Note
In "SCVNet/SCVNet.py", replace

----SCENE_FLOW_TRAIN_PATH_IMAGE

----SCENE_FLOW_TRAIN_PATH_LABEL

----SCENE_FLOW_TEST_PATH_IMAGE

----SCENE_FLOW_TEST_PATH_LABEL

----KITTI_2015_TRAIN_PATH_IMAGE

----KITTI_2015_TRAIN_PATH_LABEL

----KITTI_2015_TEST_PATH_IMAGE

----KITTI_2015_TEST_PATH_LABEL

with your own path.
