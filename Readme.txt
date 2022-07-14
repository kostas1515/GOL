#This code was tested in RHEL 8 system

#Train predefined models on standard datasets (COCO 2017) with MMDetection v2.21.0. Refer to the following websites:
https://github.com/open-mmlab/mmdetection
https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md
https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md

# Steps:
#1. Prepare environment
module load gcc/8.4.0
# download Miniconda3-py38_4.10.3-Linux-ppc64le.sh, and install the conda base enviroment without installing any extra packages

#configure channel settings
conda config --set channel_priority strict
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
conda config --prepend channels https://opence.mit.edu

# create a virtual environment
conda create --name mmdet pytorch=1.7.1 -y
source activate mmdet

#install dependency packages
conda install torchvision -y
conda install pandas scipy -y
conda install opencv -y

#2. Install MMDetection
pip install openmim
mim install mmdet==2.21.0

#pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.1/index.html
#pip install mmdet==2.21.0


#download datasets
# create data directory, download COCO 2017 datasets at https://cocodataset.org/#download
# (2017 Train images [118K/18GB], 2017 Val images [5K/1GB], 2017 Train/Val annotations [241MB])
# extract the zip files
mkdir data
cd data
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

#download and unzip LVIS annotations
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip


#4. Run the training
#modify mmdetection/configs/_base_/datasets/lvis_v1_instance.py and make sure data_root variable points to the above data directory, e.g.,
data_root = '<user_path>'

#Training on multiple GPUs. Use tools/dist_train.sh to launch training on multiple GPUs.
#submit a job with 4 GPUs, and make sure the job script includes:
./tools/dist_train.sh ./configs/<experiment>/<variant.py> 4

#To train GOL use:
./tools/dist_train.sh ./configs/droploss/droploss_normed_mask_r50_rfs_4x4_2x_gumbel.py 4

#To test GOL:
./tools/dist_test.sh ./experiments/droploss_normed_mask_rcnn_r50_rfs_4x4_2x_gumbel/droploss_normed_mask_cascade_r50_rfs_4x4_2x_gumbel.py ./experiments/droploss_normed_mask_cascade_r50_rfs_4x4_2x_gumbel/latest.pth 4 --eval bbox segm


