conda activate mti830-3d
python ./lidar-train.py
cd /media/antoinebrassardlahey/NVME/darknet
./waymo_train.sh
cd ~/MTI830