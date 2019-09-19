docker pull hirune924/cuda-conda-deep:latest

cur_path=$(cd $(dirname $0) && pwd)
mnt_path=${cur_path}
mnt_target='/root'

## visible gpus
nvidia_visible_devices='all'
#nvidia_visible_devices='0,1'

#xhost +
xhost local:

time nvidia-docker run \
	--env http_proxy=$http_proxy \
	--env https_proxy=$https_proxy \
	--env NVIDIA_VISIBLE_DEVICES=$nvidia_visible_devices \
	--ipc=host \
	-p 80:80 \
	-p 8888:8888 \
	-v ${mnt_path}:${mnt_target} \
	-w /root \
	-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
	-it --rm hirune924/cuda-conda-deep:latest

#xhost -
xhost -local:

