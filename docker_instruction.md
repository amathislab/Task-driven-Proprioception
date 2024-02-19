# Docker instruction

You need a GPU for installing the docker container. 
If you don't have docker installed:

* Install docker: https://docs.docker.com/engine/install/ubuntu/
* Create docker user:
    * `sudo groupadd docker`
    * `sudo usermod -aG docker $USER`
    * `newgrp docker`
* Download [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit):
  * `sudo apt-get install -y nvidia-container-toolkit`
  * `sudo nvidia-ctk runtime configure --runtime=docker`
  * `sudo systemctl restart docker`

If docker is already installed, you can proceed by downloading the [docker image](https://hub.docker.com/repository/docker/vargas95/opensim): `docker pull vargas95/opensim:opensim-tf`

To run the container (make sure that you are in the folder that contains dlc-docker file):
```
GPU=0 bash ./dlc-docker run -d -p 2359:8888 --cpuset-cpus="0-15" -v /media1:/media1 --name containername vargas95/opensim:opensim-tf
```

You can modify the previous command by:

* changing port: (i.e. 2359 can be 777, etc)
* change which GPU to use (check which GPU you want to use in the terminal by running nvidia-smi)
* change the name: --name containername can be anything you want
* change the volume: -v /media1:/media1 can be changed to any volume


Enter the container via the terminal (to get terminal access in container):

```
docker exec --user $USER -it containername /bin/bash
```

### Jupyter and docker

You can access Jupyter by going to the port you specified (e.g. http://localhost:2359) in Google Chrome. 

To get the token for entry, back in the terminal, look at the docker log:

```
docker logs containername 
```

Copy and paste the value after "token=". 