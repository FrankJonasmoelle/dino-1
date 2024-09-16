# Run DINO on imagenet

- sudo docker build -t dino .

- sudo docker run -it --rm --runtime nvidia --shm-size=1g -v /home/scaleout-orin/zenseact-project/dino:/dino -v /home/scaleout-orin/zenseact-project/archive/tiny-imagenet-200/train:/dino/train dino 

- python3 dino_wrapper.py