# Run DINO on imagenet

- sudo docker build -t dino .

- sudo docker run -it --rm --runtime nvidia --network host --shm-size=1g -v /home/scaleout-orin/zenseact-project/dino:/dino -v /home/scaleout-orin/zenseact-project/data:/dino/data dino 



