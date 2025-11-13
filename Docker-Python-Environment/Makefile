go:
	echo "Building Docker image"
	docker build -t my-pytorch-app .
	echo "Running container"
	docker run --rm -it --shm-size=4g --gpus all my-pytorch-app