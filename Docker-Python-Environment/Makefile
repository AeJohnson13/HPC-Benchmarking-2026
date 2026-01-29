go:
	echo "Building Docker image"
	docker build -t aejohnson13/my-pytorch-app:1.0 . 
	echo "Running container"
	docker run --rm -it --shm-size=4g --gpus all my-pytorch-app