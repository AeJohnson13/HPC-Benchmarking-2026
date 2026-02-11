# Use cuda pytorch image
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Copy repo into container
COPY . . 

# install any requirements

RUN pip install --no-cache-dir -r requirements.txt

# Run Python Program 
CMD [ "python", "./main_DDP.py"]
