# The "buster" flavor of the official docker Python image is based on Debian and includes common packages.
FROM python:3.7-buster

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Install Python dependencies
COPY env/requirements.txt ./init_requirements.txt
RUN wget https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt 
RUN sed -i 's/cu101/cpu/' requirements.txt
RUN pip install --upgrade pip~=21.0.0
RUN pip install -r init_requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Copy only the relevant directories
COPY . /repo 

# Run the web server
EXPOSE 5000
ENV PYTHONPATH /repo
#CMD python3 /repo/app/restapi.py &
CMD python3 /repo/app.py
