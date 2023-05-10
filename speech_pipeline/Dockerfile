#choosing ubuntu image as our base image.
FROM ubuntu:20.04
RUN apt-get -y update && apt-get -y upgrade

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa 
RUN apt-get install -y python3.10

#install git
RUN apt-get install git -y


#creating a working directory called code
WORKDIR /code


#copying our requirements.txt file to the working directory
COPY requirements.txt .

#install pip3 and python3
RUN apt-get install -y python3-pip
RUN python3 -m pip install --upgrade pip
RUN pip3 install --default-timeout=100 future

#installing the requirements from the requirements.txt file
RUN pip3 install -r requirements.txt

#installing the whisper package and pyannote from github.
RUN pip3 install "git+https://github.com/openai/whisper.git" 
RUN pip3 install "git+https://github.com/m-bain/whisperx.git"
RUN pip3 install -q "git+https://github.com/pyannote/pyannote-audio"
RUN apt-get install -y libsndfile1-dev
RUN apt-get install -y --no-install-recommends ffmpeg
RUN apt-get install -y nano

# copy the content of the local src directory to the working directory
COPY . .

# command to run on container start
#CMD [ "python3", "main.py"]