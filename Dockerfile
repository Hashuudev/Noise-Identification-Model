FROM python:3.9.16


RUN mkdir build

# We create folder named build for our stuff.

WORKDIR /build

# Basic WORKDIR is just /
# Now we just want to our WORKDIR to be /build

COPY . .

# FROM [path to files from the folder we run docker run]
# TO [current WORKDIR]
# We copy our files (files from .dockerignore are ignored)
# to the WORKDIR
RUN apt-get update && apt-get install -y ffmpeg


RUN pip install --no-cache-dir -r requirements.txt

# OK, now we pip install our requirements

EXPOSE 80

# Instruction informs Docker that the container listens on port 80


# Now we just want to our WORKDIR to be /build/app for simplicity
# We could skip this part and then type
# python -m uvicorn main.app:app ... below

CMD  python main.py
