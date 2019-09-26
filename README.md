# easyautoml
Easy automated machine learning

Allowing non-programmers to develop machine learning model by simplifying and automating the process.

=========================================================================

# If you are running on ubuntu.

## Dependencies
$ sudo apt-get update
$ sudo apt-get install build-essential swig


Create a virtual environment on the root directory then run

$ pip install -r requirements.txt

If error occurs, you might want to install manually the following dependencies:

$ curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

$ pip install auto-sklearn

## Running the development server.

$ python app.py


=========================================================================

# Running with docker for production and when you are not on ubuntu, although this is still under heavy development. 

Ensure that you have installed docker. On the root directory of the project.

## Build the image
docker build --tag easyautoml .

## Run a container based on the built image
docker run -p 5000:5000 -d easyautoml

## or Start the container and automatically removed on stop
docker run -p 5000:5000 --rm -d easyautoml

Then visit the app on your browser: http://127.0.0.1:5000/

=========================================================================
# Some docker commands that might help

## Force rebuild the image
docker build --no-cache .

## Delete model when already exists
docker exec -it <container_id> rm -rf ./models_trained/<model_name>


=========================================================================

# Video tutorial can be found on youtube
https://www.youtube.com/watch?v=NNnEmgt2V3s&t=21s


=========================================================================

# Note:
Support only classification.
Will work on regression.