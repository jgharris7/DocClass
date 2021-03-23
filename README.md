# DocClass
This project builds and deploys a simple document classifier onto Amazon AWS. The current version is set up to use support vector machine with the linear kernal from scikit learn
# Prerequisites:
Model training was done with scikit-learn in Anaconda with python 3.88
For deployment to AWS I have used Windows WSL (Ubuntu, WSL2).
Deployment requirments:
 docker
 Amazon AWS tools SAM and CALI
# Directories and files
data --- the model training data. You need to unzip the compressed datafile
model --- the files for building the deployed model
model/app -- the source code for the app and the model learning code. Includes some prior classifiers that have been superceded
http/ --- the website interface for running the model
test/ --- some test cases for some versions of the model
model/Dockerfile --- Docker configuration file
model/Template.yaml --- configuration file for SAM to build the Lambda function and API Gateway

# Deployment
## Deploy model itself:
The model is deployed from the "model" directory.
### Step 1 Configuration of Dockerfile and Template.yaml
To deploy the model, insure the Dockerfile and Template.yaml have the same value for the "model_name". The key lines are:

Dockerfile:
```
ARG MODEL_NAME=linSVCv0
ARG MODEL_TYPE=DocClfTLinSVC
```
Template.yaml:
```
model_name: linSVCv0
```

Also, you may should modify the stackname and S3 bucket names to have ones that are for your implementation. The stack name is in the "deploy.sh" script. 

The model code resides in the app/DocClfTLinSVC.py file, the stored "learned" model object is in the file with the model name+".pckmdl" extension, and two test cases are in the file with the model name+"testdata.txt". Both of the latter two should reside in the model/ directory. If you "relearn" the model, the configuration is currently set up to create the model object file and test case in the main project directory to be copied down if the result is satisfactory.
I have a role "S3LambdaRead" setup that has AmazonS3ReadOnlyAccess and AWSLambdaBasicExecutionRole attached. The current version does not put files on the S3, but just includes all files in the build. The next step is just to execute the build. 

### Step 2 Build the model and deploy it
To build the model, execute in the model directory:
```
sam build
```
This will create the docker container, execute a unit test on the the model to confirm that the results of the two test cases match those from the "predict" method that was applied when the learner computed the testing accuracy.
To test the integration and then deploy the model type in the model directory and test a sample request against the deployment:
```
./buildtest.sh
./deploy.sh
```
### Step 3 Capture the URL for the model and test the Lambda Service implementation.
Navigate to the Lambda function console. Select the function that was created from the deployment. It would have the stack name+function name and some random alphanumeric characters in it.  On the menu, select "Configuration" and under "API Gateway", open the "Details" item. The URL to use is the "API endpoint". Edit the curl.sh file to use this URL instead of the one there.
```
./curlit.sh
```
## Webpage service
The web interface for the model is in the http directory. To use the webpage to access the model, change the "siteUrl" line in docClass.js to use the URL obtained above. This webpage could be hosted locally or in the cloud.
