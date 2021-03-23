# DocClass
This project builds and deploys a simple document classifier onto Amazon AWS. The current version is set up to use support vector machine with the linear kernal from scikit learn
# Prerequisites:
Model training was done with scikit-learn in Anaconda with python 3.88
For deployment to AWS I have used Windows WSL (Ubuntu, WSL2).
Deployment requirments:
 docker
 Amazon AWS tools SAM and CALI
# Directories and files
data -- the model training data. You need to unzip the compressed datafile
model -- the files for building the deployed model
model/app -- the source code for the app and the model learning code. Includes some prior classifiers that have been superceded
http/ -- the website interface for running the model
test/ -- some test cases for some versions of the model
model/Dockerfile -- Docker configuration file
model/Template.yaml -- configuration file for SAM to build the Lambda function and API Gateway
model/app/app.py -- the Lambda serverless API that is called.
model/app/DocClfTLinSVC.py -- the actual Model implementation
model/linSVCv0.pckl -- the trained model object stored by pickle. This is read in for making predictions.
model/linSVCv0testdata.txt -- sample test cases from the accuracy calculation on the test holdout sample. Used for Unit tests.
model/requirements.txt -- required python modules used to build the docker container
model/Template.yaml -- Template used by AWS SAM for creating the docker file, Lambda function, and API endpoint
model/Dockerfile -- docker configuration script that gives instructions to create the docker container and run test cases
model/deploy.sh -- script to deploy the model to AWS Lambda cloud
model/curlit.sh -- simple test using curl to verify operation of the API

http/index.html -- web page for accessing the model
http/docClass.js -- javascript code for the webpage
http/docClassstyles.css -- style sheets needed for the webpage
# Other files not required for actual deployment
model/app/learnLinSVC.py -- actually used to train the model
model/app/Documents.py  -- Object to load data for training the model
model/app/*.py -- Other model objects (Naive Bayes, Complement Naive Bayes) and training file
data/*.zip -- zip file of training set (large)

# Deployment
## Deploy model itself:
The model is deployed from the "model" directory.
### Step 1 Configuration of Dockerfile and Template.yaml
The key things to be customized are:
1. The model name (model_name or MODEL_NAME) in the Dockerfile and Template.yaml file.
2. The unit_model_name in the python Model file (app/DocClfTLinSVC.py) MUST match
the model name in the Dockerfile and Template.yaml file or the model unit tests will fail. The model_name set in the "learnLinSVC.py" file does not need updating for the deployment, but local unit tests will fail if run in the IDE after training the model.
3. The rolename in the Template.yaml file
4. The name of the AWS ECR (Elastic Contain Repository at https://console.aws.amazon.com/ecr )in the deploy.sh script
5. The stack-name in the deploy.sh script
6. The python Model Object used (only if you are changing to a different class of models). in the Dockerfile
7. The URL in curlit.sh and in ../http/docClass.js


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

Also, you should modify the ECR, the stackname and S3 bucket names to have ones that are for your implementation. The stack name is in the "deploy.sh" script. 

The model code resides in the app/DocClfTLinSVC.py file, the stored "learned" model object is in the file with the model name+".pckmdl" extension, and two test cases are in the file with the model name+"testdata.txt". Both of the latter two should reside in the model/ directory. 
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
