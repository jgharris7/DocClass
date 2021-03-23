# -*- coding: utf-8 -*-
"""

Lambda Handler function
Created on Sat Mar 20 10:07:28 2021

@author: jgharris
"""


import json
import os
import boto3
import pickle
import sys
s3=boto3.client('s3')
#s3_bucket=os.environ['s3_bucket']
model_name=os.environ['model_name']
temp_file_path='./'
#
# headers set to avoid CORS errors
#   Currently CORS errors appear to be due to sending too large a URL request
#
headers={
        "Access-Control-Allow-Headers": "Origin, X-Requested-With, Content-Type,  Accept",
        "Access-Control-Allow-Origin" : "*"
        }
from DocClfTLinSVC import DocClfTLinSVC

def lambda_handler(event,context):
   statusCode=0
   try:
    params=event["queryStringParameters"]
    input=params['words']
   except:
       print("Unexpected error (query invalid):", sys.exc_info()[0])
       return {
        "statusCode": -200,
        "headers": headers,
        "body": json.dumps(
            {
            "prediction": str(sys.exc_info()[0]),
            "confidence": str(" none ")
            }
            )
        }
   statustCode=-100
   prediction=" "
   confidence=" "
   try:
       # Easier to just put the model file in the build.
       # shouldn't add much size 
       '''
       s3.download_file(s3_bucket,model_name+".pckmdl",
                      temp_file_path+model_name+".pckmdl")
                      '''
       statustCode=-1
       modelFile=temp_file_path+model_name+".pckmdl"
       myModel=pickle.load(open(modelFile,"rb"))
       statusCode=-2
       prediction=myModel.predict([input])[0]
       statusCode=-3
       confidence=myModel.getConfidence(input,prediction)
       statusCode=0
   except:
         print("Unexpected error:", sys.exc_info()[0])
         return {
        "statusCode": statusCode,
        "headers": headers,
        "body": json.dumps(
            {
            "prediction": prediction,
            "confidence": confidence,
            "model_name": model_name,
            "model_file": modelFile,
            "errMSG": str(sys.exc_info()[0])
            }
            )
        }
    
   return {
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps({
            "prediction": str(prediction),
            "confidence": str(confidence),
            "model_name": model_name,
            "description": myModel.message
        })
    }

def main():
   xtest='133d46f7ed38 96e70b2e2fc0 586242498a88 cde4f1b2a877 6bf9c0cb01b4 0562c756a2f2 fbb5efbcc5b3 d4e08985be1b 78c3a5c15b68 9ad186d42f69 6b343f522f78 bad6ff5dd7bc c337a85b8ef9 40f3a08093c6 65f888439937 2784a2673880 2272194f2c48 0969e9a2a900 9f11111004ec 1015893e384a 5e99d31d8fa4 d18e8c6fa60a c73d303c1dcb dc83f2b00468 24c356d262ce 818a7ff3bf29 578830762b27 9dc34464aa01 2e33ce9d2c13 e9f1f22efed4 fe081ae57a8b 7d9e333a86da ba02159e05b1 892d541c89eb 0e17d653f006 cf2205dbb077 fa736fae0d12 6b223a390d86 6ce6cc5a3203 2396ef1fa71b efaed70caea1 b208ae1e8232 c18666450d27 7ec02e30a5b3 ba3c06a73274 fdf32f896cc3 f36e139d9400 bf39bb85076d 1641a72fa752 1015893e384a 586242498a88 3012dd989e4f 6bf9c0cb01b4 041a934b1778 b32153b8b30c 95ef80a0b841 93790ade6682 a65259ff0092 7d9e333a86da e432c4501410 b61f1af56200 fe081ae57a8b d1dc04ffccd4 63fa15c4caa9 b136f6349cf3 8a3fc46e34c1 6d25574664d2 9cdf4a63deb0 b59e343416f7 6b7dd05245fd 0562c756a2f2 d774c0d219f8 6d40ee0e021c f02c3660e718 0562c756a2f2 4e5019f629a9 036087ac04f9 eeb86a6a04e4 98d0d51b397c ef237222adfc 1450168c4ac2 22b937b93ed2 f6f466726339 395a5e8185f8 07b4174549d4 b9925442c9c9 bfeaa5b4f65a ef4ba44cdf5f 9a42ead47d1c 73801426ea65 0226fe922dd0 376aa3d8142d e1a726bff8dc ed5d3a65ee2d 26f768da5068 5cc9caec5d01 b32153b8b30c 6a01047db3ab be9f9e5522c9 6bf9c0cb01b4 6d1fb90988cf'
   correctresult='DELETION OF INTEREST'
   queryStringParameters={"words":xtest}
   event={"queryStringParameters":queryStringParameters}
   global temp_file_path
   modelFile=temp_file_path+model_name+".pckmdl"
   if(not os.path.exists(modelFile)):
      temp_file_path='../'
   x=lambda_handler(event=event,context="")
   print(x)
   queryStringParameters={"words":xtest}
   event={"queryStringParameters":queryStringParameters}
   lambda_handler(event=event,context="")
if __name__=='__main__':
    main()         
   
