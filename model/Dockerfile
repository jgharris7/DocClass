ARG PYTHONVERSION=3.7
ARG MODEL_NAME=v2

FROM public.ecr.aws/lambda/python:${PYTHONVERSION} as docclass_1
ARG PYTHONVERSION

COPY requirements.txt ./
RUN python${PYTHONVERSION} -m pip install -r requirements.txt -t .

FROM docclass_1
ARG PYTHONVERSION
ARG MODEL_NAME
COPY  ./app/model.py ./
COPY  ./app/app.py ./
COPY  ./${MODEL_NAME}.pckmdl ./


CMD ["app.lambda_handler"]
