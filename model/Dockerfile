ARG PYTHONVERSION=3.7
ARG MODEL_NAME=linSVCv0
ARG MODEL_TYPE=DocClfTLinSVC

FROM public.ecr.aws/lambda/python:${PYTHONVERSION} as docclass_1
ARG PYTHONVERSION

COPY requirements.txt ./
RUN python${PYTHONVERSION} -m pip install -r requirements.txt -t .

FROM docclass_1
ARG PYTHONVERSION
ARG MODEL_NAME
ARG MODEL_TYPE
COPY  ./app/${MODEL_TYPE}.py ./
COPY  ./app/app.py ./
COPY  ./${MODEL_NAME}.pckmdl ./
COPY  ./${MODEL_NAME}testdata.txt ./
ENV model_name=${MODEL_NAME}
RUN python -m ${MODEL_TYPE}
RUN python -m app

CMD ["app.lambda_handler"]
