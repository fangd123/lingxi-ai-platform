FROM python:3.8-slim
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR ./
COPY deploy/requirements-CPU.txt ./
RUN pip install --no-cache-dir torch==1.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements-CPU.txt -i https://mirrors.aliyun.com/pypi/simple/
COPY deploy ./deploy
COPY *.py ./
COPY optimize/quantify/bert.opt.quant.onnx ./deploy
WORKDIR ./deploy
# the entry point
EXPOSE 8000
CMD bash run_app_onnx.sh