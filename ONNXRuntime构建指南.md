# 基于CUDA环境下ONNXRuntime的Web服务镜像构建指南

以下构建脚本将国外网址替换为了对应的镜像网址，若想使用原版地址，可以参考
[dockerfiles的cuda构建指南](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles#cuda)

1. 克隆ONNXRuntime指定版本分支
```shell
git clone -b v1.9.1 https://hub.fastgit.org/microsoft/onnxruntime.git
grep -rl "github.com" . | xargs sed -i 's/github.com/github.com.cnpmjs.org/g'
git submodule update --init
```
2. 进入`dockerfiles`文件夹中，修改`dockerfile.cuda`文件
```dockerfile
FROM nvcr.io/nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
MAINTAINER Changming Sun "chasun@microsoft.com"
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ADD . /code
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && apt-get update && apt-get install -y --no-install-recommends python3-dev ca-        certificates g++ python3-numpy gcc make git python3-setuptools python3-wheel python3-pip aria2 && aria2c -q -d /tmp -o cmake-3.21.0-linux-x86_64.tar.gz https://pd. zwc365.com/https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz && tar -zxf /tmp/cmake-3.21.0-linux-x86_64.tar.gz --         strip=1 -C /usr
RUN cd /code && /bin/bash ./build.sh --skip_submodule_sync --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_cuda --config Release --       build_wheel --update --build --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) 'CMAKE_CUDA_ARCHITECTURES=37;50;52;60;70'
```
3. 构建编译镜像
```shell
docker build -t gpu_onnx -f Dockerfile.cuda ..
```
4. 构建Web服务镜像
创建`dockerfile.cuda.web`文件，输入以下内容
```dockerfile
FROM gpu_onnx:latest
MAINTAINER Fang Wenda "fangd123@qq.com"
FROM nvcr.io/nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04
COPY --from=0 /code/build/Linux/Release/dist /root
COPY --from=0 /code/dockerfiles/LICENSE-IMAGE.txt /code/LICENSE-IMAGE.txt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 ca-certificates        python3-setuptools python3-wheel python3-pip unattended-         upgrades && unattended-upgrade &&          python3 -m pip install /root/*.whl  --no-cache-dir -i  https://mirrors.aliyun.com/pypi/simple && rm -rf /root/*.whl &&   rm -rf /var/lib/apt/lists/*
COPY . /opt
#RUN  sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list  && apt-get update && apt-get install -y --no-install-recommends software-           properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get install  -y --no-install-recommends python3.8 python3-pip && python3.8 -m pip install --no-   cache-dir pip && rm -rf /var/lib/apt/lists/*
RUN python3.8 -m pip install fastapi uvicorn loguru transformers starlette_exporter sentry_sdk aioredis pydantic[dotenv]  /opt/*.whl --no-cache-dir -i https://     mirrors.aliyun.com/pypi/simple
WORKDIR /app
CMD ["uvicorn", "run_app:app", "--host", "0.0.0.0", "--port", "80"]
```
执行命令，构建镜像
```shell
docker build -t onnxruntime-cuda -f Dockerfile.cuda.web .
```
5. 上传镜像至阿里云服务

此步骤需要阿里云账号的用户名密码，一般不需要执行
```shell
docker login --username=[阿里云镜像用户名] [阿里云镜像仓库地址]
docker tag [ImageId] [阿里云镜像仓库地址]/gpu_bert_web:[镜像版本号]
docker push [阿里云镜像仓库地址]/gpu_bert_web:[镜像版本号]
docker pull [阿里云镜像仓库地址]/gpu_bert_web:[镜像版本号]
```