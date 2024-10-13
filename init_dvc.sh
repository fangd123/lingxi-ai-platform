dvc init
git commit .dvc -m "Initialize DVC"
# 配置Minio 的 endpointurl 和 access key
dvc remote add -d myremote [s3://data]
dvc remote modify myremote endpointurl [http://s3.123.456.com]
export AWS_SECRET_ACCESS_KEY=""
export AWS_ACCESS_KEY_ID="minio"
git commit .dvc/config -m "Configure remote storage"
dvc commit


