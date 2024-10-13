from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseSettings, HttpUrl, validator,RedisDsn


class Settings(BaseSettings):
    # 项目基本信息
    # 项目名
    PROJECT_NAME: str = 'dl_template'
    # 项目类型 NER 或者 CLS
    PROJECT_TYPE: str = 'NER'
    # 模型路径
    MODEL_PATH: str = str(Path('./models',"title_classify_2.onnx"))
    MODEL_VOCAB_FOLDER_PATH: str = str(Path('./models'))
    # 项目描述
    DESCRIPTION: str = "这是一个示例"
    # 版本
    VERSION: str = '0.1.0'
    # git 地址
    GIT_URL: Optional[HttpUrl] = None
    # 项目负责人信息
    CONTACT = {
                  "name": "Deadpoolio the Amazing",
                  "url": "http://x-force.example.com/contact/",
                  "email": "dp@x-force.example.com",
              },

    # sentry 地址
    SENTRY_DSN: Optional[HttpUrl] = ''

    @validator("SENTRY_DSN", pre=True)
    def sentry_dsn_can_be_blank(cls, v: str) -> Optional[str]:
        if len(v) == 0:
            return None
        return v

    # redis 缓存信息
    CACHE_EXPIRE_TIME: int = 60

    REDIS_SERVER: str = '192.168.1.19'
    REDIS_USER: Optional[str]
    REDIS_PASSWORD: Optional[str]
    REDIS_DB: Optional[str]
    REDIS_DATABASE_URI: Optional[RedisDsn] = 'redis://192.168.1.19'

    @validator("REDIS_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return RedisDsn.build(
            scheme="redis",
            user=values.get("REDIS_USER"),
            password=values.get("REDIS_PASSWORD"),
            host=values.get("REDIS_SERVER"),
            path=f"/{values.get('REDIS_DB') or ''}",
        )

    NAMESPACE: str = 'onnx'

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()