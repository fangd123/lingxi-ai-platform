from pathlib import Path

import aioredis
from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from starlette_exporter import PrometheusMiddleware, handle_metrics
from prometheus_client import Counter
from api.utils import get_ip, model_init
from .config import settings
from .custom_logging import CustomizeLogger


def create_app():
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.DESCRIPTION,
        version=settings.VERSION,
        git_url=settings.GIT_URL,
        contact={
            "name": "Deadpoolio the Amazing",
            "url": "http://x-force.example.com/contact/",
            "email": "dp@x-force.example.com",
        },
    )

    @app.on_event('startup')
    async def startup_event():
        setup_logger(app)
        setup_redis(app)
        setup_sentry(app)
        setup_prometheus(app)
        setup_routes(app)
        setup_dl_model(app)

    @app.on_event('shutdown')
    async def shutdown_event():
        await FastAPICache.clear(namespace=settings.NAMESPACE)
        await app.redis.close()

    return app


def setup_logger(app):
    """Set up the logger."""
    config_path = Path(__file__).with_name("logging_config.json")
    app.logger = CustomizeLogger.make_logger(config_path)


def setup_redis(app):
    """Set up a Redis client."""
    app.redis = aioredis.from_url(settings.REDIS_DATABASE_URI,decode_responses=True)
    FastAPICache.init(RedisBackend(app.redis), prefix=settings.PROJECT_NAME, expire=settings.CACHE_EXPIRE_TIME)


def setup_routes(app):
    """Register routes."""
    from .routers import classify, ner
    if settings.PROJECT_TYPE == "CLS":
        app.include_router(classify.router)
    elif settings.PROJECT_TYPE == "NER":
        app.include_router(ner.router)


def setup_sentry(app):
    if '172.26' in get_ip():
        import sentry_sdk
        from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

        sentry_sdk.init(dsn=settings.SENTRY_DSN)
        app.add_middleware(SentryAsgiMiddleware)


def setup_prometheus(app):
    app.add_middleware(PrometheusMiddleware, app_name=settings.PROJECT_NAME, filter_unhandled_paths=True)
    app.add_route("/metrics", handle_metrics)
    app.TEXT_LENGTH_COUNTER = Counter(
        "texts_length_total",
        "有多少文本字数被处理了，通过app_name,method,path区分",
        ['app_name', 'method', 'path'])

    app.TEXT_LINE_COUNTER = Counter(
        "texts_line_total",
        "有多少文本行数被处理了，通过app_name,method,path区分",
        ['app_name', 'method', 'path'])


def setup_dl_model(app):
    app.dl_model, app.tokenizer, app.processor = model_init(settings.MODEL_PATH)