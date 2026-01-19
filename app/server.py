import uvicorn
from app.core.config import settings

def main():
    uvicorn.run("app.main:app", host=settings.service.host, port=settings.service.port, log_level=settings.service.log_level)
