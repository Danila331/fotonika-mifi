FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Папки для входных изображений и результатов (монтируйте их с хоста)
VOLUME ["/in", "/out"]

# По умолчанию — демонстрационный комплект из репозитория; со своими данными см. README
ENTRYPOINT ["python", "predict.py"]
CMD ["--in-dir", "/app/data/demo/shapes", "--out", "/out"]
