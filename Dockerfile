FROM --platform=linux/amd64 docker.1ms.run/pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# GDAL 需系统库；与 libsqlite 同装自 conda-forge，避免 libgdal.so 运行时找不到 sqlite3_total_changes64
RUN conda install -y -c conda-forge gdal sqlite && conda clean -afy
# 运行时优先加载 conda 的 lib，保证 libgdal 链接到 conda 的 libsqlite
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}

# 排除 torch（base 已有）、gdal（上一步已装），其余用 pip 安装
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN sed -e '/^torch==/d' -e '/^gdal==/d' requirements.txt > requirements-docker.txt && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    rm requirements-docker.txt

WORKDIR /App

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PROJ_PATH=/App
ENV PYTHONPATH=/App/src
ENV DATA_DIR_PATH=Data

CMD ["bash"]
