FROM python:3.9

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv (via pip só para obter o uv; depois, o resto é instalado com uv)
RUN python -m pip install --no-cache-dir uv

# Dependências do servidor
RUN uv pip install --system --no-cache-dir flask gevent gunicorn

# Copiar código do serving
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY Classificador-binario /opt/program
WORKDIR /opt/program

# Instalar dependências da inferência (do teu modelo) via uv
# (este requirements.txt deve estar dentro de pasta classificador-binario)
RUN uv pip install --system --no-cache-dir -r requirements.txt
