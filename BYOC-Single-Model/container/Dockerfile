FROM python:3.9

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv (via pip só para obter o uv; depois, o resto é instalado com uv)
RUN python -m pip install --no-cache-dir uv

# Copiar código do serving
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

WORKDIR /opt/program

# Copiar pyproject.toml e uv.lock 
COPY Classificador-binario/pyproject.toml Classificador-binario/uv.lock /opt/program/

# # Forçar o uv a instalar o ambiente do projeto aqui
ENV UV_PROJECT_ENVIRONMENT=/opt/program/.venv

# Instalar dependências exatas do lockfile (uv) no ambiente do projeto
# --frozen: falha se uv.lock não corresponder ao pyproject
# --no-cache: evita cache no layer
RUN uv sync --frozen --no-cache

# Copiar o resto do código/ficheiros para a workdir
COPY Classificador-binario /opt/program

# Garantir que o script 'serve' é executável
RUN chmod 755 /opt/program/serve

# garantir que os executáveis e o python do venv são usados em runtime
ENV VIRTUAL_ENV=/opt/program/.venv
ENV PATH="/opt/program/.venv/bin:/opt/program:${PATH}"