FROM python:3.12-slim

# The model keys weights by hash(str). Python randomises that per process unless this is
# pinned, which would make two runs of the same image score differently.
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp is LightGBM's OpenMP runtime. The wheel imports it at load time, so without this
# the install succeeds and the import fails.
RUN apt-get update -qq && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
RUN pip install --no-cache-dir "numpy==2.3.4" "lightgbm==4.6.0"

COPY app/ /srv/app/

# Build the targeting index and the ANN index once, here, and ship them in the image. A
# container then starts by loading them instead of deriving them, which took 48 s on an idle
# machine and over three minutes on a loaded one -- paid again on every restart.
RUN python -m app.build_index

EXPOSE 8080
CMD ["python", "-m", "app.serve"]
