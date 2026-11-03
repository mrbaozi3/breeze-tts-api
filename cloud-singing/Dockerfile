ARG PATCH_BASE=crpi-wg7rf21kayb2lkvr.cn-hangzhou.personal.cr.aliyuncs.com/breeze-tts/breeze-tts2-api@sha256:f8ceff17177b4d731598cc58c179880be535053bacb0bd2205d5cc65fb1cef49
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime AS full
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/opt/seed-vc/checkpoints/hf_cache HF_HUB_CACHE=/opt/seed-vc/checkpoints/hf_cache TORCH_HOME=/opt/torch-cache MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 librosa==0.10.2 soundfile==0.12.1 munch==4.0.0 einops==0.8.0 pyyaml matplotlib==3.8.4 huggingface-hub==0.28.1 transformers==4.46.3 fastapi==0.115.6 uvicorn==0.34.0 python-multipart==0.0.20
RUN git clone https://github.com/Plachtaa/seed-vc.git /opt/seed-vc && cd /opt/seed-vc && git checkout 51383efd921027683c89e5348211d93ff12ac2a8
COPY prepare.py server.py /opt/singing/
RUN python /opt/singing/prepare.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /opt/seed-vc
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "server:app", "--app-dir", "/opt/singing", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

# Reuse the verified model layers for service fixes. Use --target full to rebuild all models.
FROM ${PATCH_BASE} AS runtime
COPY server.py /opt/singing/server.py
