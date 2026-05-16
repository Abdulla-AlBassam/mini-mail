FROM python:3.11-slim

# libmilter-dev is the C library pymilter wraps. gcc + python3-dev are
# only needed during pip install (pymilter compiles a C extension); we
# could two-stage this to drop them later, but image size has not been
# a concern for this project.
# Source: pymilter installation docs (https://github.com/sdgathman/pymilter)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmilter-dev gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (includes torch + transformers; sized for both classifiers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK corpora that preprocessing.py and the attack scripts use. Download
# once at build time so the running container has zero network deps and
# can start up quickly.
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

# Only the milter script lives in the image. The ML code and the model
# files are bind-mounted at /app/ml at runtime via docker-compose.yml,
# so I can iterate on either without rebuilding the image.
COPY spam_milter.py .

EXPOSE 9900

CMD ["python", "spam_milter.py"]
