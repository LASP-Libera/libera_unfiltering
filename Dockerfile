FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl gcc ca-certificates && \
    update-ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Put project in container
# Add requirements file in the container - this would be done with git clone for established projects
COPY pyproject.toml ./pyproject.toml
# Add source code in the container - this would be done with git clone for established projects
COPY ./unfiltered_radiances/algorithm.py ./algorithm.py
# Add product definition file in the container
COPY ./unfiltered_radiances/l2-unfiltered-radiance-product-definition.yml ./l2-unfiltered-radiance-product-definition.yml
# Add the coefficient generatation code in the container
COPY ./prod/std/standard_method.py ./standard_method.py


# Handle installs
# Create virtual environment and permanently activate it for this image
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
# This adds not only the venv python executable but also all installed entrypoints to the PATH
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Upgrade pip to the latest version because poetry uses pip in the background to install packages
RUN pip install --upgrade pip
# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
# Add poetry to path
ENV PATH="$PATH:/root/.local/bin"
# Install libera_utils and all its (non-dev) dependencies according to pyproject.toml
RUN poetry lock && poetry sync --only main --no-root

# Define container entry point
ENTRYPOINT ["python", "algorithm.py"]
CMD [""]