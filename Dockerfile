FROM python:3.12.3
# Copy container directory into docker image
COPY container/ /opt/container

# Set working directory
WORKDIR /opt/container

# Install poetry using pip
RUN curl -sSL https://install.python-poetry.org | python3 -
# Add poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Verify poetry installation
RUN poetry --version

# Install dependencies using Poetry
RUN poetry install 