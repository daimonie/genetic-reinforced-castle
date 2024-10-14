FROM python:3.12.3

# Create a non-root user
RUN useradd -m -s /bin/bash python_user

# Copy container directory into docker image
COPY container/ /opt/container

# Set working directory
WORKDIR /opt/container

# Install poetry
RUN pip install poetry==1.7.1

# Install dependencies globally
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Set permissions for python_user
RUN chown -R python_user:python_user /opt/container
RUN chmod -R 755 /opt/container

# Switch to python_user
USER python_user

# Set the working directory again (now as python_user)
WORKDIR /opt/container

# Set POETRY_VIRTUALENVS_CREATE to false for python_user
ENV POETRY_VIRTUALENVS_CREATE=false
