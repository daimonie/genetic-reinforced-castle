from python:3.12.3

# update the debian packages
RUN apt-get update
RUN apt install pre-commit -y

# install poetry
RUN pip install poetry
RUN poetry --version


# copy container directory into docker image, move to it
COPY container/ /opt/container
WORKDIR /opt/container

# Create a group and user
RUN groupadd -r python-role && useradd -r -g python-role pythonuser

# Additional read/write for container & home
RUN chmod a+rw /opt/container/
RUN chmod a+rw /home

# Switch to this user
USER pythonuser

# install poetry from pypoetry.toml
RUN poetry install
