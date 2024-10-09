install:
	docker run hello-world
	docker build  -f Dockerfile -t python-template .

dev:
	docker run -v $(PWD)/container/:/opt/container \
	-v $(PWD)/data/:/opt/container/data \
	-v $(PWD)/notebook:/opt/container/notebook \
	--rm -ti --entrypoint=bash python-template

run:
	docker run -v $(PWD)/container/:/opt/container \
	-v $(PWD)/data/:/opt/container/data \
	-v $(PWD)/notebook:/opt/container/notebook \
	python-template poetry run python main.py

test:
	docker run -v $(PWD)/container/:/opt/container \
	-v $(PWD)/data/:/opt/container/data \
	-v $(PWD)/notebook:/opt/container/notebook \
	python-template poetry run pytest --color yes

lab:
	docker run -v $(PWD)/container/:/opt/container \
	-v $(PWD)/data/:/opt/container/data \
	-v $(PWD)/notebook:/opt/container/notebook \
	-p 8888:8888 \
	python-template poetry run jupyter lab --notebook-dir /opt/container/notebook \
	--no-browser --ip="0.0.0.0"
