install:
	docker run --memory=16g hello-world
	docker build -f Dockerfile -t genetic-castle .

dev:
	docker run -v $(PWD)/container/:/opt/container \
	--memory=16g \
	-v $(PWD)/data/:/opt/container/data \
	-v $(PWD)/notebook:/opt/container/notebook \
	--rm -ti --entrypoint=bash genetic-castle

run:
	docker run -v $(PWD)/container/:/opt/container \
	--memory=16g \
	-v $(PWD)/data/:/opt/container/data \
	-v $(PWD)/notebook:/opt/container/notebook \
	genetic-castle poetry run python main.py

test:
	docker run -v $(PWD)/container/:/opt/container \
	--memory=16g \
	-v $(PWD)/data/:/opt/container/data \
	-v $(PWD)/notebook:/opt/container/notebook \
	genetic-castle poetry run python -m pytest --color yes

lab:
	docker run -v $(PWD)/container/:/opt/container \
	--memory=16g \
	-v $(PWD)/data/:/opt/container/data \
	-v $(PWD)/notebook:/opt/container/notebook \
	-p 8888:8888 \
	genetic-castle poetry run jupyter lab --notebook-dir /opt/container/notebook \
	--no-browser --ip="0.0.0.0"
