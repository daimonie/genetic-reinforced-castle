install:
	docker run --memory=16g hello-world
	docker build -f Dockerfile -t genetic-castle .

dev:
	docker run -v $(PWD)/container/:/opt/container \
	--memory=16g \
	-v $(PWD)/data/:/opt/container/data \
	--rm -ti --entrypoint=bash genetic-castle

run:
	docker run -v $(PWD)/container/:/opt/container \
	--memory=16g \
	-v $(PWD)/data/:/opt/container/data \
	genetic-castle poetry run python main.py

test:
	docker run -v $(PWD)/container/:/opt/container \
	--memory=16g \
	-v $(PWD)/data/:/opt/container/data \
	genetic-castle poetry run python -m pytest --color yes
 