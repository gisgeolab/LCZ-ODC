setup:
	build up init

build:
	DOCKER_BUILDKIT=1 docker compose build

up:
	docker compose up -d

init:
	docker compose exec -T lcz-odc datacube -v system init

down:
	docker compose down