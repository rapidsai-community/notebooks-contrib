.PHONY: all build run

DOCKER_ID=$(shell docker ps --no-trunc -q)

#docker inspect --format="{{.Id}}"

all: run

build: 
	docker build -t notebooks-extended .

run: build
	docker run -p 8888:8888 -it notebooks-extended

login:
	docker exec -it $(DOCKER_ID) bash

clean:
	rm -rf tutorials

copy: clean
	docker cp $(DOCKER_ID):/rapids/notebooks/extended/tutorials .

convert:
	jupyter nbconvert --to script foo.ipynb 

