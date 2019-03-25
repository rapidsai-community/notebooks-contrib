.PHONY: all build run

all: build run

build: 
	docker build -t notebooks-extended .

run:
	docker run -p 8888:8888 -it notebooks-extended

