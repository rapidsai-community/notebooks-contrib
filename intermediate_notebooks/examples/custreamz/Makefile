.PHONY: all build run

all: build run

build: 
	docker build -t notebooks-contrib/custreamz .

run:
	docker run -p 8880:8888 -p 8788:8787 notebooks-contrib/custreamz:latest
