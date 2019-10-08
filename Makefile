.PHONY: all build run

all: build run

build: 
	docker build -t notebooks-contrib .

run:
	docker run -it -p 8888:8888 -p 8787:8787 -p 8786:8786 notebooks-contrib:latest

