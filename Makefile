IMAGE_NAME := layout-corrector
DEV_CONTAINER_NAME := $(IMAGE_NAME)-dev
PORT_START := 10000
PORT_END := 10005
TENSORBOARD_PORT := 10006

ROOT := /app
CMD = "/bin/bash"


.PHONY: build
build:
	docker build -t $(DEV_CONTAINER_NAME) --build-arg WORKDIR=$(ROOT) .

.PHONY: run
run:
	docker run -it --name $(DEV_CONTAINER_NAME) \
		--gpus all \
		-p $(PORT_START)-$(PORT_END):$(PORT_START)-$(PORT_END) \
        -p $(TENSORBOARD_PORT):6006 \
		-v `pwd`:$(ROOT) \
		--shm-size 32GB \
		$(DEV_CONTAINER_NAME) \
		$(CMD)
