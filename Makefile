MELOTTS ?= 0.1.2
JETSON_BASE ?= registry.lazycat.cloud/x/lzc-aipod-onnx:5da656e
IMAGE_REPO ?= registry.lazycat.cloud/x/lzc-aipod-melotts-ov
MELO_LANGS ?= EN,ES,FR,JP,KR,ZH,ZH_MIX_EN

.PHONY: build_jetson build_jetson_latest

build_jetson:
	docker build \
		--build-arg HTTP_PROXY="http://wa.lan:7890" \
		--build-arg HTTPS_PROXY="http://wa.lan:7890" \
		--build-arg ALL_PROXY="http://wa.lan:7890" \
		--build-arg NO_PROXY="localhost,127.0.0.1" \
		--build-arg BASE_IMAGE="$(JETSON_BASE)" \
		--build-arg MELOTTS="$(MELOTTS)" \
		--build-arg MELO_LANGS="$(MELO_LANGS)" \
		-t $(IMAGE_REPO):$(MELOTTS) \
		-f Dockerfile.jetson .
