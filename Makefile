.DEFAULT_GOAL := help

##@ Utility
.PHONY: help
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

## Docker
.PHONY: build
build: ## Builds docker image from Dockerfile
	@docker build -t slidev .

.PHONY: serve
serve: ## Serves Slidev project
	@docker compose up -d

.PHONY: export
export: ## Exports SLidev slides
	@docker exec -i nlp1-alignment-slidev npx slidev export --timeout 2m --output /export/slides.pdf --with-clicks --wait 1000
