#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# push_to_dockerhub.sh — Build and push the pipeline image to Docker Hub
#
# Usage:
#   export DOCKER_HUB_USERNAME=myusername
#   ./push_to_dockerhub.sh              # tags as :latest
#   ./push_to_dockerhub.sh 1.2.0        # tags as :1.2.0  AND  :latest
#
# Prerequisites:
#   - Run  docker login  first so credentials are cached by Docker
#   - DOCKER_HUB_USERNAME must be set in the environment
# -----------------------------------------------------------------------------
set -euo pipefail

IMAGE_NAME="fabric-pii-pipeline"
DOCKER_HUB_USERNAME="${DOCKER_HUB_USERNAME:?ERROR: export DOCKER_HUB_USERNAME=<your-hub-user> before running this script}"
VERSION="${1:-latest}"

VERSIONED_TAG="${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${VERSION}"
LATEST_TAG="${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest"

echo "==> Building ${VERSIONED_TAG}  (platform linux/amd64)"
docker build --platform linux/amd64 -t "${VERSIONED_TAG}" .

if [[ "${VERSION}" != "latest" ]]; then
    echo "==> Tagging as ${LATEST_TAG}"
    docker tag "${VERSIONED_TAG}" "${LATEST_TAG}"
fi

echo "==> Pushing ${VERSIONED_TAG}"
docker push "${VERSIONED_TAG}"

if [[ "${VERSION}" != "latest" ]]; then
    echo "==> Pushing ${LATEST_TAG}"
    docker push "${LATEST_TAG}"
fi

echo ""
echo "Done. Pull with:"
echo "  docker pull ${VERSIONED_TAG}"
