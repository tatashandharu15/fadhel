$ErrorActionPreference = 'Stop'

docker compose up -d
docker update --memory 8g automotive-rag-api
docker restart automotive-rag-api
docker inspect --format '{{.Name}} {{.HostConfig.Memory}}' automotive-rag-api