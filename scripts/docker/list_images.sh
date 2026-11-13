#!/bin/bash
set -euo pipefail
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | sort
