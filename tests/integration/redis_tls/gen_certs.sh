#!/usr/bin/env bash
# Generate a self-signed CA + server cert for the redis-sdk-tls test service.
#
# Output (all git-ignored, NEVER commit): tests/integration/redis_tls/certs/
#   ca.crt / ca.key       — local test CA
#   redis.crt / redis.key — server cert signed by that CA (CN/SAN = localhost)
#
# Usage:
#   tests/integration/redis_tls/gen_certs.sh
#   docker compose --profile tls-test up redis-sdk-tls
#   pytest tests/integration/test_redis_session_tls.py
set -euo pipefail

CERT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/certs"
mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

# 1) Local test CA
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 \
  -subj "/CN=continuum-test-ca" -out ca.crt

# 2) Server key + CSR (localhost, with SAN so hostname verification passes)
openssl genrsa -out redis.key 2048
openssl req -new -key redis.key -subj "/CN=localhost" -out redis.csr

# 3) Sign the server cert with the CA
openssl x509 -req -in redis.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out redis.crt -days 825 -sha256 \
  -extfile <(printf "subjectAltName=DNS:localhost,IP:127.0.0.1")

rm -f redis.csr ca.srl
chmod 644 ./*.crt ./*.key  # redis container reads them read-only
echo "Certs written to $CERT_DIR"
