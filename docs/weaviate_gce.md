## Self-hosted Weaviate on Google Compute Engine

This guide shows how to run Weaviate yourself on a small Compute Engine VM. You pay only for the VM + disk + any optional GCS backups, rather than the fully managed Marketplace subscription.

### 1. Plan the resources
- **Machine type**: `e2-standard-2` (2 vCPU / 8 GB RAM) is a balanced starting point. Drop to `e2-medium` for lower cost or scale up for heavier workloads.
- **Disk**: 50–100 GB SSD persistent disk mounted at `/var/lib/weaviate`.
- **Region**: pick the same region you use for the rest of the stack (e.g., `us-central1`) to minimize latency.
- **Network**: allow TCP 8080 (HTTP) and 50051 (gRPC). Restrict source ranges to your office IPs or VPC peers whenever possible.

### 2. Create the VM
```bash
gcloud compute instances create weaviate-vm \
  --zone=us-central1-a \
  --machine-type=e2-standard-2 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-type=pd-ssd \
  --boot-disk-size=100GB \
  --tags=weaviate \
  --scopes=https://www.googleapis.com/auth/devstorage.read_write

# Open ports (restrict --source-ranges to your needs)
gcloud compute firewall-rules create weaviate-http `
  --allow "tcp:8080,tcp:50051" `
  --target-tags=weaviate `
  --source-ranges=0.0.0.0/0

```

> For production, front the VM with Cloud Load Balancing + HTTPS and lock down the firewall to known clients or private service connect.

### 3. Bootstrap Weaviate
SSH into the VM and run the provided script:
```bash
gcloud compute ssh weaviate-vm --zone=us-central1-a
# On the VM:
curl -sSf https://raw.githubusercontent.com/<your-repo>/scripts/weaviate_vm_setup.sh | bash
```

The script installs Docker, creates `/var/lib/weaviate`, and runs the official `semitechnologies/weaviate` container with API-key auth enabled. Customize the env vars inside the script (see below).

### 4. Environment variables set by the script
| Variable | Purpose |
| --- | --- |
| `AUTHENTICATION_APIKEY_ALLOWED_KEYS` | Comma-separated API keys accepted by Weaviate. Set this to a long random string. |
| `AUTHENTICATION_APIKEY_USERS` | Labels attached to those keys (e.g., `admin`). |
| `PERSISTENCE_DATA_PATH` | Mounted to `/var/lib/weaviate` so vectors survive restarts. |
| `QUERY_DEFAULTS_LIMIT`, `ENABLE_MODULES` | Optional tuning knobs; left at defaults in the script. |

To rotate the API key, stop the container (`docker stop weaviate`), update the env vars, and rerun `docker run ...`.

### 5. Connect the application
Update `.env` in this repo:
```
WEAVIATE_URL=http://<EXTERNAL_IP>:8080
WEAVIATE_API_KEY=<the key you configured>
WEAVIATE_COLLECTION=LegalChunk
WEAVIATE_GRPC_PORT=50051
```

The ingestion + FastAPI services will now talk to your self-hosted instance. No additional code changes are required.

### 6. Cost & maintenance tips
- **Stop/start**: shut down the VM when not ingesting to avoid compute charges (persistent disk costs continue, but are small).
- **Preemptible VMs**: if occasional restarts are acceptable, run as `--preemptible` to save ~70%. Use the setup script in a startup script to auto-respawn the container.
- **Backups**: run Weaviate’s GCS backup command when needed (set `BACKUP_GCS_*` env vars) or snapshot the persistent disk periodically.
- **Monitoring**: enable Cloud Monitoring or run `docker logs weaviate` to spot ingestion/query errors.

Once this VM is reachable, continue with the ingestion workflow documented in `README.md`.
