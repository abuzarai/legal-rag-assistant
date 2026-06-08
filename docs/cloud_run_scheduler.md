Cloud Run ingestion trigger via Cloud Scheduler

Overview
- Cloud Scheduler periodically calls the Cloud Run endpoint `POST /ingest`.
- The request uses OIDC authentication; the Scheduler job’s service account needs the Cloud Run Invoker role on the service.

Prereqs
- gcloud CLI authenticated and configured for your project
- Deployed Cloud Run service (note its URL and service account)

Create a Scheduler job
1) Set variables
```
PROJECT_ID=your-project
REGION=us-central1
SERVICE_NAME=legal-rag-api
SCHEDULE="0 * * * *"   # hourly; adjust as needed
INGEST_URL=https://<cloud-run-url>/ingest
SCHEDULER_SA=projects/$PROJECT_ID/serviceAccounts/scheduler-invoker@$PROJECT_ID.iam.gserviceaccount.com
```

2) Create the service account and grant roles
```
gcloud iam service-accounts create scheduler-invoker \
  --project $PROJECT_ID --display-name "Scheduler Invoker"

gcloud run services add-iam-policy-binding $SERVICE_NAME \
  --region $REGION --member serviceAccount:scheduler-invoker@$PROJECT_ID.iam.gserviceaccount.com \
  --role roles/run.invoker --project $PROJECT_ID
```

3) Create the job with OIDC auth
```
gcloud scheduler jobs create http ingest-legal-rag \
  --project $PROJECT_ID --location $REGION \
  --schedule "$SCHEDULE" --time-zone "Etc/UTC" \
  --http-method POST --uri "$INGEST_URL" \
  --oidc-service-account-email scheduler-invoker@$PROJECT_ID.iam.gserviceaccount.com \
  --oidc-token-audience "$INGEST_URL"
```

Notes
- Share the root Drive folder (and anything it references via shortcuts) with the Cloud Run service account email for access.
- To persist ingestion state across revisions, set `INGESTION_STATE_BACKEND=gcs` and provide a bucket.
- If you keep `file` backend, state persists only within the container instance lifecycle.
