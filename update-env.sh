#!/bin/bash

# ==============================================================================
#                 Aegis Agent End-to-End Deployment Script
#
# This script handles cleanup, building, and deploying the agent to Cloud Run
# in the us-central1 region to ensure compatibility with all Gemini models.
# ==============================================================================

# Exit immediately if any command fails
set -e

# --- Configuration ---
PROJECT_ID="aegis-prod-001"
NEW_REGION="us-central1"
SERVICE_NAME="aegis-security-agent"
NEW_REPO_NAME="adk-repo-us-central1"
IMAGE_TAG="${NEW_REGION}-docker.pkg.dev/${PROJECT_ID}/${NEW_REPO_NAME}/adk-agent:latest"


# --- Phase 1: Setup and Build in the New Region ---
echo " "
echo "--> Phase 1: Building and pushing image to '${NEW_REGION}'"
echo "---------------------------------------------------------------------"
echo "Building and submitting the container image..."
echo "IMAGE: ${IMAGE_TAG}"
gcloud builds submit \
    --tag ${IMAGE_TAG} \
    --project=${PROJECT_ID} \
    .
echo "---------------------------------------------------------------------"


# --- Phase 2: Deploy to Cloud Run in us-central1 ---
echo " "
echo "--> Phase 2: Deploying to Cloud Run in '${NEW_REGION}' with L4 GPU"
echo "---------------------------------------------------------------------"
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_TAG} \
  --region ${NEW_REGION} \
  --project ${PROJECT_ID} \
  --port 8080 \
  --allow-unauthenticated \
  --cpu 8 \
  --memory 32Gi \
  --execution-environment gen2 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${NEW_REGION},GOOGLE_GENAI_USE_VERTEXAI=true"

echo " "
echo "====================================================================="
echo "          DEPLOYMENT SCRIPT FINISHED SUCCESSFULLY! 🎉"
echo "====================================================================="
echo " "