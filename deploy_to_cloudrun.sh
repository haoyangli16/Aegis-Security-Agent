#!/bin/bash
# AEGIS Security Co-Pilot - Cloud Run Deployment Script
# Automated deployment for Google ADK Competition
# Uses unified requirements.txt for simplified dependency management

set -e  # Exit on any error

echo "🛡️ AEGIS Security Co-Pilot - Cloud Run Deployment"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required environment variables are set
check_environment() {
    print_status "Checking environment variables..."
    
    if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
        print_error "GOOGLE_CLOUD_PROJECT environment variable is not set"
        echo "Please run: export GOOGLE_CLOUD_PROJECT=\"your-project-id\""
        exit 1
    fi
    
    if [ -z "$GOOGLE_CLOUD_LOCATION" ]; then
        print_warning "GOOGLE_CLOUD_LOCATION not set, using default: us-central1"
        export GOOGLE_CLOUD_LOCATION="us-central1"
    fi
    
    if [ -z "$SERVICE_NAME" ]; then
        print_warning "SERVICE_NAME not set, using default: aegis-security-copilot"
        export SERVICE_NAME="aegis-security-copilot"
    fi
    
    if [ -z "$APP_NAME" ]; then
        print_warning "APP_NAME not set, using default: aegis-agent"
        export APP_NAME="aegis-agent"
    fi
    
    print_success "Environment variables configured"
    echo "  Project: $GOOGLE_CLOUD_PROJECT"
    echo "  Location: $GOOGLE_CLOUD_LOCATION"
    echo "  Service: $SERVICE_NAME"
    echo "  App: $APP_NAME"
}

# Check authentication
check_auth() {
    print_status "Checking Google Cloud authentication..."
    
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_error "No active Google Cloud authentication found"
        echo "Please run: gcloud auth login"
        exit 1
    fi
    
    print_success "Authentication verified"
}

# Enable required APIs
enable_apis() {
    print_status "Enabling required Google Cloud APIs..."
    
    gcloud services enable run.googleapis.com \
        cloudbuild.googleapis.com \
        artifactregistry.googleapis.com \
        aiplatform.googleapis.com \
        compute.googleapis.com \
        --project=$GOOGLE_CLOUD_PROJECT
    
    print_success "APIs enabled successfully"
}

# Choose deployment method
choose_deployment_method() {
    echo ""
    print_status "Choose deployment method:"
    echo "1. ADK CLI (simple, may have package issues)"
    echo "2. Dockerfile (recommended, full control)"
    echo ""
    read -p "Enter choice (1 or 2, default: 2): " DEPLOY_METHOD
    DEPLOY_METHOD=${DEPLOY_METHOD:-2}
    
    if [ "$DEPLOY_METHOD" = "1" ]; then
        print_status "Using ADK CLI deployment method"
        export USE_DOCKERFILE=false
    else
        print_status "Using Dockerfile deployment method (recommended)"
        export USE_DOCKERFILE=true
    fi
}

# Deploy using ADK CLI
deploy_with_adk() {
    print_status "Deploying AEGIS using ADK CLI..."
    
    # Prepare deployment directory
    DEPLOY_DIR="aegis-deployment-$(date +%Y%m%d-%H%M%S)"
    mkdir -p $DEPLOY_DIR
    
    # Copy AEGIS code and ensure proper structure
    cp -r aegis/ $DEPLOY_DIR/
    
    # Copy viclab if it exists
    if [ -d "viclab/" ]; then
        cp -r viclab/ $DEPLOY_DIR/
    fi
    
    # Copy project files
    cp requirements.txt $DEPLOY_DIR/requirements.txt
    if [ -f "pyproject.yaml" ]; then
        cp pyproject.yaml $DEPLOY_DIR/
    fi
    if [ -f "setup.py" ]; then
        cp setup.py $DEPLOY_DIR/
    fi
    
    cd $DEPLOY_DIR
    export AGENT_PATH=./aegis
    
    # Deploy with UI enabled
    adk deploy cloud_run \
        --project=$GOOGLE_CLOUD_PROJECT \
        --region=$GOOGLE_CLOUD_LOCATION \
        --service_name=$SERVICE_NAME \
        --app_name=$APP_NAME \
        --with_ui \
        $AGENT_PATH

    cd ..
    print_success "ADK deployment completed"
    export DEPLOY_DIR
}

# Deploy using Dockerfile
deploy_with_dockerfile() {
    print_status "Deploying AEGIS using Dockerfile..."
    
    # Check if Dockerfile exists
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found. Please ensure Dockerfile exists in the project root."
        exit 1
    fi
    
    # Deploy using gcloud with Dockerfile
    gcloud run deploy $SERVICE_NAME \
        --source . \
        --region $GOOGLE_CLOUD_LOCATION \
        --project $GOOGLE_CLOUD_PROJECT \
        --allow-unauthenticated \
        --memory 4Gi \
        --cpu 2 \
        --timeout 3600 \
        --set-env-vars="GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT,GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION,GOOGLE_GENAI_USE_VERTEXAI=True"
    
    print_success "Dockerfile deployment completed"
}

# Get service information
get_service_info() {
    print_status "Retrieving service information..."
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
        --region=$GOOGLE_CLOUD_LOCATION \
        --project=$GOOGLE_CLOUD_PROJECT \
        --format="value(status.url)")
    
    print_success "Deployment completed successfully! 🎉"
    echo ""
    echo "🔗 Service URL: $SERVICE_URL"
    echo "📱 AEGIS Interface: $SERVICE_URL"
    echo ""
    
    export SERVICE_URL
}

# Test deployment
test_deployment() {
    print_status "Testing deployment..."
    
    # Wait a moment for service to start
    sleep 10
    
    # Test list apps endpoint
    print_status "Testing /list-apps endpoint..."
    if curl -f -s "$SERVICE_URL/list-apps" > /dev/null 2>&1; then
        print_success "Apps endpoint accessible"
        
        # Show available apps
        echo ""
        echo "Available apps:"
        curl -s "$SERVICE_URL/list-apps" | python3 -m json.tool 2>/dev/null || echo "Could not format JSON response"
    else
        print_warning "Apps endpoint may still be starting up"
        print_status "Waiting 30 seconds and trying again..."
        sleep 30
        
        if curl -f -s "$SERVICE_URL/list-apps" > /dev/null 2>&1; then
            print_success "Apps endpoint now accessible"
        else
            print_error "Apps endpoint still not accessible. Check logs:"
            echo "gcloud logs tail --service=$SERVICE_NAME --region=$GOOGLE_CLOUD_LOCATION"
        fi
    fi
}

# Cleanup
cleanup() {
    if [ -n "$DEPLOY_DIR" ] && [ -d "$DEPLOY_DIR" ]; then
        print_status "Cleaning up deployment directory..."
        rm -rf $DEPLOY_DIR
        print_success "Cleanup completed"
    fi
}

# Main execution
main() {
    echo "Starting AEGIS deployment process..."
    echo ""
    
    check_environment
    check_auth
    enable_apis
    choose_deployment_method
    
    # Deploy based on chosen method
    if [ "$USE_DOCKERFILE" = "true" ]; then
        deploy_with_dockerfile
    else
        deploy_with_adk
    fi
    
    get_service_info
    test_deployment
    
    echo ""
    print_success "🏆 AEGIS Security Co-Pilot is ready for the competition!"
    echo ""
    print_status "📋 Next Steps:"
    echo "  1. Open $SERVICE_URL in your browser"
    echo "  2. Test the security agent interface"
    echo "  3. Verify all tools are working"
    echo "  4. Monitor logs: gcloud logs tail --service=$SERVICE_NAME --region=$GOOGLE_CLOUD_LOCATION"
    echo ""
    
    print_status "🔧 Troubleshooting commands:"
    echo "  • Check service status: gcloud run services describe $SERVICE_NAME --region=$GOOGLE_CLOUD_LOCATION"
    echo "  • View logs: gcloud logs tail --service=$SERVICE_NAME --region=$GOOGLE_CLOUD_LOCATION"
    echo "  • Test endpoint: curl $SERVICE_URL/list-apps"
    
    cleanup
}

# Handle script interruption
trap cleanup EXIT

# Run main function
main "$@" 