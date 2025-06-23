#!/bin/bash
# AEGIS Security Co-Pilot - Quick Vertex AI Setup Script
# Simplified version that skips potentially problematic steps

set -e  # Exit on any error

echo "🛡️ AEGIS Security Co-Pilot - Quick Vertex AI Setup"
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

# Quick Python check
check_python() {
    print_status "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_success "Python $PYTHON_VERSION found"
}

# Install Vertex AI SDK
install_vertex_ai() {
    print_status "Installing Vertex AI SDK with Agent Engine support..."
    
    pip3 install --upgrade "google-cloud-aiplatform[adk,agent_engines]" || {
        print_warning "pip3 failed, trying pip..."
        pip install --upgrade "google-cloud-aiplatform[adk,agent_engines]"
    }
    
    print_success "Vertex AI SDK installed successfully"
}

# Quick gcloud check (no updates)
check_gcloud() {
    print_status "Checking Google Cloud CLI..."
    
    if ! command -v gcloud &> /dev/null; then
        print_error "Google Cloud CLI not found"
        print_error "Please install it manually: https://cloud.google.com/sdk/docs/install"
        exit 1
    else
        print_success "Google Cloud CLI found"
        gcloud version
    fi
}

# Setup authentication
setup_auth() {
    print_status "Setting up Google Cloud authentication..."
    
    # Check if already authenticated
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_success "Already authenticated with Google Cloud"
        CURRENT_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
        print_status "Current account: $CURRENT_ACCOUNT"
    else
        print_status "Please authenticate with Google Cloud..."
        gcloud auth login
        print_success "Authentication completed"
    fi
    
    # Set up application default credentials
    print_status "Setting up application default credentials..."
    if gcloud auth application-default print-access-token &> /dev/null; then
        print_success "Application default credentials already configured"
    else
        gcloud auth application-default login
        print_success "Application default credentials configured"
    fi
}

# Configure project
configure_project() {
    print_status "Configuring Google Cloud project..."
    
    # Get current project or prompt for one
    CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
    
    if [ -z "$CURRENT_PROJECT" ]; then
        echo ""
        print_status "Available projects:"
        gcloud projects list --limit=10
        echo ""
        read -p "Enter your Google Cloud Project ID: " PROJECT_ID
        
        if [ -z "$PROJECT_ID" ]; then
            print_error "Project ID is required"
            exit 1
        fi
        
        gcloud config set project $PROJECT_ID
        export GOOGLE_CLOUD_PROJECT=$PROJECT_ID
    else
        export GOOGLE_CLOUD_PROJECT=$CURRENT_PROJECT
        print_success "Using project: $CURRENT_PROJECT"
    fi
    
    # Set location
    read -p "Enter location (default: us-central1): " LOCATION
    LOCATION=${LOCATION:-us-central1}
    export GOOGLE_CLOUD_LOCATION=$LOCATION
    
    print_success "Project configuration completed"
    echo "  Project: $GOOGLE_CLOUD_PROJECT"
    echo "  Location: $GOOGLE_CLOUD_LOCATION"
}

# Enable required APIs
enable_apis() {
    print_status "Enabling required Google Cloud APIs..."
    
    gcloud services enable \
        aiplatform.googleapis.com \
        compute.googleapis.com \
        storage.googleapis.com \
        --project=$GOOGLE_CLOUD_PROJECT
    
    print_success "APIs enabled successfully"
}

# Create staging bucket
create_staging_bucket() {
    print_status "Creating staging bucket for Vertex AI..."
    
    BUCKET_NAME="$GOOGLE_CLOUD_PROJECT-aegis-staging"
    STAGING_BUCKET="gs://$BUCKET_NAME"
    
    # Check if bucket exists
    if gsutil ls $STAGING_BUCKET > /dev/null 2>&1; then
        print_success "Staging bucket already exists: $STAGING_BUCKET"
    else
        print_status "Creating staging bucket: $STAGING_BUCKET"
        gsutil mb -l $GOOGLE_CLOUD_LOCATION $STAGING_BUCKET
        print_success "Staging bucket created successfully"
    fi
    
    export STAGING_BUCKET
}

# Save environment configuration
save_config() {
    print_status "Saving environment configuration..."
    
    cat > .env.vertex_ai << EOF
# AEGIS Security Co-Pilot - Vertex AI Configuration
export GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT
export GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION
export STAGING_BUCKET=$STAGING_BUCKET
EOF
    
    print_success "Configuration saved to .env.vertex_ai"
}

# Main setup function
main() {
    echo "Quick setup for AEGIS Vertex AI deployment..."
    echo ""
    
    check_python
    install_vertex_ai
    check_gcloud
    setup_auth
    configure_project
    enable_apis
    create_staging_bucket
    save_config
    
    echo ""
    print_success "🎉 Quick setup completed!"
    echo ""
    print_status "📋 Next Steps:"
    echo "1. Load environment configuration:"
    echo "   source .env.vertex_ai"
    echo ""
    echo "2. Deploy AEGIS to Vertex AI:"
    echo "   python3 deploy_to_vertex_ai.py"
    echo ""
}

# Run main function
main "$@" 