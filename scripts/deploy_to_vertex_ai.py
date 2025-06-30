#!/usr/bin/env python3
"""
AEGIS Security Co-Pilot - Vertex AI Agent Engine Deployment Script
Deploy AEGIS to Vertex AI Agent Engine for GPU-powered security monitoring
"""

import os
import sys
import time
import vertexai
from vertexai import agent_engines
from vertexai.preview import reasoning_engines

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import our AEGIS agent
from aegis.aegis_agent.aegis_security_agent import create_aegis_agent

# Color codes for output
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

def print_status(message):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def print_success(message):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

def print_warning(message):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

def print_error(message):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def get_environment_config():
    """Get and validate environment configuration"""
    print_status("🛡️ AEGIS Security Co-Pilot - Vertex AI Deployment")
    print("=" * 60)
    
    # Get configuration from environment or prompt user
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        project_id = input("Enter your Google Cloud Project ID: ").strip()
        if not project_id:
            print_error("Project ID is required")
            sys.exit(1)
    
    location = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
    staging_bucket = os.environ.get('STAGING_BUCKET')
    if not staging_bucket:
        staging_bucket = f"gs://{project_id}-aegis-staging"
        print_warning(f"Using default staging bucket: {staging_bucket}")
        print_warning("Make sure this bucket exists or create it manually")
    
    return {
        'project_id': project_id,
        'location': location,
        'staging_bucket': staging_bucket
    }

def initialize_vertex_ai(config):
    """Initialize Vertex AI with project configuration"""
    print_status(f"Initializing Vertex AI...")
    print(f"  Project: {config['project_id']}")
    print(f"  Location: {config['location']}")
    print(f"  Staging Bucket: {config['staging_bucket']}")
    
    try:
        vertexai.init(
            project=config['project_id'],
            location=config['location'],
            staging_bucket=config['staging_bucket'],
        )
        print_success("Vertex AI initialized successfully")
        return True
    except Exception as e:
        print_error(f"Failed to initialize Vertex AI: {e}")
        return False

def create_and_prepare_agent():
    """Create and prepare the AEGIS agent for deployment"""
    print_status("Creating AEGIS Security Agent...")
    
    try:
        # Create the AEGIS agent
        aegis_agent = create_aegis_agent()
        print_success(f"AEGIS agent created with {len(aegis_agent.tools)} security tools")
        
        # Wrap agent for Agent Engine deployment
        print_status("Preparing agent for Vertex AI Agent Engine...")
        app = reasoning_engines.AdkApp(
            agent=aegis_agent,
            enable_tracing=True,
        )
        print_success("Agent prepared for deployment")
        
        return aegis_agent, app
        
    except Exception as e:
        print_error(f"Failed to create agent: {e}")
        return None, None

def deploy_to_agent_engine(agent, config):
    """Deploy the agent to Vertex AI Agent Engine"""
    print_status("Deploying AEGIS to Vertex AI Agent Engine...")
    print_warning("This may take several minutes...")
    
    try:
        # Prepare requirements for deployment
        requirements = [
            "google-cloud-aiplatform[adk,agent_engines]",
            "google-adk>=1.3.0",
            "opencv-python",
            "pillow",
            "numpy",
            "pandas",
            "requests",
            "faster-whisper",
            "sqlalchemy",
            "python-dotenv",
            "pydantic",
            "uvicorn",
            "fastapi"
        ]
        
        # Deploy to Agent Engine
        remote_app = agent_engines.create(
            agent_engine=agent,
            requirements=requirements
        )
        
        print_success("🎉 AEGIS deployed to Vertex AI Agent Engine!")
        return remote_app
        
    except Exception as e:
        print_error(f"Deployment failed: {e}")
        return None

def test_deployment(remote_app):
    """Test the deployed agent"""
    print_status("Testing deployed AEGIS agent...")
    
    try:
        # Create a test session
        print_status("Creating test session...")
        test_session = remote_app.create_session(user_id="test_user")
        session_id = test_session["id"]
        print_success(f"Test session created: {session_id}")
        
        # Test with a security query
        print_status("Testing security monitoring capability...")
        test_query = "Please analyze the current security status and list available cameras"
        
        response_parts = []
        for event in remote_app.stream_query(
            user_id="test_user",
            session_id=session_id,
            message=test_query,
        ):
            response_parts.append(event)
            
        print_success("Agent responded successfully!")
        print("Sample response:")
        for part in response_parts[-3:]:  # Show last 3 parts
            if 'parts' in part and part['parts']:
                for p in part['parts']:
                    if 'text' in p:
                        print(f"  📝 {p['text'][:100]}...")
                        break
        
        return True
        
    except Exception as e:
        print_error(f"Testing failed: {e}")
        return False

def get_usage_info(remote_app, config):
    """Display usage information"""
    print_success("🏆 AEGIS Security Co-Pilot is ready!")
    print("=" * 60)
    
    resource_name = remote_app.resource_name
    print(f"🔗 Resource Name: {resource_name}")
    
    print("\n📋 Next Steps:")
    print("1. Use the Vertex AI Console to manage your agent")
    print("2. Create sessions and interact with AEGIS programmatically")
    print("3. Monitor usage and performance in Vertex AI")
    
    print("\n🔧 Example Usage (Python):")
    print(f"""
import vertexai
from vertexai import agent_engines

# Initialize Vertex AI
vertexai.init(
    project="{config['project_id']}",
    location="{config['location']}"
)

# Get your deployed agent
remote_app = agent_engines.get("{resource_name}")

# Create a session
session = remote_app.create_session(user_id="your_user_id")

# Send security queries
for event in remote_app.stream_query(
    user_id="your_user_id",
    session_id=session["id"],
    message="Analyze camera feed 1 for suspicious activity"
):
    print(event)
""")
    
    print("\n🌐 Access via REST API:")
    print("Your agent is accessible via Vertex AI's REST API endpoints")
    print("Check the Vertex AI documentation for API details")

def main():
    """Main deployment function"""
    try:
        # Get configuration
        config = get_environment_config()
        
        # Initialize Vertex AI
        if not initialize_vertex_ai(config):
            sys.exit(1)
        
        # Create and prepare agent
        agent, app = create_and_prepare_agent()
        if not agent:
            sys.exit(1)
        
        # Deploy to Agent Engine
        remote_app = deploy_to_agent_engine(agent, config)
        if not remote_app:
            sys.exit(1)
        
        # Test deployment
        if test_deployment(remote_app):
            get_usage_info(remote_app, config)
        else:
            print_warning("Deployment completed but testing failed")
            print_warning("Your agent may still be starting up")
            get_usage_info(remote_app, config)
        
    except KeyboardInterrupt:
        print_warning("\nDeployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 