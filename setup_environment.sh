#!/bin/bash

echo "🛡️  AEGIS Security Co-Pilot - Environment Setup"
echo "=================================================="
echo "Professional AI Security Intelligence Platform"
echo ""

# Check system requirements
echo "🔍 Checking system requirements..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o 'Python [0-9]\.[0-9]' | grep -o '[0-9]\.[0-9]')
if [ "$(echo "$python_version >= 3.8" | bc 2>/dev/null)" != "1" ]; then
    echo "❌ Python 3.8+ required. Current: Python $python_version"
    exit 1
fi
echo "✅ Python version: $python_version"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    echo "⚠️  No NVIDIA GPU detected. CPU-only mode will be slower."
fi

echo ""

# Handle Google API key
if [ "$1" ]; then
    export GOOGLE_API_KEY="$1"
    export GOOGLE_GENAI_USE_VERTEXAI="False"
    echo "✅ Google API Key set from argument"
elif [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "YOUR_GOOGLE_API_KEY" ]; then
    echo "❌ Google API Key required!"
    echo ""
    echo "Usage: ./setup_environment.sh YOUR_API_KEY"
    echo "   OR: export GOOGLE_API_KEY='your_api_key_here'"
    echo ""
    echo "🔗 Get your API key from: https://aistudio.google.com/app/apikey"
    exit 1
else
    echo "✅ Google API Key already configured"
fi

echo ""
echo "🚀 Starting installation process..."
echo ""

# Step 1: Install viclab Vision Framework
echo "📹 Step 1: Installing viclab Vision Framework..."
echo "------------------------------------------------"

if [ ! -d "viclab" ]; then
    echo "❌ viclab directory not found! Make sure you're in the project root."
    exit 1
fi

cd viclab || exit 1
echo "Installing viclab dependencies..."
pip install -r requirements.txt

echo "Installing viclab package..."
pip install -e .

# Download vision model checkpoints
echo "📦 Downloading vision model checkpoints..."
mkdir -p viclab/image/checkpoints
cd viclab/image/checkpoints || exit 1

# SAM (Segment Anything Model)
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    echo "Downloading SAM checkpoint..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    echo "✅ SAM checkpoint downloaded"
else
    echo "✅ SAM checkpoint already exists"
fi

# Return to project root
cd ../../../.. || exit 1

echo "✅ viclab Vision Framework installed successfully"
echo ""

# Step 2: Install AEGIS Security Platform
echo "🛡️  Step 2: Installing AEGIS Security Platform..."
echo "------------------------------------------------"

echo "Installing AEGIS dependencies..."
pip install -r requirements.txt

echo "Installing AEGIS package..."
pip install -e .

# Install Google ADK if not present
if ! python -c "import google.adk" 2>/dev/null; then
    echo "Installing Google ADK..."
    pip install google-adk
fi

# Download YOLO11 checkpoint to aegis directory
echo "📦 Downloading YOLO11 checkpoint..."
mkdir -p aegis/checkpoints
cd aegis/checkpoints || exit 1

if [ ! -f "yolo11n.pt" ]; then
    echo "Downloading YOLO11 checkpoint..."
    wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
    echo "✅ YOLO11 checkpoint downloaded"
else
    echo "✅ YOLO11 checkpoint already exists"
fi

cd ../.. || exit 1

echo "✅ AEGIS Security Platform installed successfully"
echo ""

# Step 3: Verify installation
echo "🧪 Step 3: Verifying installation..."
echo "-----------------------------------"

# Test viclab
if python -c "from viclab.image import Dou2DTools; print('viclab: ✅')" 2>/dev/null; then
    echo "✅ viclab Vision Framework working"
else
    echo "❌ viclab installation failed"
    exit 1
fi

# Test AEGIS
if python -c "from aegis.aegis_agent import create_aegis_agent; print('AEGIS: ✅')" 2>/dev/null; then
    echo "✅ AEGIS Security Platform working"
else
    echo "❌ AEGIS installation failed"
    exit 1
fi

# Save environment variables
echo "💾 Saving environment configuration..."
cat > .env << EOF
GOOGLE_API_KEY=$GOOGLE_API_KEY
GOOGLE_GENAI_USE_VERTEXAI=False
EOF

echo ""
echo "🎉 INSTALLATION COMPLETE!"
echo "=========================="
echo ""
echo "🚀 Quick Start Options:"
echo ""
echo "1️⃣  Interactive Security Agent:"
echo "   python run_agent.py"
echo ""
echo "2️⃣  Full Web Platform:"
echo "   python aegis/adk_server.py"
echo ""
echo "3️⃣  viclab Vision Tools Demo:"
echo "   cd viclab && python image_perception_quick_start.py"
echo ""
echo "📡 Web Interfaces:"
echo "   🔗 ADK Interface: http://localhost:4001"
echo "   📺 Security Dashboard: http://localhost:4001/video"
echo ""
echo "💡 Example Commands:"
echo '   "Scan all cameras for threats"'
echo '   "Check camera 1 for suspicious packages"'
echo '   "Analyze crowd density at main entrance"'
echo ""
echo "🛡️  AEGIS Security Co-Pilot is ready to protect!" 