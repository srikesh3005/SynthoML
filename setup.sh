#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   CTGAN Synthetic Data Generator - Automated Setup        ║"
echo "║   Optimized for Mac M2 (Apple Silicon)                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed!${NC}"
    echo "Please install Python 3.9 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"

if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed!${NC}"
    echo "Please install Node.js 18+ from https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
echo -e "${GREEN}✓ Node.js ${NODE_VERSION} found${NC}"

if ! command -v npm &> /dev/null; then
    echo -e "${RED}npm is not installed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ npm $(npm --version) found${NC}"

echo -e "\n${YELLOW}[2/6] Setting up Python virtual environment...${NC}"

cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"

echo -e "\n${YELLOW}[3/6] Installing Python dependencies...${NC}"
echo "This may take a few minutes..."

pip install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python dependencies installed successfully${NC}"
else
    echo -e "${RED}Failed to install some dependencies${NC}"
    echo -e "${YELLOW}Trying fallback installation with ctgan...${NC}"
    pip install pandas numpy ctgan joblib fastapi uvicorn[standard] --quiet
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Fallback installation successful${NC}"
    else
        echo -e "${RED}Installation failed. Please check the error messages above.${NC}"
        exit 1
    fi
fi

echo -e "\n${YELLOW}[4/6] Training CTGAN model...${NC}"
echo "This will take a few minutes depending on your system..."

if [ -f "ctgan_model.joblib" ]; then
    echo -e "${YELLOW}Model file already exists.${NC}"
    read -p "Do you want to retrain? (y/N): " retrain
    if [[ $retrain =~ ^[Yy]$ ]]; then
        python train_and_save_ctgan.py --data toy_medical.csv --epochs 100
    else
        echo -e "${GREEN}✓ Using existing model${NC}"
    fi
else
    python train_and_save_ctgan.py --data toy_medical.csv --epochs 100
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model trained and saved successfully${NC}"
    else
        echo -e "${RED}Model training failed${NC}"
        exit 1
    fi
fi

echo -e "\n${YELLOW}[5/6] Testing inference...${NC}"

python test_inference.py --n 10 > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Inference test passed${NC}"
else
    echo -e "${RED}Inference test failed${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[6/6] Setting up frontend...${NC}"

cd "$PROJECT_DIR/frontend"

if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Frontend dependencies already installed${NC}"
fi

cd "$PROJECT_DIR"

echo -e "\n${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ Setup Complete!                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Next Steps:${NC}\n"

echo -e "${YELLOW}Terminal 1 - Start Backend:${NC}"
echo "  cd \"$PROJECT_DIR\""
echo "  source venv/bin/activate"
echo "  python server.py"
echo ""

echo -e "${YELLOW}Terminal 2 - Start Frontend:${NC}"
echo "  cd \"$PROJECT_DIR/frontend\""
echo "  npm run dev"
echo ""

echo -e "${BLUE}Access Points:${NC}"
echo "  Frontend: http://localhost:5173"
echo "  API:      http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""

echo -e "${BLUE}Documentation:${NC}"
echo "  README.md          - Comprehensive guide"
echo "  COMMANDS.md        - Command reference"
echo "  PROJECT_SUMMARY.md - Project overview"
echo ""

echo -e "${GREEN}Happy Synthetic Data Generation! 🎉${NC}"
