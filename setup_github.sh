#!/bin/bash

# 🧘 Monk CLI - GitHub Repository Setup Script
# This script will help you set up your GitHub repository

echo "🌟 Monk CLI - GitHub Repository Setup"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed. Please install Git first.${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "monk.py" ]; then
    echo -e "${RED}❌ Please run this script from the Monk CLI project directory${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Git is installed${NC}"
echo -e "${GREEN}✅ Monk CLI project found${NC}"
echo ""

# Show current git status
echo -e "${BLUE}📊 Current Git Status:${NC}"
git status --short
echo ""

# Show commit history
echo -e "${BLUE}📜 Recent Commits:${NC}"
git log --oneline -5
echo ""

echo -e "${YELLOW}🚀 Ready to set up GitHub repository!${NC}"
echo ""
echo "📋 Next Steps:"
echo "1. Go to: https://github.com/new"
echo "2. Repository name: monk-cli"
echo "3. Description: 🧘 Monk CLI - Enhanced with TreeQuest AI Agents"
echo "4. Make it Public (recommended)"
echo "5. ✅ Add README file"
echo "6. ✅ Add .gitignore (Python)"
echo "7. ✅ Choose MIT License"
echo "8. Click 'Create repository'"
echo ""
echo -e "${GREEN}After creating the repository, run:${NC}"
echo "  ./setup_github.sh --push"
echo ""

# Check if --push flag is provided
if [ "$1" = "--push" ]; then
    echo -e "${BLUE}🚀 Pushing to GitHub...${NC}"
    echo ""
    
    # Check if remote exists
    if git remote get-url origin &> /dev/null; then
        echo -e "${YELLOW}⚠️  Remote 'origin' already exists. Updating...${NC}"
        git remote remove origin
    fi
    
    # Add new remote
    echo -e "${BLUE}🔗 Adding GitHub remote...${NC}"
    git remote add origin https://github.com/Das-rebel/monk-cli.git
    
    # Verify remote
    echo -e "${BLUE}✅ Remote configured:${NC}"
    git remote -v
    echo ""
    
    # Push to GitHub
    echo -e "${BLUE}📤 Pushing code to GitHub...${NC}"
    if git push -u origin main; then
        echo ""
        echo -e "${GREEN}🎉 Successfully pushed to GitHub!${NC}"
        echo ""
        echo "🌐 Your repository is now available at:"
        echo "   https://github.com/Das-rebel/monk-cli"
        echo ""
        echo "📋 Next steps:"
        echo "   • Enable GitHub Pages in repository settings"
        echo "   • Set up repository topics: cli, ai, python, treequest, monk"
        echo "   • Create your first release"
        echo "   • Share with the community!"
        echo ""
    else
        echo -e "${RED}❌ Failed to push to GitHub${NC}"
        echo "Please check your GitHub repository URL and try again."
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}✨ Setup script completed!${NC}"
