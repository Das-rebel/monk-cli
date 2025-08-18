# 🚀 Deployment Guide for Monk CLI

> **Complete deployment instructions for the TreeQuest AI agent system**

## 📋 **Quick Start**

### **1. Clone and Install**
```bash
git clone https://github.com/yourusername/monk-cli.git
cd monk-cli
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **2. Configure API Keys**
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

### **3. Test Installation**
```bash
python3 monk.py --treequest /agents
```

## 🔧 **Production Deployment**

### **Systemd Service**
```ini
[Unit]
Description=Smart AI Enhanced CLI
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/smart-ai-enhanced
ExecStart=/path/to/smart-ai-enhanced/venv/bin/python smart_ai_enhanced_v3.py --treequest
Restart=always

[Install]
WantedBy=multi-user.target
```

### **Docker**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "smart_ai_enhanced_v3.py", "--treequest"]
```

## 📊 **Monitoring**
- Check logs: `tail -f logs/smart-ai.log`
- Health check: `curl http://localhost:8000/health`
- Resource usage: `htop`

## 🔒 **Security**
- Never commit API keys
- Use environment variables
- Enable firewall rules
- Regular security updates

For detailed instructions, see the main README.md
