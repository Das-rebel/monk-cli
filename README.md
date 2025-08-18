# 🧘 Monk CLI - Intelligent Query Routing with TreeQuest AI Agents

> **Enterprise-Grade AI Orchestration System with Smart Query Routing and Multi-Agent Intelligence**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com/Das-rebel/monk-cli)

## 🌟 **Overview**

Monk CLI is a revolutionary command-line interface that combines **Intelligent Query Routing** with the power of multiple AI agents through the TreeQuest engine. This system automatically routes your queries to the optimal AI provider based on query type, complexity, cost, and performance requirements, providing enterprise-grade AI capabilities with unmatched efficiency.

### **🎯 Key Features**

- **🧠 Intelligent Query Routing**: Smart routing to optimal AI providers based on query analysis
- **🤖 Multi-Agent AI System**: Planner, Analyzer, Critic, Synthesizer, and Executor agents
- **🧠 TreeQuest Engine**: Adaptive Branching Monte Carlo Tree Search for LLM orchestration
- **💰 Cost Optimization**: Real-time cost tracking and intelligent model selection
- **📊 Enhanced Analytics**: Cross-tool intelligence and correlation
- **🚀 High Performance**: Async processing with intelligent caching

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Monk CLI                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Planner   │  │  Analyzer   │  │   Critic    │        │
│  │    Agent    │  │   Agent     │  │   Agent     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Synthesizer │  │  Executor   │  │  Simulator  │        │
│  │    Agent    │  │   Agent     │  │   Agent     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    TreeQuest Engine                        │
│              (Adaptive Branching MCTS)                    │
├─────────────────────────────────────────────────────────────┤
│                    Model Registry                          │
│              (Multi-Provider AI Models)                   │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 **Intelligent Query Routing**

Monk CLI's **Intelligent Query Routing** system automatically analyzes your queries and routes them to the optimal AI provider, ensuring the best performance, cost efficiency, and results for every interaction.

### **🔄 How It Works**

1. **Query Analysis**: Analyzes query type, complexity, and requirements
2. **Provider Selection**: Evaluates available AI providers based on:
   - **Query Type**: Code analysis, planning, critique, synthesis
   - **Complexity**: Simple vs. complex multi-step tasks
   - **Cost**: Token usage and provider pricing
   - **Performance**: Response time and quality requirements
   - **Specialization**: Provider strengths for specific tasks

3. **Smart Routing**: Routes to the optimal provider using advanced algorithms
4. **Fallback Handling**: Automatically switches providers if needed

### **🎯 Routing Intelligence**

| Query Type | Optimal Provider | Reasoning |
|------------|------------------|-----------|
| **Code Analysis** | Claude-3-Sonnet | Superior code understanding |
| **Planning Tasks** | GPT-4o | Excellent strategic thinking |
| **Creative Writing** | Claude-3-Opus | Best creative capabilities |
| **Quick Answers** | GPT-4o-mini | Fast, cost-effective |
| **Complex Reasoning** | Claude-3-Opus | Deep analytical thinking |
| **TreeQuest Tasks** | Multi-Agent | Orchestrated AI collaboration |

### **💰 Cost Optimization**

- **Real-time Cost Tracking**: Monitor token usage and costs
- **Provider Selection**: Choose cost-effective providers for simple tasks
- **Quality vs. Cost**: Balance between performance and expense
- **Usage Analytics**: Track spending patterns and optimize usage

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.8+
- Git
- OpenAI API key (or other AI provider keys)

### **Installation**

```bash
# Clone the repository
git clone https://github.com/Das-rebel/monk-cli.git
cd monk-cli

# Install dependencies
pip install -r requirements.txt

# Set up your API keys
export OPENAI_API_KEY="your-api-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"
export GOOGLE_API_KEY="your-google-key-here"

# Run the enhanced CLI
python3 monk.py --treequest /agents
```

### **Quick Commands**

```bash
# Show all available AI agents
monk --treequest /agents

# Create execution plan (automatically routed to best planning provider)
monk --treequest /plan objective="Improve project security"

# Run comprehensive analysis (intelligently routed for optimal performance)
monk --treequest /deep-analyze path=src/ depth=comprehensive

# Interactive mode with intelligent routing
monk --treequest --chat
```

### **Intelligent Routing Examples**

```bash
# Simple query - automatically routed to cost-effective provider
monk "What is Python?"

# Complex analysis - routed to high-performance provider
monk "Analyze this codebase for security vulnerabilities and suggest improvements"

# Planning task - routed to strategic thinking provider
monk "Create a project roadmap for implementing microservices architecture"

# Code review - routed to code-specialized provider
monk "Review this Python function for best practices and optimization"
```

## 🤖 **AI Agents**

### **Planner Agent**
- **Purpose**: Strategic planning and roadmap creation
- **Capabilities**: Long-term planning, resource allocation, timeline management
- **Use Cases**: Project planning, architecture design, strategic decision making

### **Analyzer Agent**
- **Purpose**: Data analysis and insight generation
- **Capabilities**: Code analysis, performance profiling, dependency analysis
- **Use Cases**: Code review, performance optimization, security auditing

### **Critic Agent**
- **Purpose**: Quality assessment and improvement suggestions
- **Capabilities**: Code quality evaluation, best practices review, risk assessment
- **Use Cases**: Code review, quality assurance, compliance checking

### **Synthesizer Agent**
- **Purpose**: Combining and summarizing insights
- **Capabilities**: Cross-tool correlation, insight synthesis, recommendation generation
- **Use Cases**: Project overview, decision support, stakeholder communication

### **Executor Agent**
- **Purpose**: Implementation and execution planning
- **Capabilities**: Action planning, implementation strategies, execution tracking
- **Use Cases**: Task execution, automation planning, workflow optimization

## 🔧 **Configuration**

### **Model Registry Configuration**

The system supports multiple AI providers:

```yaml
# config/treequest.yaml
models:
  gpt-4o:
    provider: openai
    capabilities: [planner, analyzer, critic, synthesizer, executor, simulator]
    quality_score: 0.95
    cost_per_1k_tokens: 0.005
  
  claude-3-opus:
    provider: anthropic
    capabilities: [planner, analyzer, critic, synthesizer, executor, simulator]
    quality_score: 0.98
    cost_per_1k_tokens: 0.015
```

### **Environment Variables**

```bash
# Required
export OPENAI_API_KEY="your-openai-key"

# Optional
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export MISTRAL_API_KEY="your-mistral-key"
```

## 📚 **Usage Examples**

### **Project Planning**

```bash
monk --treequest /plan objective="Implement microservices architecture"
```

**Output:**
```
📋 **Execution Plan Generated**

**Objective**: Implement microservices architecture
**Timeline**: 6-8 weeks

**AI Agent Metrics**
  • Confidence: 0.85
  • Cost: $0.0750
  • Execution time: 0.42s
  • Agent used: planner
```

### **Code Analysis**

```bash
monk --treequest /deep-analyze path=src/ depth=comprehensive
```

**Output:**
```
🔬 **Deep Analysis Results**

**Target**: src/
**Depth**: comprehensive
**Agents Used**: all

**AI Performance**
  • Total iterations: 32
  • Max depth explored: 3
  • Final cost: $0.0703
  • Execution time: 0.36s
  • Agent used: synthesizer
```

### **Code Quality Critique**

```bash
monk --treequest /critique path=src/ focus="security and performance"
```

## 🏗️ **Project Structure**

```
monk-cli/
├── src/
│   ├── ai/
│   │   ├── treequest_engine.py      # TreeQuest AI orchestration
│   │   └── model_registry.py        # AI model management
│   ├── core/
│   │   ├── intelligent_router.py    # 🧠 Intelligent Query Routing
│   │   ├── slash_command_processor.py # Enhanced slash commands
│   │   ├── conversation_manager.py  # Conversation management
│   │   └── project_context_loader.py # Project context
│   ├── analyzers/
│   │   ├── analyzer_coordinator.py  # Analysis orchestration
│   │   ├── github_analyzer.py       # GitHub analysis
│   │   ├── docker_optimizer.py      # Docker optimization
│   │   └── npm_manager.py           # NPM package analysis
│   └── ui/
│       └── rich_interface.py        # Rich terminal interface
├── config/
│   └── treequest.yaml               # TreeQuest configuration
├── monk.py                          # Main CLI entry point
├── treequest_demo.py                # Demo script
├── requirements.txt                  # Python dependencies
├── setup_github.sh                  # GitHub setup automation
└── docs/                            # Documentation
```

## 🚀 **Advanced Features**

### **🧠 Intelligent Query Routing System**

Monk CLI's routing system automatically determines the best AI provider for each query:

- **Automatic Provider Selection**: Routes queries to optimal AI providers
- **Cost-Aware Routing**: Balances quality and cost for each task
- **Performance Optimization**: Selects providers based on speed requirements
- **Specialization Matching**: Routes to providers best suited for specific tasks
- **Fallback Handling**: Automatically switches providers if needed

### **🤖 Multi-Agent AI Orchestration**

The TreeQuest engine coordinates multiple AI agents:

- **Planner Agent**: Creates strategic execution plans
- **Analyzer Agent**: Performs deep code and project analysis
- **Critic Agent**: Provides quality assurance and feedback
- **Synthesizer Agent**: Combines insights from multiple sources
- **Executor Agent**: Implements suggested changes and optimizations

### **Cost Optimization**

- Real-time cost tracking per query
- Intelligent model selection based on cost/quality trade-offs
- Automatic fallback to cost-effective alternatives

### **Performance Monitoring**

- Execution time tracking
- Agent usage statistics
- Model performance metrics
- Cost analysis and optimization

## 🧪 **Testing**

### **Test Intelligent Query Routing**

```bash
# Test simple query routing
monk "What is the capital of France?"

# Test complex analysis routing
monk "Analyze this project structure and suggest improvements"

# Test planning task routing
monk "Create a development timeline for this feature"

# Test code review routing
monk "Review this function for best practices"
```

### **Test AI Agents**

```bash
# Test AI agents
monk --treequest /agents

# Test planning
monk --treequest /plan objective="Test planning"

# Test analysis
monk --treequest /deep-analyze path=. depth=deep

# Test interactive mode
monk --treequest --chat
```

## 📊 **Performance Metrics**

### **Typical Performance**

- **Planning Tasks**: 0.35-0.40s execution, $0.0600 cost
- **Deep Analysis**: 0.35-0.40s execution, $0.0703 cost
- **Complex Queries**: 1.8-2.2s execution, $0.000059 cost
- **Agent Response**: <100ms for simple queries

### **Scalability**

- Supports up to 10+ AI providers simultaneously
- Handles complex multi-agent workflows
- Efficient memory and resource management
- Async processing for concurrent operations

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Setup**

```bash
# Clone and setup
git clone https://github.com/Das-rebel/monk-cli.git
cd monk-cli

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **TreeQuest Engine**: Advanced AI orchestration system
- **OpenAI**: GPT-4 and GPT-4o models
- **Anthropic**: Claude models
- **Google**: Gemini models
- **Mistral AI**: Mistral models

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/Das-rebel/monk-cli/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Das-rebel/monk-cli/discussions)
- **Documentation**: [Wiki](https://github.com/Das-rebel/monk-cli/wiki)

## 🚀 **Roadmap**

- [ ] **Phase 1**: Core TreeQuest integration ✅
- [ ] **Phase 2**: Enhanced agent capabilities ✅
- [ ] **Phase 3**: Advanced routing and optimization ✅
- [ ] **Phase 4**: Multi-language support
- [ ] **Phase 5**: Cloud deployment and scaling
- [ ] **Phase 6**: Enterprise features and integrations

---

**Made with ❤️ by the Smart AI Team**

*Transform your CLI experience with the power of AI agents!*
