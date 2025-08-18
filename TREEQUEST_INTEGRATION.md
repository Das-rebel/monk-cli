# TreeQuest AI Agent Integration

## Overview

The Smart AI Enhanced CLI has been enhanced with **TreeQuest AI Agent Integration**, bringing sophisticated multi-agent AI orchestration to your command-line interface. This integration leverages the TreeQuest engine's Adaptive Branching Monte Carlo Tree Search (AB-MCTS) algorithm to coordinate multiple AI agents for complex tasks.

## 🚀 Key Features

### Multi-Agent Architecture
- **Planner Agent**: Strategic planning and long-term thinking
- **Analyzer Agent**: Data analysis and insight generation
- **Critic Agent**: Quality assessment and constructive feedback
- **Synthesizer Agent**: Integration of multiple perspectives
- **Executor Agent**: Implementation feasibility and execution planning

### Enhanced Slash Commands
- `/agents` - Manage AI agents and their capabilities
- `/plan` - Create execution plans using TreeQuest planner agent
- `/critique` - Get code quality critique using TreeQuest critic agent
- `/synthesize` - Synthesize insights using TreeQuest synthesizer agent
- `/deep-analyze` - Multi-agent deep analysis using TreeQuest
- `/optimize` - Get optimization recommendations using TreeQuest
- `/security-scan` - Comprehensive security analysis

### Intelligent Analysis Coordination
- **Cross-tool correlation**: Find patterns across multiple analyzers
- **AI-powered insights**: Generate strategic recommendations
- **Risk assessment**: Automated project health evaluation
- **Priority actions**: Extract actionable next steps

## 🏗️ Architecture

```
Smart AI CLI v3.0
├── TreeQuest Engine
│   ├── AB-MCTS Algorithm
│   ├── Model Registry
│   ├── Agent Orchestration
│   └── Cost Management
├── Enhanced Slash Commands
│   ├── AI Agent Commands
│   ├── Analysis Commands
│   └── Planning Commands
├── Enhanced Analyzer Coordinator
│   ├── Multi-analyzer Execution
│   ├── Cross-tool Insights
│   └── TreeQuest Integration
└── Model-Specific Agents
    ├── Role-based Task Handling
    ├── Specialized Prompts
    └── Agent-specific Logic
```

## 🚀 Getting Started

### 1. Run the Demo
```bash
python treequest_demo.py
```

### 2. Use Enhanced CLI
```bash
# Interactive mode with TreeQuest
python smart_ai_enhanced_v3.py --treequest

# Direct TreeQuest commands
python smart_ai_enhanced_v3.py /agents
python smart_ai_enhanced_v3.py /plan
python smart_ai_enhanced_v3.py /deep-analyze
```

### 3. Interactive Commands
```
💭 Smart AI> /agents
💭 Smart AI> /plan objective="Improve project security"
💭 Smart AI> /deep-analyze path=. depth=comprehensive
💭 Smart AI> /synthesize topic="Performance optimization"
```

## 🔧 Configuration

### TreeQuest Engine Settings
```python
treequest_config = TreeQuestConfig(
    max_depth=3,              # Maximum search depth
    branching=4,              # Branches per node
    rollout_budget=32,        # Simulation iterations
    cost_cap_usd=0.50,       # Maximum cost per operation
    objective="quality"       # quality, latency, or cost
)
```

### Model Registry
The system automatically manages multiple AI providers:
- **OpenAI**: GPT-4o, GPT-4o-mini
- **Anthropic**: Claude-3-Opus, Claude-3-Sonnet
- **Mistral**: Mistral-Large
- **Google**: Gemini-Pro

## 📊 Usage Examples

### 1. AI Agent Management
```bash
/agents
```
Shows available AI agents, their capabilities, and cost analysis.

### 2. Strategic Planning
```bash
/plan objective="Improve code quality" timeline="2 weeks"
```
Creates an execution plan using the TreeQuest planner agent.

### 3. Code Quality Critique
```bash
/critique path=src/ focus="Security and performance"
```
Gets comprehensive code quality assessment using the critic agent.

### 4. Insight Synthesis
```bash
/synthesize topic="Project optimization" depth=comprehensive
```
Synthesizes insights from multiple analyzers using the synthesizer agent.

### 5. Deep Analysis
```bash
/deep-analyze path=. depth=deep agents=planner,analyzer,critic
```
Runs multi-agent analysis with specified agent roles.

## 🧠 How TreeQuest Works

### 1. Task Decomposition
TreeQuest breaks complex tasks into subtasks and assigns appropriate agents.

### 2. Agent Selection
Based on task type and depth, TreeQuest selects the most suitable agent:
- **Depth 0**: Planner agent (strategic thinking)
- **Depth 1**: Analyzer agent (data analysis)
- **Depth 2**: Critic agent (quality assessment)
- **Depth 3+**: Synthesizer agent (integration)

### 3. Monte Carlo Tree Search
- **Selection**: Choose promising nodes using UCB1 algorithm
- **Expansion**: Create new child nodes
- **Simulation**: Use agent-specific logic to evaluate states
- **Backpropagation**: Update node statistics

### 4. Solution Extraction
TreeQuest extracts the best solution path and provides:
- Confidence scores
- Cost analysis
- Agent usage metrics
- Execution recommendations

## 🔍 Agent Capabilities

### Planner Agent
- **Role**: Strategic planning and long-term thinking
- **Strengths**: High-level strategy, resource allocation, risk mitigation
- **Use Cases**: Project planning, architecture decisions, roadmap creation

### Analyzer Agent
- **Role**: Data analysis and insight generation
- **Strengths**: Pattern recognition, data correlation, metric analysis
- **Use Cases**: Code analysis, performance profiling, dependency analysis

### Critic Agent
- **Role**: Quality assessment and constructive feedback
- **Strengths**: Code review, issue identification, improvement suggestions
- **Use Cases**: Code quality assessment, security review, best practices

### Synthesizer Agent
- **Role**: Integration of multiple perspectives
- **Strengths**: Cross-tool insights, unified recommendations, holistic view
- **Use Cases**: Project health assessment, optimization planning, risk analysis

### Executor Agent
- **Role**: Implementation feasibility and execution planning
- **Strengths**: Action planning, resource estimation, implementation steps
- **Use Cases**: Task breakdown, implementation planning, resource allocation

## 📈 Performance Metrics

### TreeQuest Metrics
- **Max Depth Reached**: How deep the search explored
- **Total Iterations**: Number of MCTS iterations
- **Final Cost**: Total cost in USD
- **Execution Time**: Total processing time
- **Best Node Reward**: Confidence score of best solution
- **Agent Role Used**: Which agent processed the final solution

### Analysis Metrics
- **Overall Score**: Project health score (0-100)
- **Risk Assessment**: Low/Medium/High risk level
- **Confidence Score**: AI confidence in recommendations
- **Cross-tool Insights**: Number of correlated findings
- **Priority Actions**: Number of actionable recommendations

## 🛠️ Development

### Adding New Agents
```python
# 1. Add agent role to ModelRole enum
class ModelRole(Enum):
    NEW_AGENT = "new_agent"

# 2. Add agent handler to TreeQuestEngine
self.agent_handlers["new_agent"] = self._handle_new_agent_task

# 3. Implement agent handler
async def _handle_new_agent_task(self, prompt: str, model_name: str, node: TreeNode) -> float:
    # Agent-specific logic
    return reward_score
```

### Custom Agent Logic
```python
async def _handle_custom_task(self, prompt: str, model_name: str, node: TreeNode) -> float:
    """Custom agent behavior"""
    # Analyze prompt content
    if "custom_keyword" in prompt.lower():
        base_reward = 0.8
    else:
        base_reward = 0.6
    
    # Add depth-based bonus
    depth_bonus = min(0.2, node.depth * 0.05)
    
    return min(1.0, base_reward + depth_bonus)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. TreeQuest Engine Not Initialized
```bash
❌ TreeQuest engine not initialized
```
**Solution**: Ensure the enhanced slash command processor is properly initialized.

#### 2. Model Registry Errors
```bash
❌ No available models for role planner
```
**Solution**: Check API keys and model availability in the model registry.

#### 3. Agent Simulation Failures
```bash
❌ Agent simulation failed
```
**Solution**: Check agent handler implementations and error handling.

### Debug Mode
Enable verbose logging to debug TreeQuest operations:
```bash
python smart_ai_enhanced_v3.py --verbose --treequest
```

## 📚 API Reference

### TreeQuestEngine
```python
class TreeQuestEngine:
    async def solve(task: str, context: Dict[str, Any]) -> Dict[str, Any]
    async def synthesize_insights(analyzer_results: Dict[str, Any]) -> Dict[str, Any]
```

### EnhancedSlashCommandProcessor
```python
class EnhancedSlashCommandProcessor:
    async def initialize()
    async def execute_command(command: SlashCommand) -> Dict[str, Any]
```

### EnhancedAnalyzerCoordinator
```python
class EnhancedAnalyzerCoordinator:
    async def comprehensive_analysis(project_path: Path, options: Dict[str, Any]) -> TreeQuestAnalysisResult
    async def get_analysis_summary(project_path: Path) -> Dict[str, Any]
```

## 🚀 Future Enhancements

### Planned Features
- **Real-time Agent Communication**: Inter-agent messaging and coordination
- **Dynamic Agent Creation**: Runtime agent generation based on task requirements
- **Advanced Cost Optimization**: Smarter model selection based on cost-benefit analysis
- **Agent Learning**: Agents that improve based on feedback and results
- **Multi-project Coordination**: Cross-project analysis and insights

### Integration Opportunities
- **CI/CD Integration**: Automated analysis in build pipelines
- **IDE Plugins**: TreeQuest integration in VS Code, IntelliJ, etc.
- **API Endpoints**: RESTful API for external tool integration
- **Web Dashboard**: Visual TreeQuest exploration and results

## 🤝 Contributing

### Development Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up API keys for AI providers
4. Run tests: `python -m pytest tests/`
5. Run demo: `python treequest_demo.py`

### Code Standards
- Follow existing code style and patterns
- Add comprehensive error handling
- Include logging for debugging
- Write tests for new functionality
- Update documentation for new features

## 📄 License

This project is licensed under the same terms as the main Smart AI Enhanced CLI project.

---

**🎉 Congratulations!** Your Smart AI CLI is now powered by sophisticated TreeQuest AI agents, providing enterprise-grade AI orchestration capabilities in a simple command-line interface.
