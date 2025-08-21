"""
MONK CLI Phase 2 Setup and Deployment Script
Automated setup for Phase 2 features: Community Intelligence and Cross-Attention Memory
"""
import os
import sys
import asyncio
import subprocess
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import click
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import config
from src.core.database import startup_database, shutdown_database, db_manager
from src.community.intelligence_system import community_intelligence
from src.memory.cross_attention_memory import cross_attention_memory

logger = logging.getLogger(__name__)


class Phase2EnvironmentSetup:
    """Phase 2 specific environment setup and deployment"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.deploy_dir = self.project_root / "deploy"
        
    def setup_phase2_dependencies(self):
        """Setup Phase 2 specific dependencies"""
        print("🔧 Setting up Phase 2 dependencies...")
        
        # Determine pip path
        venv_dir = self.project_root / "venv"
        if os.name == 'nt':  # Windows
            pip_path = venv_dir / "Scripts" / "pip.exe"
        else:  # Unix/Linux/macOS
            pip_path = venv_dir / "bin" / "pip"
        
        # Phase 2 specific requirements
        phase2_requirements = [
            # Advanced ML for cross-attention
            "torch>=2.1.0",
            "transformers>=4.35.0",
            "accelerate>=0.24.1",
            "tokenizers>=0.15.0",
            
            # Web scraping for research monitoring
            "beautifulsoup4>=4.12.2",
            "feedparser>=6.0.10",
            "lxml>=4.9.3",
            
            # Additional ML utilities
            "datasets>=2.14.0",
            "evaluate>=0.4.1",
            
            # Performance monitoring
            "psutil>=5.9.6",
            "memory-profiler>=0.61.0",
            
            # Advanced testing
            "pytest-benchmark>=4.0.0",
            "pytest-mock>=3.12.0"
        ]
        
        print("Installing Phase 2 ML and AI dependencies...")
        subprocess.run([str(pip_path), "install"] + phase2_requirements, check=True)
        
        print("✅ Phase 2 dependencies installed")
    
    def setup_phase2_environment_variables(self):
        """Setup Phase 2 specific environment variables"""
        print("📝 Setting up Phase 2 environment variables...")
        
        env_file_path = self.project_root / ".env"
        
        # Phase 2 specific environment variables
        phase2_env_vars = {
            # Community Intelligence Configuration
            "CI_RESEARCH_MONITORING_ENABLED": "true",
            "CI_ARXIV_CHECK_FREQUENCY_HOURS": "24",
            "CI_GITHUB_CHECK_FREQUENCY_HOURS": "12",
            "CI_SIGNIFICANCE_THRESHOLD": "0.4",
            "CI_MAX_FINDINGS_PER_CYCLE": "100",
            "CI_ENHANCEMENT_AUTO_GENERATION": "true",
            
            # Cross-Attention Memory Configuration
            "CAM_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
            "CAM_HIDDEN_SIZE": "384",
            "CAM_ATTENTION_HEADS": "12",
            "CAM_NUM_LAYERS": "4",
            "CAM_MAX_SEQ_LENGTH": "512",
            "CAM_CACHE_SIZE": "1000",
            "CAM_DEVICE": "auto",  # auto-detect GPU/CPU
            
            # Performance Targets
            "PHASE2_TARGET_FINDINGS_PER_HOUR": "50",
            "PHASE2_TARGET_MEMORY_RETRIEVAL_MS": "100",
            "PHASE2_TARGET_ENHANCEMENT_GENERATION_S": "5",
            
            # Research Source URLs (can be customized)
            "ARXIV_API_URL": "http://export.arxiv.org/api/query",
            "GITHUB_API_URL": "https://api.github.com",
            "RESEARCH_SOURCES_CONFIG": "research_sources.json"
        }
        
        # Read existing .env file
        existing_vars = {}
        if env_file_path.exists():
            with open(env_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        existing_vars[key] = value
        
        # Merge with Phase 2 variables
        all_vars = {**existing_vars, **phase2_env_vars}
        
        # Write updated .env file
        with open(env_file_path, 'w') as f:
            f.write("# MONK CLI Environment Configuration (Updated with Phase 2)\n")
            f.write(f"# Updated on {datetime.now().isoformat()}\n\n")
            
            # Group variables by category
            categories = {
                "Environment": ["ENVIRONMENT", "DEBUG"],
                "API Configuration": ["API_HOST", "API_PORT", "SECRET_KEY"],
                "Database": ["POSTGRES_", "REDIS_"],
                "AI Models": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PINECONE_"],
                "Phase 1 Agent": ["MAX_CONCURRENT_AGENTS", "AGENT_TIMEOUT_SECONDS"],
                "Phase 1 Memory": ["MAX_MEMORIES_PER_USER", "MEMORY_RETENTION_DAYS"],
                "Phase 2 Community Intelligence": ["CI_"],
                "Phase 2 Cross-Attention Memory": ["CAM_"],
                "Phase 2 Performance": ["PHASE2_"],
                "Research Sources": ["ARXIV_", "GITHUB_", "RESEARCH_"]
            }
            
            for category, prefixes in categories.items():
                category_vars = {}
                for key, value in all_vars.items():
                    if any(key.startswith(prefix) for prefix in prefixes):
                        category_vars[key] = value
                
                if category_vars:
                    f.write(f"# {category}\n")
                    for key, value in sorted(category_vars.items()):
                        f.write(f"{key}={value}\n")
                    f.write("\n")
            
            # Write remaining variables
            written_vars = set()
            for category_vars in categories.values():
                for prefix in category_vars:
                    written_vars.update([k for k in all_vars.keys() if k.startswith(prefix)])
            
            remaining_vars = {k: v for k, v in all_vars.items() if k not in written_vars}
            if remaining_vars:
                f.write("# Other Configuration\n")
                for key, value in sorted(remaining_vars.items()):
                    f.write(f"{key}={value}\n")
        
        print(f"✅ Environment variables updated: {env_file_path}")
    
    def setup_research_sources_config(self):
        """Setup research sources configuration file"""
        print("🔍 Setting up research sources configuration...")
        
        config_path = self.project_root / "research_sources.json"
        
        research_sources_config = {
            "version": "2.0",
            "last_updated": datetime.now().isoformat(),
            "sources": {
                "arxiv": {
                    "enabled": True,
                    "api_url": "http://export.arxiv.org/api/query",
                    "focus_areas": [
                        "multi-agent", 
                        "memory_systems", 
                        "tool_orchestration", 
                        "reasoning", 
                        "collaboration",
                        "attention_mechanisms",
                        "knowledge_graphs",
                        "reinforcement_learning"
                    ],
                    "significance_threshold": 0.4,
                    "max_results_per_query": 100,
                    "update_frequency_hours": 24,
                    "search_categories": ["cs.AI", "cs.LG", "cs.CL", "cs.MA"],
                    "keywords": [
                        "artificial intelligence",
                        "machine learning",
                        "deep learning",
                        "neural network",
                        "transformer",
                        "attention",
                        "multi-agent",
                        "memory",
                        "reasoning",
                        "planning",
                        "tool use",
                        "language model"
                    ]
                },
                "github": {
                    "enabled": True,
                    "api_url": "https://api.github.com",
                    "trending_repositories": True,
                    "specific_repositories": [
                        "langchain-ai/langchain",
                        "microsoft/autogen",
                        "openai/openai-python",
                        "anthropics/anthropic-sdk-python",
                        "hwchase17/langchain",
                        "microsoft/semantic-kernel",
                        "deepmind/lab2d",
                        "facebookresearch/llama"
                    ],
                    "topics": [
                        "ai-agent",
                        "llm",
                        "machine-learning",
                        "artificial-intelligence",
                        "neural-network",
                        "deep-learning",
                        "multi-agent",
                        "memory-systems",
                        "reasoning",
                        "planning"
                    ],
                    "languages": ["Python", "JavaScript", "TypeScript", "Jupyter Notebook"],
                    "minimum_stars": 100,
                    "update_frequency_hours": 12
                },
                "blogs": {
                    "enabled": True,
                    "sources": [
                        {
                            "name": "OpenAI Blog",
                            "url": "https://openai.com/blog",
                            "rss_feed": "https://openai.com/blog/rss.xml",
                            "significance_threshold": 0.7
                        },
                        {
                            "name": "Anthropic Blog", 
                            "url": "https://anthropic.com/blog",
                            "rss_feed": "https://anthropic.com/blog/rss",
                            "significance_threshold": 0.7
                        },
                        {
                            "name": "DeepMind Blog",
                            "url": "https://deepmind.google/discover/blog",
                            "rss_feed": "https://deepmind.google/blog/rss.xml",
                            "significance_threshold": 0.6
                        },
                        {
                            "name": "Meta AI Blog",
                            "url": "https://ai.meta.com/blog",
                            "rss_feed": "https://ai.meta.com/blog/rss",
                            "significance_threshold": 0.6
                        }
                    ],
                    "update_frequency_hours": 24
                },
                "communities": {
                    "enabled": False,  # Phase 3 feature
                    "sources": [
                        {
                            "name": "Reddit r/MachineLearning",
                            "url": "https://reddit.com/r/MachineLearning",
                            "api_endpoint": "https://reddit.com/r/MachineLearning.json"
                        },
                        {
                            "name": "Hacker News AI",
                            "url": "https://news.ycombinator.com",
                            "search_terms": ["AI", "machine learning", "neural network"]
                        }
                    ]
                }
            },
            "enhancement_criteria": {
                "minimum_significance_score": 0.6,
                "minimum_implementation_potential": 0.5,
                "auto_generate_plans": True,
                "max_enhancements_per_day": 5,
                "priority_focus_areas": [
                    "multi-agent",
                    "memory_systems", 
                    "attention_mechanisms",
                    "reasoning"
                ]
            },
            "monitoring_schedule": {
                "daily_update_time": "02:00",  # 2 AM UTC
                "weekly_deep_scan": "Sunday",
                "monthly_trends_analysis": "1st",
                "emergency_breakthrough_check_hours": 4
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(research_sources_config, f, indent=2)
        
        print(f"✅ Research sources configuration created: {config_path}")
    
    async def setup_phase2_database_tables(self):
        """Setup Phase 2 database tables"""
        print("🗄️  Setting up Phase 2 database tables...")
        
        try:
            # Initialize database connections
            await startup_database()
            
            # Create Phase 2 tables
            await db_manager.create_tables()
            
            # Initialize community intelligence tracking
            async with db_manager.get_session() as session:
                from src.core.models import CommunityIntelligence
                
                # Check if CI record exists
                ci_result = await session.execute("SELECT * FROM community_intelligence LIMIT 1")
                if not ci_result.fetchone():
                    # Create initial CI record
                    ci = CommunityIntelligence(
                        system_status="active",
                        monitoring_config={
                            "sources_enabled": ["arxiv", "github", "blogs"],
                            "update_frequency_hours": 24,
                            "significance_threshold": 0.4
                        },
                        enhancement_thresholds={
                            "min_significance": 0.6,
                            "min_implementation_potential": 0.5,
                            "auto_generate": True
                        }
                    )
                    session.add(ci)
                    await session.commit()
            
            print("✅ Phase 2 database tables initialized")
            
        except Exception as e:
            print(f"❌ Database setup failed: {e}")
            raise
        finally:
            await shutdown_database()
    
    async def initialize_phase2_systems(self):
        """Initialize Phase 2 systems"""
        print("🚀 Initializing Phase 2 systems...")
        
        try:
            # Initialize community intelligence system
            print("  🔍 Initializing community intelligence...")
            # Community intelligence will be initialized when API server starts
            
            # Initialize cross-attention memory system
            print("  🧠 Initializing cross-attention memory...")
            await cross_attention_memory.initialize()
            
            print("✅ Phase 2 systems initialized")
            
        except Exception as e:
            print(f"❌ Phase 2 system initialization failed: {e}")
            raise
    
    def create_phase2_startup_scripts(self):
        """Create Phase 2 startup scripts"""
        print("📜 Creating Phase 2 startup scripts...")
        
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Phase 2 development startup script
        phase2_dev_script = """#!/bin/bash
# MONK CLI Phase 2 Development Startup Script

echo "🧘 Starting MONK CLI Phase 2 Development Environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check Phase 2 dependencies
echo "🔧 Checking Phase 2 dependencies..."
python -c "import torch, transformers, beautifulsoup4" || {
    echo "❌ Phase 2 dependencies missing. Installing..."
    pip install torch transformers accelerate beautifulsoup4 feedparser lxml
}

# Check if Docker containers are running
if ! docker-compose ps | grep -q "Up"; then
    echo "🐳 Starting database containers..."
    docker-compose up -d
fi

# Wait for databases to be ready
echo "⏳ Waiting for databases..."
sleep 5

# Start community intelligence monitoring
echo "🔍 Starting community intelligence monitoring..."

# Start the API server with Phase 2 features
echo "🚀 Starting MONK CLI API server with Phase 2 features..."
export MONK_PHASE=2
export CI_RESEARCH_MONITORING_ENABLED=true
export CAM_ENABLED=true

python -m src.api.server

echo "✅ MONK CLI Phase 2 is now running!"
echo "📊 API: http://localhost:8080"
echo "📖 API Docs: http://localhost:8080/docs"
echo "🔍 Community Intelligence: http://localhost:8080/api/v1/community/status"
"""
        
        phase2_dev_path = scripts_dir / "start_phase2_dev.sh"
        with open(phase2_dev_path, 'w') as f:
            f.write(phase2_dev_script)
        os.chmod(phase2_dev_path, 0o755)
        
        # Windows version
        phase2_windows_script = """@echo off
REM MONK CLI Phase 2 Development Startup Script for Windows

echo 🧘 Starting MONK CLI Phase 2 Development Environment...

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found. Run setup first.
    exit /b 1
)

REM Activate virtual environment
call venv\\Scripts\\activate.bat

REM Check Phase 2 dependencies
echo 🔧 Checking Phase 2 dependencies...
python -c "import torch, transformers, beautifulsoup4" >nul 2>&1 || (
    echo ❌ Phase 2 dependencies missing. Installing...
    pip install torch transformers accelerate beautifulsoup4 feedparser lxml
)

REM Start Docker containers
echo 🐳 Starting database containers...
docker-compose up -d

REM Wait for databases
echo ⏳ Waiting for databases...
timeout /t 5 > nul

REM Set Phase 2 environment variables
set MONK_PHASE=2
set CI_RESEARCH_MONITORING_ENABLED=true
set CAM_ENABLED=true

REM Start the API server
echo 🚀 Starting MONK CLI API server with Phase 2 features...
python -m src.api.server

echo ✅ MONK CLI Phase 2 is now running!
echo 📊 API: http://localhost:8080
echo 📖 API Docs: http://localhost:8080/docs
echo 🔍 Community Intelligence: http://localhost:8080/api/v1/community/status
"""
        
        phase2_windows_path = scripts_dir / "start_phase2_dev.bat"
        with open(phase2_windows_path, 'w') as f:
            f.write(phase2_windows_script)
        
        # Phase 2 testing script
        phase2_test_script = """#!/bin/bash
# MONK CLI Phase 2 Test Script

echo "🧪 Running MONK CLI Phase 2 Tests..."

# Activate virtual environment
source venv/bin/activate

# Run Phase 1 tests first
echo "Running Phase 1 tests..."
python -m pytest tests/test_phase1_comprehensive.py -v

# Run Phase 2 specific tests
echo "Running Phase 2 tests..."
python -m pytest tests/test_phase2_comprehensive.py -v

# Run Phase 2 performance benchmarks
echo "Running Phase 2 performance benchmarks..."
python tests/test_phase2_comprehensive.py benchmark

# Run integration tests
echo "Running Phase 1 + Phase 2 integration tests..."
python -m pytest tests/ -k "test_integration" -v

echo "✅ All Phase 2 tests completed!"
"""
        
        phase2_test_path = scripts_dir / "run_phase2_tests.sh"
        with open(phase2_test_path, 'w') as f:
            f.write(phase2_test_script)
        os.chmod(phase2_test_path, 0o755)
        
        print(f"✅ Phase 2 startup scripts created in {scripts_dir}/")
    
    def create_phase2_docker_compose(self):
        """Create Phase 2 Docker Compose configuration"""
        print("🐳 Creating Phase 2 Docker Compose configuration...")
        
        # Extend existing docker-compose with Phase 2 services
        docker_compose_phase2 = {
            "version": "3.8",
            "services": {
                # Existing services (postgres, redis) from Phase 1
                "postgres": {
                    "image": "postgres:15.4-alpine",
                    "environment": {
                        "POSTGRES_DB": config.database.postgres_db,
                        "POSTGRES_USER": config.database.postgres_user,
                        "POSTGRES_PASSWORD": config.database.postgres_password or "development-password"
                    },
                    "ports": ["5432:5432"],
                    "volumes": ["postgres_data:/var/lib/postgresql/data"],
                    "healthcheck": {
                        "test": ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"],
                        "interval": "10s",
                        "timeout": "5s", 
                        "retries": 5
                    }
                },
                "redis": {
                    "image": "redis:7.0-alpine",
                    "ports": ["6379:6379"],
                    "volumes": ["redis_data:/data"],
                    "command": "redis-server --appendonly yes",
                    "healthcheck": {
                        "test": ["CMD", "redis-cli", "ping"],
                        "interval": "10s",
                        "timeout": "3s",
                        "retries": 5
                    }
                },
                
                # Phase 2 specific services
                "monk-api-phase2": {
                    "build": {
                        "context": ".",
                        "dockerfile": "Dockerfile.phase2"
                    },
                    "environment": {
                        "MONK_PHASE": "2",
                        "CI_RESEARCH_MONITORING_ENABLED": "true",
                        "CAM_ENABLED": "true",
                        "POSTGRES_HOST": "postgres",
                        "REDIS_HOST": "redis"
                    },
                    "ports": ["8080:8080"],
                    "depends_on": {
                        "postgres": {"condition": "service_healthy"},
                        "redis": {"condition": "service_healthy"}
                    },
                    "volumes": [
                        "./src:/app/src",
                        "./research_sources.json:/app/research_sources.json",
                        "model_cache:/app/.cache"
                    ],
                    "restart": "unless-stopped"
                },
                
                # Optional: Dedicated research monitoring service
                "research-monitor": {
                    "build": {
                        "context": ".",
                        "dockerfile": "Dockerfile.research"
                    },
                    "environment": {
                        "CI_RESEARCH_MONITORING_ENABLED": "true",
                        "POSTGRES_HOST": "postgres",
                        "REDIS_HOST": "redis"
                    },
                    "depends_on": ["postgres", "redis"],
                    "volumes": ["./research_sources.json:/app/research_sources.json"],
                    "restart": "unless-stopped"
                },
                
                # Optional: Memory optimization service
                "memory-optimizer": {
                    "build": {
                        "context": ".",
                        "dockerfile": "Dockerfile.memory"
                    },
                    "environment": {
                        "CAM_ENABLED": "true",
                        "POSTGRES_HOST": "postgres",
                        "REDIS_HOST": "redis"
                    },
                    "depends_on": ["postgres", "redis"],
                    "volumes": ["model_cache:/app/.cache"],
                    "restart": "unless-stopped"
                }
            },
            
            "volumes": {
                "postgres_data": {},
                "redis_data": {},
                "model_cache": {}  # For ML model caching
            },
            
            "networks": {
                "monk-network": {
                    "driver": "bridge"
                }
            }
        }
        
        # Write docker-compose-phase2.yml
        compose_path = self.project_root / "docker-compose-phase2.yml"
        with open(compose_path, 'w') as f:
            yaml.dump(docker_compose_phase2, f, default_flow_style=False)
        
        print(f"✅ Phase 2 Docker Compose configuration created: {compose_path}")
    
    async def run_phase2_health_check(self):
        """Run Phase 2 system health check"""
        print("🏥 Running Phase 2 health check...")
        
        health_status = {
            "phase1_systems": {
                "database": False,
                "redis": False,
                "agents": False,
                "memory": False
            },
            "phase2_systems": {
                "community_intelligence": False,
                "cross_attention_memory": False,
                "research_monitoring": False,
                "enhancement_generation": False
            }
        }
        
        try:
            # Check Phase 1 systems
            await startup_database()
            health = await db_manager.health_check()
            
            health_status["phase1_systems"]["database"] = health.get("postgres", False)
            health_status["phase1_systems"]["redis"] = health.get("redis", False)
            
            # Check community intelligence system
            try:
                ci_status = await community_intelligence.get_system_status()
                health_status["phase2_systems"]["community_intelligence"] = ci_status["status"] == "active"
            except Exception:
                health_status["phase2_systems"]["community_intelligence"] = False
            
            # Check cross-attention memory system
            try:
                await cross_attention_memory.initialize()
                health_status["phase2_systems"]["cross_attention_memory"] = True
            except Exception:
                health_status["phase2_systems"]["cross_attention_memory"] = False
                
        except Exception as e:
            print(f"Health check failed: {e}")
        finally:
            await shutdown_database()
        
        # Report health status
        print("\n📊 Phase 2 Health Check Results:")
        
        print("  Phase 1 Systems:")
        for component, status in health_status["phase1_systems"].items():
            status_icon = "✅" if status else "❌"
            print(f"    {status_icon} {component.replace('_', ' ').title()}")
        
        print("  Phase 2 Systems:")
        for component, status in health_status["phase2_systems"].items():
            status_icon = "✅" if status else "❌"
            print(f"    {status_icon} {component.replace('_', ' ').title()}")
        
        # Overall health
        phase1_healthy = all(health_status["phase1_systems"].values())
        phase2_healthy = all(health_status["phase2_systems"].values())
        overall_healthy = phase1_healthy and phase2_healthy
        
        print(f"\n🎯 Overall Phase 2 Health: {'✅ Healthy' if overall_healthy else '⚠️ Issues Detected'}")
        
        if not overall_healthy:
            print("\n🔧 Recommendations:")
            if not phase1_healthy:
                print("  - Run Phase 1 setup first: python deploy/setup_environment.py setup")
            if not phase2_healthy:
                print("  - Check Phase 2 dependencies and configuration")
                print("  - Verify API keys are set in .env file")
                print("  - Ensure sufficient system resources for ML models")
        
        return overall_healthy


# CLI Commands
@click.group()
def cli():
    """MONK CLI Phase 2 Setup and Deployment Tools"""
    pass


@cli.command()
@click.option("--environment", "-e", default="development",
              type=click.Choice(["development", "testing", "production"]),
              help="Environment to setup")
def setup_phase2(environment):
    """Setup complete MONK CLI Phase 2 environment"""
    click.echo(f"🧘 Setting up MONK CLI Phase 2 for {environment} environment...")
    
    setup_manager = Phase2EnvironmentSetup(environment)
    
    try:
        # Setup Phase 2 dependencies
        setup_manager.setup_phase2_dependencies()
        click.echo("✅ Phase 2 dependencies installed")
        
        # Setup environment variables
        setup_manager.setup_phase2_environment_variables()
        click.echo("✅ Phase 2 environment variables configured")
        
        # Setup research sources
        setup_manager.setup_research_sources_config()
        click.echo("✅ Research sources configured")
        
        # Setup database tables
        async def init_db():
            await setup_manager.setup_phase2_database_tables()
        asyncio.run(init_db())
        click.echo("✅ Phase 2 database tables created")
        
        # Initialize Phase 2 systems
        async def init_systems():
            await setup_manager.initialize_phase2_systems()
        asyncio.run(init_systems())
        click.echo("✅ Phase 2 systems initialized")
        
        # Create startup scripts
        setup_manager.create_phase2_startup_scripts()
        click.echo("✅ Phase 2 startup scripts created")
        
        # Create Docker configuration
        setup_manager.create_phase2_docker_compose()
        click.echo("✅ Phase 2 Docker configuration created")
        
        click.echo("\n🎉 MONK CLI Phase 2 setup complete!")
        click.echo("\n📝 Next steps:")
        click.echo("1. Update API keys in .env file (OpenAI, Anthropic, etc.)")
        click.echo("2. Run 'python deploy/setup_phase2.py health-check' to verify setup")
        click.echo("3. Use './scripts/start_phase2_dev.sh' to start with Phase 2 features")
        click.echo("4. Run Phase 2 tests with './scripts/run_phase2_tests.sh'")
        click.echo("5. Access community intelligence at http://localhost:8080/api/v1/community/")
        
    except Exception as e:
        click.echo(f"❌ Phase 2 setup failed: {e}")
        sys.exit(1)


@cli.command()
def health_check():
    """Run Phase 2 system health check"""
    async def check():
        setup_manager = Phase2EnvironmentSetup()
        healthy = await setup_manager.run_phase2_health_check()
        return healthy
    
    healthy = asyncio.run(check())
    sys.exit(0 if healthy else 1)


@cli.command()
@click.option("--config-file", default="research_sources.json",
              help="Research sources configuration file")
def update_research_sources(config_file):
    """Update research sources configuration"""
    click.echo("🔍 Updating research sources configuration...")
    
    setup_manager = Phase2EnvironmentSetup()
    setup_manager.setup_research_sources_config()
    
    click.echo(f"✅ Research sources configuration updated: {config_file}")


@cli.command()
@click.option("--component", type=click.Choice(["community", "memory", "all"]),
              default="all", help="Which Phase 2 component to test")
def test_phase2(component):
    """Run Phase 2 component tests"""
    click.echo(f"🧪 Running Phase 2 {component} tests...")
    
    if component in ["community", "all"]:
        # Test community intelligence
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_phase2_comprehensive.py::TestCommunityIntelligenceSystem", 
            "-v"
        ])
        if result.returncode != 0:
            click.echo("❌ Community intelligence tests failed")
            sys.exit(1)
    
    if component in ["memory", "all"]:
        # Test cross-attention memory
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/test_phase2_comprehensive.py::TestCrossAttentionMemoryRetrieval",
            "-v"
        ])
        if result.returncode != 0:
            click.echo("❌ Cross-attention memory tests failed")
            sys.exit(1)
    
    click.echo("✅ Phase 2 tests passed!")


@cli.command()
def benchmark_phase2():
    """Run Phase 2 performance benchmarks"""
    click.echo("🏃 Running Phase 2 performance benchmarks...")
    
    result = subprocess.run([
        sys.executable, "tests/test_phase2_comprehensive.py", "benchmark"
    ])
    
    if result.returncode == 0:
        click.echo("✅ Phase 2 benchmarks completed!")
    else:
        click.echo("⚠️ Phase 2 benchmarks completed with issues")
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    cli()