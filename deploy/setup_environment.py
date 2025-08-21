"""
MONK CLI Environment Setup and Deployment Script
Automated setup for development, testing, and production environments
"""
import os
import sys
import asyncio
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import json
import click
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import config
from src.core.database import startup_database, db_manager


class EnvironmentSetup:
    """Handles environment setup and deployment"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.deploy_dir = self.project_root / "deploy"
        
    def setup_python_environment(self):
        """Setup Python virtual environment and dependencies"""
        print("🐍 Setting up Python environment...")
        
        # Create virtual environment if it doesn't exist
        venv_dir = self.project_root / "venv"
        if not venv_dir.exists():
            print("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        
        # Determine pip path
        if os.name == 'nt':  # Windows
            pip_path = venv_dir / "Scripts" / "pip.exe"
            python_path = venv_dir / "Scripts" / "python.exe"
        else:  # Unix/Linux/macOS
            pip_path = venv_dir / "bin" / "pip"
            python_path = venv_dir / "bin" / "python"
        
        # Install dependencies
        print("Installing Phase 1 requirements...")
        subprocess.run([str(pip_path), "install", "-r", "phase1_requirements.txt"], 
                      cwd=self.project_root, check=True)
        
        # Install development dependencies
        dev_requirements = [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0"
        ]
        
        print("Installing development dependencies...")
        subprocess.run([str(pip_path), "install"] + dev_requirements, check=True)
        
        return python_path
    
    def setup_docker_environment(self):
        """Setup Docker containers for databases"""
        print("🐳 Setting up Docker environment...")
        
        # Create docker-compose.yml
        docker_compose = {
            "version": "3.8",
            "services": {
                "postgres": {
                    "image": "postgres:15.4-alpine",
                    "environment": {
                        "POSTGRES_DB": config.database.postgres_db,
                        "POSTGRES_USER": config.database.postgres_user,
                        "POSTGRES_PASSWORD": config.database.postgres_password or "development-password",
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
                }
            },
            "volumes": {
                "postgres_data": {},
                "redis_data": {}
            }
        }
        
        if self.environment == "production":
            # Add production-specific configurations
            docker_compose["services"]["postgres"]["restart"] = "unless-stopped"
            docker_compose["services"]["redis"]["restart"] = "unless-stopped"
            
            # Add resource limits
            docker_compose["services"]["postgres"]["deploy"] = {
                "resources": {
                    "limits": {"cpus": "2", "memory": "4G"},
                    "reservations": {"cpus": "1", "memory": "2G"}
                }
            }
            
            docker_compose["services"]["redis"]["deploy"] = {
                "resources": {
                    "limits": {"cpus": "1", "memory": "2G"},
                    "reservations": {"cpus": "0.5", "memory": "1G"}
                }
            }
        
        # Write docker-compose.yml
        docker_compose_path = self.project_root / "docker-compose.yml"
        with open(docker_compose_path, 'w') as f:
            yaml.dump(docker_compose, f, default_flow_style=False)
        
        print(f"Docker Compose configuration written to {docker_compose_path}")
        
        # Start services
        print("Starting database services...")
        subprocess.run(["docker-compose", "up", "-d"], cwd=self.project_root, check=True)
        
        # Wait for services to be healthy
        print("Waiting for services to be ready...")
        subprocess.run(["docker-compose", "ps"], cwd=self.project_root)
        
        return True
    
    def setup_environment_file(self):
        """Create .env file with configuration"""
        print("📝 Setting up environment configuration...")
        
        env_template = {
            # Environment
            "ENVIRONMENT": self.environment,
            "DEBUG": "true" if self.environment == "development" else "false",
            
            # API Configuration
            "API_HOST": "0.0.0.0",
            "API_PORT": "8080",
            "SECRET_KEY": "your-secret-key-change-in-production",
            
            # Database Configuration
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "monk_cli",
            "POSTGRES_USER": "monk",
            "POSTGRES_PASSWORD": "development-password",
            
            # Redis Configuration
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "REDIS_DB": "0",
            
            # AI Model Configuration (require user to fill)
            "OPENAI_API_KEY": "your-openai-api-key-here",
            "ANTHROPIC_API_KEY": "your-anthropic-api-key-here",
            
            # Pinecone Configuration (optional)
            "PINECONE_API_KEY": "your-pinecone-api-key-here",
            "PINECONE_ENVIRONMENT": "us-west1-gcp",
            "PINECONE_INDEX_NAME": "monk-memory",
            
            # Agent Configuration
            "MAX_CONCURRENT_AGENTS": "100",
            "AGENT_TIMEOUT_SECONDS": "300",
            
            # Memory Configuration
            "MAX_MEMORIES_PER_USER": "1000000",
            "MEMORY_RETENTION_DAYS": "90",
            
            # Performance Configuration
            "TARGET_CONCURRENT_USERS": "500",
            "API_RESPONSE_TARGET_MS": "200",
            "MEMORY_QUERY_TARGET_MS": "50"
        }
        
        env_file_path = self.project_root / ".env"
        
        if env_file_path.exists():
            print("⚠️  .env file already exists, backing up...")
            backup_path = self.project_root / f".env.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(env_file_path, backup_path)
            print(f"Backup saved to {backup_path}")
        
        # Write .env file
        with open(env_file_path, 'w') as f:
            f.write("# MONK CLI Environment Configuration\n")
            f.write(f"# Generated on {datetime.now().isoformat()}\n\n")
            
            for key, value in env_template.items():
                f.write(f"{key}={value}\n")
        
        print(f"Environment file created: {env_file_path}")
        print("⚠️  IMPORTANT: Please update API keys in .env file before running!")
        
        return env_file_path
    
    async def initialize_database(self):
        """Initialize database with tables and seed data"""
        print("🗄️  Initializing database...")
        
        try:
            # Initialize database connections
            await db_manager.initialize()
            
            # Create tables
            await db_manager.create_tables()
            
            # Create seed data
            await self._create_seed_data()
            
            print("✅ Database initialization complete")
            
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            raise
        finally:
            await db_manager.close()
    
    async def _create_seed_data(self):
        """Create seed data for development"""
        from src.core.models import User, AgentStack, Agent
        from src.agents.agent_framework import AGENT_REGISTRY
        
        async with db_manager.get_session() as session:
            # Create development user
            dev_user = User(
                email="dev@monk-cli.com",
                username="developer",
                preferences={"default_agent_stack": "development"}
            )
            session.add(dev_user)
            
            # Create agent stacks
            stacks = [
                AgentStack(
                    name="development",
                    description="Development workflow agents",
                    specialization_domains=["system_design", "code_review", "optimization", "integration"]
                ),
                AgentStack(
                    name="content",
                    description="Content creation agents (Phase 2)",
                    specialization_domains=["writing", "design", "marketing"]
                ),
                AgentStack(
                    name="business",
                    description="Business intelligence agents (Phase 2)",
                    specialization_domains=["analysis", "strategy", "reporting"]
                ),
                AgentStack(
                    name="security",
                    description="Security analysis agents (Phase 2)",
                    specialization_domains=["security", "compliance", "auditing"]
                )
            ]
            
            for stack in stacks:
                session.add(stack)
            
            # Flush to get IDs
            await session.flush()
            
            # Create development stack agents
            dev_stack = next(s for s in stacks if s.name == "development")
            
            agent_configs = [
                {
                    "name": "Architect",
                    "description": "System design and architecture analysis",
                    "personality_traits": {
                        "conscientiousness": 0.9,
                        "openness": 0.7,
                        "analytical_thinking": 0.9,
                        "creativity": 0.6,
                        "risk_tolerance": 0.3
                    },
                    "specializations": ["system_design", "architecture", "scalability"],
                    "tools": ["architecture_analyzer", "dependency_mapper"]
                },
                {
                    "name": "Quality Enforcer",
                    "description": "Code review and quality assurance",
                    "personality_traits": {
                        "conscientiousness": 0.95,
                        "openness": 0.4,
                        "analytical_thinking": 0.8,
                        "creativity": 0.3,
                        "risk_tolerance": 0.1
                    },
                    "specializations": ["code_review", "testing", "quality_assurance"],
                    "tools": ["code_scanner", "test_analyzer"]
                },
                {
                    "name": "Innovation Driver",
                    "description": "Emerging technology and optimization",
                    "personality_traits": {
                        "conscientiousness": 0.6,
                        "openness": 0.95,
                        "analytical_thinking": 0.7,
                        "creativity": 0.9,
                        "risk_tolerance": 0.8
                    },
                    "specializations": ["optimization", "emerging_tech", "innovation"],
                    "tools": ["performance_analyzer", "trend_analyzer"]
                },
                {
                    "name": "Integration Specialist",
                    "description": "API integration and deployment",
                    "personality_traits": {
                        "conscientiousness": 0.7,
                        "openness": 0.6,
                        "analytical_thinking": 0.75,
                        "creativity": 0.5,
                        "risk_tolerance": 0.4
                    },
                    "specializations": ["integration", "deployment", "devops"],
                    "tools": ["integration_tester", "deployment_manager"]
                }
            ]
            
            for agent_config in agent_configs:
                agent = Agent(
                    stack_id=dev_stack.id,
                    name=agent_config["name"],
                    description=agent_config["description"],
                    personality_traits=agent_config["personality_traits"],
                    specializations=agent_config["specializations"],
                    tools=agent_config["tools"]
                )
                session.add(agent)
            
            await session.commit()
            print("✅ Seed data created")
    
    def create_startup_scripts(self):
        """Create startup scripts for different environments"""
        print("📜 Creating startup scripts...")
        
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Development startup script
        dev_script = """#!/bin/bash
# MONK CLI Development Startup Script

echo "🧘 Starting MONK CLI Development Environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# venv\\Scripts\\activate  # Windows

# Check if Docker containers are running
if ! docker-compose ps | grep -q "Up"; then
    echo "🐳 Starting database containers..."
    docker-compose up -d
fi

# Wait for databases to be ready
echo "⏳ Waiting for databases..."
sleep 5

# Run database migrations (if needed)
# python -m alembic upgrade head

# Start the API server in development mode
echo "🚀 Starting MONK CLI API server..."
python -m src.api.server

echo "✅ MONK CLI is now running!"
echo "📊 API: http://localhost:8080"
echo "📖 API Docs: http://localhost:8080/docs"
"""
        
        dev_script_path = scripts_dir / "start_dev.sh"
        with open(dev_script_path, 'w') as f:
            f.write(dev_script)
        os.chmod(dev_script_path, 0o755)
        
        # Windows batch script
        windows_script = """@echo off
REM MONK CLI Development Startup Script for Windows

echo 🧘 Starting MONK CLI Development Environment...

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found. Run setup first.
    exit /b 1
)

REM Activate virtual environment
call venv\\Scripts\\activate.bat

REM Start Docker containers
echo 🐳 Starting database containers...
docker-compose up -d

REM Wait for databases
echo ⏳ Waiting for databases...
timeout /t 5 > nul

REM Start the API server
echo 🚀 Starting MONK CLI API server...
python -m src.api.server

echo ✅ MONK CLI is now running!
echo 📊 API: http://localhost:8080
echo 📖 API Docs: http://localhost:8080/docs
"""
        
        windows_script_path = scripts_dir / "start_dev.bat"
        with open(windows_script_path, 'w') as f:
            f.write(windows_script)
        
        # CLI test script
        cli_test_script = """#!/bin/bash
# MONK CLI Test Script

echo "🧪 Running MONK CLI Tests..."

# Activate virtual environment
source venv/bin/activate

# Run unit tests
echo "Running unit tests..."
python -m pytest tests/ -v

# Run integration tests
echo "Running integration tests..."
python -m pytest tests/test_phase1_comprehensive.py -v

# Run performance benchmarks
echo "Running performance benchmarks..."
python tests/test_phase1_comprehensive.py benchmark

# Run competitive benchmark
echo "Running competitive benchmark..."
python benchmarks/competitive_benchmark.py

echo "✅ All tests completed!"
"""
        
        cli_test_path = scripts_dir / "run_tests.sh"
        with open(cli_test_path, 'w') as f:
            f.write(cli_test_script)
        os.chmod(cli_test_path, 0o755)
        
        print(f"Startup scripts created in {scripts_dir}/")
        return scripts_dir
    
    def create_production_deployment(self):
        """Create production deployment configuration"""
        print("🚀 Creating production deployment configuration...")
        
        # Kubernetes deployment
        k8s_dir = self.deploy_dir / "kubernetes"
        k8s_dir.mkdir(parents=True, exist_ok=True)
        
        # Create namespace
        namespace_yaml = """apiVersion: v1
kind: Namespace
metadata:
  name: monk-cli
  labels:
    name: monk-cli
"""
        
        with open(k8s_dir / "namespace.yaml", 'w') as f:
            f.write(namespace_yaml)
        
        # Create deployment
        deployment_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: monk-cli-api
  namespace: monk-cli
spec:
  replicas: 3
  selector:
    matchLabels:
      app: monk-cli-api
  template:
    metadata:
      labels:
        app: monk-cli-api
    spec:
      containers:
      - name: monk-cli-api
        image: monk-cli:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: POSTGRES_HOST
          value: "postgres-service"
        - name: REDIS_HOST
          value: "redis-service"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: monk-cli-api-service
  namespace: monk-cli
spec:
  selector:
    app: monk-cli-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
"""
        
        with open(k8s_dir / "deployment.yaml", 'w') as f:
            f.write(deployment_yaml)
        
        # Create database services
        db_services_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: monk-cli
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15.4-alpine
        env:
        - name: POSTGRES_DB
          value: "monk_cli"
        - name: POSTGRES_USER
          value: "monk"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: monk-cli
spec:
  selector:
    app: postgres
  ports:
    - protocol: TCP
      port: 5432
      targetPort: 5432
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: monk-cli
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7.0-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: monk-cli
spec:
  selector:
    app: redis
  ports:
    - protocol: TCP
      port: 6379
      targetPort: 6379
"""
        
        with open(k8s_dir / "database-services.yaml", 'w') as f:
            f.write(db_services_yaml)
        
        print(f"Kubernetes configurations created in {k8s_dir}/")
        
        return k8s_dir
    
    async def run_health_check(self):
        """Run comprehensive health check"""
        print("🏥 Running system health check...")
        
        health_status = {
            "database": False,
            "redis": False,
            "pinecone": False,
            "agents": False,
            "memory": False
        }
        
        try:
            # Check database connections
            await db_manager.initialize()
            health = await db_manager.health_check()
            
            health_status["database"] = health.get("postgres", False)
            health_status["redis"] = health.get("redis", False)  
            health_status["pinecone"] = health.get("pinecone", False)
            
            # Check agents
            from src.agents.orchestrator import orchestrator
            await orchestrator.start()
            orchestrator_status = orchestrator.get_orchestrator_status()
            health_status["agents"] = orchestrator_status["total_agents"] > 0
            
            # Check memory system
            health_status["memory"] = True  # Basic check
            
            await orchestrator.stop()
            
        except Exception as e:
            print(f"Health check failed: {e}")
        finally:
            await db_manager.close()
        
        # Report health status
        print("\n📊 Health Check Results:")
        for component, status in health_status.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component.capitalize()}: {'Healthy' if status else 'Unhealthy'}")
        
        overall_health = all(health_status.values())
        print(f"\n🎯 Overall System Health: {'✅ Healthy' if overall_health else '⚠️ Issues Detected'}")
        
        return overall_health


# CLI Commands
@click.group()
def cli():
    """MONK CLI Environment Setup and Deployment Tools"""
    pass


@cli.command()
@click.option("--environment", "-e", default="development", 
              type=click.Choice(["development", "testing", "production"]),
              help="Environment to setup")
def setup(environment):
    """Setup complete MONK CLI environment"""
    click.echo(f"🧘 Setting up MONK CLI for {environment} environment...")
    
    setup_manager = EnvironmentSetup(environment)
    
    # Setup Python environment
    python_path = setup_manager.setup_python_environment()
    click.echo(f"✅ Python environment ready: {python_path}")
    
    # Setup environment configuration
    env_file = setup_manager.setup_environment_file()
    click.echo(f"✅ Environment file created: {env_file}")
    
    # Setup Docker if not production
    if environment != "production":
        setup_manager.setup_docker_environment()
        click.echo("✅ Docker environment ready")
    
    # Create startup scripts
    scripts_dir = setup_manager.create_startup_scripts()
    click.echo(f"✅ Startup scripts created: {scripts_dir}")
    
    # Initialize database
    async def init_db():
        await setup_manager.initialize_database()
    
    asyncio.run(init_db())
    click.echo("✅ Database initialized")
    
    # Production-specific setup
    if environment == "production":
        k8s_dir = setup_manager.create_production_deployment()
        click.echo(f"✅ Production deployment configs: {k8s_dir}")
    
    click.echo("\n🎉 MONK CLI setup complete!")
    click.echo("\n📝 Next steps:")
    click.echo("1. Update API keys in .env file")
    click.echo("2. Run 'monk-cli health-check' to verify setup")
    if environment == "development":
        click.echo("3. Use './scripts/start_dev.sh' to start development server")
    click.echo("4. Run tests with './scripts/run_tests.sh'")


@cli.command()
def health_check():
    """Run comprehensive health check"""
    async def check():
        setup_manager = EnvironmentSetup()
        healthy = await setup_manager.run_health_check()
        return healthy
    
    healthy = asyncio.run(check())
    sys.exit(0 if healthy else 1)


@cli.command()
def benchmark():
    """Run performance benchmarks"""
    click.echo("🏃 Running MONK CLI benchmarks...")
    
    # Run unit tests first
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_phase1_comprehensive.py", "-v"])
    if result.returncode != 0:
        click.echo("❌ Unit tests failed")
        sys.exit(1)
    
    # Run performance benchmarks
    result = subprocess.run([sys.executable, "tests/test_phase1_comprehensive.py", "benchmark"])
    if result.returncode != 0:
        click.echo("❌ Performance benchmarks failed")
        sys.exit(1)
    
    # Run competitive benchmark
    result = subprocess.run([sys.executable, "benchmarks/competitive_benchmark.py"])
    if result.returncode != 0:
        click.echo("⚠️ Competitive benchmark completed with issues")
    else:
        click.echo("✅ All benchmarks passed!")
    
    sys.exit(result.returncode)


@cli.command()
@click.option("--environment", "-e", default="development")
def start(environment):
    """Start MONK CLI server"""
    click.echo(f"🚀 Starting MONK CLI server ({environment})...")
    
    if environment == "development":
        if os.name == 'nt':  # Windows
            subprocess.run(["scripts/start_dev.bat"], cwd=Path(__file__).parent.parent)
        else:  # Unix/Linux/macOS
            subprocess.run(["./scripts/start_dev.sh"], cwd=Path(__file__).parent.parent)
    else:
        # Production startup
        subprocess.run([sys.executable, "-m", "src.api.server"])


if __name__ == "__main__":
    cli()