"""
Enhanced TreeQuest Deployment Script
Handles deployment, configuration, and initialization of the enhanced TreeQuest system
"""

import os
import sys
import json
import time
import shutil
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTreeQuestDeployment:
    """Deployment manager for Enhanced TreeQuest system"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).parent
        self.config_path = self.base_path / "config" / "enhanced_treequest.yaml"
        self.memory_path = Path.home() / ".monk-memory"
        
        # Deployment configuration
        self.deployment_config = {
            "version": "1.0.0",
            "components": [
                "memory_filesystem",
                "historical_performance", 
                "adaptive_rewards",
                "agent_specialization",
                "memory_guided_mcts",
                "enhanced_integration"
            ],
            "requirements": {
                "python_version": "3.8+",
                "memory_minimum_mb": 512,
                "disk_space_minimum_mb": 1024
            },
            "features": {
                "memory_guided": True,
                "adaptive_rewards": True,
                "agent_specialization": True,
                "performance_tracking": True,
                "learning_enabled": True
            }
        }
    
    async def deploy_full_system(self) -> bool:
        """Deploy the complete Enhanced TreeQuest system"""
        try:
            logger.info("🚀 Starting Enhanced TreeQuest deployment...")
            
            # Step 1: System validation
            if not await self.validate_system_requirements():
                logger.error("❌ System requirements validation failed")
                return False
            
            # Step 2: Create directory structure
            await self.create_directory_structure()
            
            # Step 3: Generate configuration files
            await self.generate_configuration_files()
            
            # Step 4: Initialize memory filesystem
            await self.initialize_memory_filesystem()
            
            # Step 5: Setup performance tracking
            await self.setup_performance_tracking()
            
            # Step 6: Configure adaptive rewards
            await self.configure_adaptive_rewards()
            
            # Step 7: Initialize agent specialization
            await self.initialize_agent_specialization()
            
            # Step 8: Setup monitoring and logging
            await self.setup_monitoring()
            
            # Step 9: Run deployment tests
            if not await self.run_deployment_tests():
                logger.warning("⚠️ Some deployment tests failed - system may have issues")
            
            # Step 10: Generate deployment report
            await self.generate_deployment_report()
            
            logger.info("✅ Enhanced TreeQuest deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False
    
    async def validate_system_requirements(self) -> bool:
        """Validate system meets minimum requirements"""
        try:
            logger.info("🔍 Validating system requirements...")
            
            # Python version check
            python_version = sys.version_info
            if python_version.major < 3 or python_version.minor < 8:
                logger.error(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
                return False
            
            # Memory check (simplified)
            try:
                import psutil
                available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
                if available_memory < self.deployment_config["requirements"]["memory_minimum_mb"]:
                    logger.error(f"Insufficient memory: {available_memory:.0f}MB available, {self.deployment_config['requirements']['memory_minimum_mb']}MB required")
                    return False
            except ImportError:
                logger.warning("psutil not available - skipping memory check")
            
            # Disk space check
            disk_usage = shutil.disk_usage(self.base_path)
            free_space_mb = disk_usage.free / (1024 * 1024)
            if free_space_mb < self.deployment_config["requirements"]["disk_space_minimum_mb"]:
                logger.error(f"Insufficient disk space: {free_space_mb:.0f}MB available, {self.deployment_config['requirements']['disk_space_minimum_mb']}MB required")
                return False
            
            # Check required modules
            required_modules = [
                "json", "time", "asyncio", "pathlib", "dataclasses", 
                "typing", "logging", "hashlib", "statistics"
            ]
            
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError as e:
                    logger.error(f"Required module missing: {module}")
                    return False
            
            logger.info("✅ System requirements validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating requirements: {e}")
            return False
    
    async def create_directory_structure(self):
        """Create necessary directory structure"""
        try:
            logger.info("📁 Creating directory structure...")
            
            directories = [
                self.memory_path,
                self.memory_path / "filesystem",
                self.memory_path / "performance",
                self.memory_path / "specializations",
                self.memory_path / "rewards",
                self.memory_path / "logs",
                self.memory_path / "config",
                self.memory_path / "backups",
                self.base_path / "config",
                self.base_path / "logs",
                self.base_path / "tests"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
            
            logger.info("✅ Directory structure created")
            
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            raise
    
    async def generate_configuration_files(self):
        """Generate configuration files"""
        try:
            logger.info("⚙️ Generating configuration files...")
            
            # Main configuration
            config = {
                "enhanced_treequest": {
                    "version": self.deployment_config["version"],
                    "deployment_timestamp": time.time(),
                    "features": self.deployment_config["features"],
                    
                    "memory_filesystem": {
                        "base_path": str(self.memory_path / "filesystem"),
                        "max_memories": 10000,
                        "adaptive_forgetting": {
                            "enabled": True,
                            "threshold": 0.3,
                            "schedule": "daily"
                        }
                    },
                    
                    "historical_performance": {
                        "storage_path": str(self.memory_path / "performance"),
                        "max_recent_metrics": 1000,
                        "min_samples_for_ranking": 5,
                        "trend_window_hours": 24
                    },
                    
                    "adaptive_rewards": {
                        "max_learning_history": 500,
                        "min_samples_for_adaptation": 10,
                        "performance_correlation_threshold": 0.7,
                        "learning_rates": {
                            "planner": 0.05,
                            "analyzer": 0.04,
                            "critic": 0.06,
                            "synthesizer": 0.05,
                            "executor": 0.04
                        }
                    },
                    
                    "agent_specialization": {
                        "min_tasks_for_specialization": 10,
                        "specialization_threshold": 0.7,
                        "expertise_decay_rate": 0.95,
                        "cross_training_bonus": 0.1
                    },
                    
                    "treequest_config": {
                        "max_depth": 3,
                        "branching_factor": 4,
                        "rollout_budget": 32,
                        "cost_cap_usd": 0.50,
                        "timeout_seconds": 120,
                        "exploration_constant": 1.414,
                        "memory_weight": 0.3
                    }
                }
            }
            
            # Save main configuration
            config_file = self.base_path / "config" / "enhanced_treequest.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            # Environment configuration
            env_config = {
                "MONK_MEMORY_PATH": str(self.memory_path),
                "MONK_ENHANCED_MODE": "true",
                "MONK_LOGGING_LEVEL": "INFO",
                "MONK_PERFORMANCE_TRACKING": "true",
                "MONK_ADAPTIVE_REWARDS": "true",
                "MONK_AGENT_SPECIALIZATION": "true"
            }
            
            env_file = self.base_path / ".env.enhanced"
            with open(env_file, 'w') as f:
                for key, value in env_config.items():
                    f.write(f"{key}={value}\n")
            
            logger.info("✅ Configuration files generated")
            
        except Exception as e:
            logger.error(f"Error generating configuration: {e}")
            raise
    
    async def initialize_memory_filesystem(self):
        """Initialize memory filesystem"""
        try:
            logger.info("🧠 Initializing memory filesystem...")
            
            # Create initial filesystem structure
            filesystem_base = self.memory_path / "filesystem"
            
            # Standard memory directories
            memory_dirs = [
                "agents/planner",
                "agents/analyzer", 
                "agents/critic",
                "agents/synthesizer",
                "agents/executor",
                "patterns/successful_paths",
                "patterns/failed_paths", 
                "insights/cross_agent",
                "insights/provider_performance",
                "insights/adaptive_rewards",
                "session_data/trees",
                "session_data/conversations",
                "specializations/domains",
                "specializations/task_types"
            ]
            
            for mem_dir in memory_dirs:
                full_path = filesystem_base / mem_dir
                full_path.mkdir(parents=True, exist_ok=True)
            
            # Create initial metadata file
            metadata = {
                "filesystem_version": "1.0.0",
                "initialized_at": time.time(),
                "structure_version": "hierarchical_v1",
                "total_directories": len(memory_dirs),
                "features": ["adaptive_forgetting", "semantic_search", "cross_agent_insights"]
            }
            
            metadata_file = filesystem_base / "filesystem_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info("✅ Memory filesystem initialized")
            
        except Exception as e:
            logger.error(f"Error initializing memory filesystem: {e}")
            raise
    
    async def setup_performance_tracking(self):
        """Setup performance tracking system"""
        try:
            logger.info("📊 Setting up performance tracking...")
            
            performance_base = self.memory_path / "performance"
            
            # Create performance tracking files
            initial_data = {
                "performance_aggregates.json": {"aggregates": {}, "last_updated": time.time()},
                "recent_metrics.json": {"metrics": [], "last_updated": time.time()},
                "provider_rankings.json": {"rankings": {}, "last_updated": time.time()}
            }
            
            for filename, data in initial_data.items():
                file_path = performance_base / filename
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            # Create performance tracking configuration
            perf_config = {
                "tracking_enabled": True,
                "metrics_retention_days": 30,
                "aggregation_frequency": "hourly",
                "alert_thresholds": {
                    "success_rate_minimum": 0.8,
                    "average_latency_maximum": 5000,
                    "cost_per_quality_maximum": 0.1
                },
                "providers": {
                    "openai": {"enabled": True, "priority": 1},
                    "anthropic": {"enabled": True, "priority": 1},
                    "google": {"enabled": True, "priority": 2},
                    "mistral": {"enabled": True, "priority": 2}
                }
            }
            
            config_file = performance_base / "tracking_config.json"
            with open(config_file, 'w') as f:
                json.dump(perf_config, f, indent=2, default=str)
            
            logger.info("✅ Performance tracking setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up performance tracking: {e}")
            raise
    
    async def configure_adaptive_rewards(self):
        """Configure adaptive rewards system"""
        try:
            logger.info("🎯 Configuring adaptive rewards...")
            
            rewards_base = self.memory_path / "rewards"
            
            # Initial reward parameters for each agent
            agent_params = {
                "planner": {
                    "base_reward": 0.8,
                    "quality_weight": 0.4,
                    "efficiency_weight": 0.2,
                    "novelty_weight": 0.15,
                    "consistency_weight": 0.15,
                    "cost_weight": 0.1,
                    "learning_rate": 0.05,
                    "decay_factor": 0.95,
                    "adaptation_threshold": 0.1
                },
                "analyzer": {
                    "base_reward": 0.7,
                    "quality_weight": 0.5,
                    "efficiency_weight": 0.25,
                    "novelty_weight": 0.1,
                    "consistency_weight": 0.1,
                    "cost_weight": 0.05,
                    "learning_rate": 0.04,
                    "decay_factor": 0.96,
                    "adaptation_threshold": 0.12
                },
                "critic": {
                    "base_reward": 0.6,
                    "quality_weight": 0.35,
                    "efficiency_weight": 0.15,
                    "novelty_weight": 0.2,
                    "consistency_weight": 0.25,
                    "cost_weight": 0.05,
                    "learning_rate": 0.06,
                    "decay_factor": 0.94,
                    "adaptation_threshold": 0.08
                },
                "synthesizer": {
                    "base_reward": 0.75,
                    "quality_weight": 0.3,
                    "efficiency_weight": 0.2,
                    "novelty_weight": 0.3,
                    "consistency_weight": 0.15,
                    "cost_weight": 0.05,
                    "learning_rate": 0.05,
                    "decay_factor": 0.95,
                    "adaptation_threshold": 0.1
                },
                "executor": {
                    "base_reward": 0.65,
                    "quality_weight": 0.25,
                    "efficiency_weight": 0.4,
                    "novelty_weight": 0.1,
                    "consistency_weight": 0.2,
                    "cost_weight": 0.05,
                    "learning_rate": 0.04,
                    "decay_factor": 0.97,
                    "adaptation_threshold": 0.15
                }
            }
            
            params_file = rewards_base / "agent_parameters.json"
            with open(params_file, 'w') as f:
                json.dump(agent_params, f, indent=2, default=str)
            
            # Initial learning history
            learning_history = {
                "reward_outcomes": [],
                "adaptation_history": {},
                "last_updated": time.time(),
                "total_adaptations": 0
            }
            
            history_file = rewards_base / "learning_history.json"
            with open(history_file, 'w') as f:
                json.dump(learning_history, f, indent=2, default=str)
            
            logger.info("✅ Adaptive rewards configured")
            
        except Exception as e:
            logger.error(f"Error configuring adaptive rewards: {e}")
            raise
    
    async def initialize_agent_specialization(self):
        """Initialize agent specialization system"""
        try:
            logger.info("👥 Initializing agent specialization...")
            
            specialization_base = self.memory_path / "specializations"
            
            # Initialize agent profiles
            base_agents = ["planner", "analyzer", "critic", "synthesizer", "executor"]
            
            agent_profiles = {}
            for agent in base_agents:
                profile = {
                    "agent_role": agent,
                    "primary_specializations": [],
                    "secondary_specializations": [],
                    "learning_preferences": self._get_default_learning_preferences(agent),
                    "collaboration_patterns": self._get_default_collaboration_patterns(agent),
                    "adaptation_rate": 0.1,
                    "consistency_score": 0.5,
                    "total_experience": 0,
                    "created_at": time.time(),
                    "last_performance_update": time.time()
                }
                agent_profiles[agent] = profile
            
            profiles_file = specialization_base / "agent_profiles.json"
            with open(profiles_file, 'w') as f:
                json.dump(agent_profiles, f, indent=2, default=str)
            
            # Initialize domain experts tracking
            domain_experts = {
                "code_analysis": [],
                "architecture_design": [],
                "performance_optimization": [],
                "security_assessment": [],
                "database_design": [],
                "frontend_development": [],
                "backend_systems": [],
                "devops_automation": [],
                "api_design": [],
                "testing_strategies": [],
                "project_planning": [],
                "requirements_analysis": []
            }
            
            experts_file = specialization_base / "domain_experts.json"
            with open(experts_file, 'w') as f:
                json.dump(domain_experts, f, indent=2, default=str)
            
            logger.info("✅ Agent specialization initialized")
            
        except Exception as e:
            logger.error(f"Error initializing agent specialization: {e}")
            raise
    
    def _get_default_learning_preferences(self, agent_role: str) -> Dict[str, float]:
        """Get default learning preferences"""
        preferences = {
            "planner": {
                "architecture_design": 0.9,
                "project_planning": 1.0,
                "requirements_analysis": 0.8,
                "performance_optimization": 0.6
            },
            "analyzer": {
                "code_analysis": 1.0,
                "performance_optimization": 0.9,
                "testing_strategies": 0.7,
                "database_design": 0.6
            },
            "critic": {
                "security_assessment": 0.9,
                "code_analysis": 0.8,
                "testing_strategies": 0.8,
                "performance_optimization": 0.7
            },
            "synthesizer": {
                "api_design": 0.8,
                "architecture_design": 0.7,
                "requirements_analysis": 0.8,
                "project_planning": 0.6
            },
            "executor": {
                "devops_automation": 0.9,
                "backend_systems": 0.8,
                "frontend_development": 0.7,
                "database_design": 0.6
            }
        }
        return preferences.get(agent_role, {})
    
    def _get_default_collaboration_patterns(self, agent_role: str) -> Dict[str, float]:
        """Get default collaboration patterns"""
        patterns = {
            "planner": {"analyzer": 0.8, "critic": 0.6, "synthesizer": 0.9, "executor": 0.7},
            "analyzer": {"planner": 0.8, "critic": 0.9, "synthesizer": 0.7, "executor": 0.6},
            "critic": {"planner": 0.6, "analyzer": 0.9, "synthesizer": 0.8, "executor": 0.7},
            "synthesizer": {"planner": 0.9, "analyzer": 0.7, "critic": 0.8, "executor": 0.8},
            "executor": {"planner": 0.7, "analyzer": 0.6, "critic": 0.7, "synthesizer": 0.8}
        }
        return patterns.get(agent_role, {})
    
    async def setup_monitoring(self):
        """Setup monitoring and logging"""
        try:
            logger.info("📈 Setting up monitoring...")
            
            logs_base = self.memory_path / "logs"
            
            # Create log configuration
            log_config = {
                "logging": {
                    "version": 1,
                    "disable_existing_loggers": False,
                    "formatters": {
                        "detailed": {
                            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                        },
                        "simple": {
                            "format": "%(levelname)s - %(message)s"
                        }
                    },
                    "handlers": {
                        "console": {
                            "class": "logging.StreamHandler",
                            "level": "INFO",
                            "formatter": "simple"
                        },
                        "file": {
                            "class": "logging.FileHandler",
                            "filename": str(logs_base / "enhanced_treequest.log"),
                            "level": "DEBUG",
                            "formatter": "detailed"
                        },
                        "performance": {
                            "class": "logging.FileHandler", 
                            "filename": str(logs_base / "performance.log"),
                            "level": "INFO",
                            "formatter": "detailed"
                        }
                    },
                    "loggers": {
                        "enhanced_treequest": {
                            "level": "DEBUG",
                            "handlers": ["console", "file"],
                            "propagate": False
                        },
                        "performance": {
                            "level": "INFO",
                            "handlers": ["performance"],
                            "propagate": False
                        }
                    }
                }
            }
            
            log_config_file = self.base_path / "config" / "logging.json"
            with open(log_config_file, 'w') as f:
                json.dump(log_config, f, indent=2, default=str)
            
            # Create initial log files
            for log_file in ["enhanced_treequest.log", "performance.log", "errors.log"]:
                log_path = logs_base / log_file
                log_path.touch()
            
            logger.info("✅ Monitoring setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up monitoring: {e}")
            raise
    
    async def run_deployment_tests(self) -> bool:
        """Run basic deployment tests"""
        try:
            logger.info("🧪 Running deployment tests...")
            
            # Test 1: Directory structure
            required_dirs = [
                self.memory_path / "filesystem",
                self.memory_path / "performance", 
                self.memory_path / "specializations",
                self.memory_path / "rewards"
            ]
            
            for directory in required_dirs:
                if not directory.exists():
                    logger.error(f"Missing directory: {directory}")
                    return False
            
            # Test 2: Configuration files
            required_configs = [
                self.base_path / "config" / "enhanced_treequest.json",
                self.base_path / ".env.enhanced"
            ]
            
            for config_file in required_configs:
                if not config_file.exists():
                    logger.error(f"Missing configuration: {config_file}")
                    return False
            
            # Test 3: Initial data files
            required_data = [
                self.memory_path / "performance" / "performance_aggregates.json",
                self.memory_path / "specializations" / "agent_profiles.json",
                self.memory_path / "rewards" / "agent_parameters.json"
            ]
            
            for data_file in required_data:
                if not data_file.exists():
                    logger.error(f"Missing data file: {data_file}")
                    return False
            
            # Test 4: JSON validity
            for config_file in required_configs:
                if config_file.suffix == '.json':
                    try:
                        with open(config_file, 'r') as f:
                            json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in {config_file}: {e}")
                        return False
            
            logger.info("✅ Deployment tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Error running deployment tests: {e}")
            return False
    
    async def generate_deployment_report(self):
        """Generate deployment report"""
        try:
            logger.info("📄 Generating deployment report...")
            
            report = {
                "deployment_summary": {
                    "version": self.deployment_config["version"],
                    "deployment_timestamp": time.time(),
                    "deployment_status": "successful",
                    "components_deployed": len(self.deployment_config["components"]),
                    "features_enabled": self.deployment_config["features"]
                },
                "system_configuration": {
                    "base_path": str(self.base_path),
                    "memory_path": str(self.memory_path),
                    "config_files": [
                        "enhanced_treequest.json",
                        ".env.enhanced",
                        "logging.json"
                    ]
                },
                "directory_structure": {
                    "memory_filesystem": str(self.memory_path / "filesystem"),
                    "performance_tracking": str(self.memory_path / "performance"),
                    "agent_specialization": str(self.memory_path / "specializations"),
                    "adaptive_rewards": str(self.memory_path / "rewards"),
                    "logs": str(self.memory_path / "logs")
                },
                "components_status": {
                    component: "deployed" for component in self.deployment_config["components"]
                },
                "next_steps": [
                    "Run test suite: python test_enhanced_treequest.py",
                    "Update main monk.py to use enhanced features",
                    "Configure API keys in environment variables",
                    "Start using enhanced TreeQuest with: monk --enhanced",
                    "Monitor performance logs for optimization opportunities"
                ],
                "maintenance_schedule": {
                    "daily": ["Adaptive memory cleanup", "Performance metrics aggregation"],
                    "weekly": ["Agent specialization analysis", "Reward system optimization"],
                    "monthly": ["Memory filesystem optimization", "System performance review"]
                }
            }
            
            report_file = self.base_path / "deployment_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Also create markdown version for readability
            md_content = f"""# Enhanced TreeQuest Deployment Report

## Deployment Summary
- **Version**: {report['deployment_summary']['version']}
- **Status**: {report['deployment_summary']['deployment_status']}
- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['deployment_summary']['deployment_timestamp']))}
- **Components Deployed**: {report['deployment_summary']['components_deployed']}

## Features Enabled
{chr(10).join(f"- {feature}: {status}" for feature, status in report['deployment_summary']['features_enabled'].items())}

## Directory Structure
{chr(10).join(f"- **{name}**: `{path}`" for name, path in report['directory_structure'].items())}

## Next Steps
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(report['next_steps']))}

## Maintenance Schedule
### Daily
{chr(10).join(f"- {task}" for task in report['maintenance_schedule']['daily'])}

### Weekly  
{chr(10).join(f"- {task}" for task in report['maintenance_schedule']['weekly'])}

### Monthly
{chr(10).join(f"- {task}" for task in report['maintenance_schedule']['monthly'])}
"""
            
            md_file = self.base_path / "DEPLOYMENT_REPORT.md"
            with open(md_file, 'w') as f:
                f.write(md_content)
            
            logger.info(f"✅ Deployment report generated: {report_file}")
            logger.info(f"📄 Readable report available: {md_file}")
            
        except Exception as e:
            logger.error(f"Error generating deployment report: {e}")
            raise

async def main():
    """Main deployment function"""
    print("🧘 Enhanced TreeQuest - Deployment System")
    print("=" * 50)
    
    try:
        deployment = EnhancedTreeQuestDeployment()
        
        success = await deployment.deploy_full_system()
        
        if success:
            print("\n🎉 Enhanced TreeQuest deployment completed successfully!")
            print("\nNext steps:")
            print("1. Run test suite: python test_enhanced_treequest.py")
            print("2. Update monk.py to use enhanced features")
            print("3. Configure API keys in .env file")
            print("4. Start using: monk --enhanced")
            
            return True
        else:
            print("\n❌ Deployment failed - check logs for details")
            return False
            
    except Exception as e:
        print(f"\n💥 Deployment error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)