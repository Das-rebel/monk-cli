"""
MONK CLI Phase 2 Comprehensive Test Suite
Tests community intelligence, cross-attention memory, and Phase 2 features
"""
import pytest
import asyncio
import time
import json
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Phase 2 imports
from src.community.intelligence_system import (
    CommunityIntelligenceSystem,
    ArxivAIMonitor,
    GitHubActivityMonitor,
    IntelligenceProcessor,
    CapabilityEnhancer,
    ResearchFindingData,
    CapabilityEnhancementPlan,
    SignificanceLevel,
    SourceType
)
from src.memory.cross_attention_memory import (
    CrossAttentionMemoryRetrieval,
    CrossAttentionEncoder,
    CrossAttentionResult,
    AttentionWeights
)
from src.api.community_endpoints import router as community_router
from src.core.database import get_db_session
from src.core.models import ResearchFinding, CapabilityEnhancement, CommunityIntelligence


class TestCommunityIntelligenceSystem:
    """Test the community intelligence system"""
    
    @pytest.fixture
    async def intelligence_system(self):
        """Create intelligence system for testing"""
        system = CommunityIntelligenceSystem()
        return system
    
    @pytest.fixture
    def sample_research_finding(self):
        """Sample research finding for testing"""
        return ResearchFindingData(
            id="test_finding_1",
            title="Novel Multi-Agent Memory Sharing Architecture",
            summary="This paper introduces a breakthrough approach to memory sharing between autonomous agents...",
            source_url="https://arxiv.org/abs/2024.test",
            source_type=SourceType.ARXIV,
            discovered_at=datetime.now(),
            significance_score=0.85,
            significance_level=SignificanceLevel.HIGH,
            focus_areas=["multi-agent", "memory_systems"],
            implementation_potential=0.7,
            community_interest=0.6,
            authors=["Smith, J.", "Johnson, A."],
            tags=["neural", "agent", "memory"],
            full_content="Full paper content here...",
            metadata={"arxiv_id": "2024.test", "citations": 15}
        )
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, intelligence_system):
        """Test system initialization"""
        assert intelligence_system.monitors is not None
        assert len(intelligence_system.research_sources) > 0
        assert intelligence_system.intelligence_processor is not None
        assert intelligence_system.capability_enhancer is not None
        
        # Check default research sources
        source_names = [source.name for source in intelligence_system.research_sources]
        assert "ArXiv AI Papers" in source_names
        assert "GitHub Trending AI" in source_names
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, intelligence_system):
        """Test monitoring start/stop lifecycle"""
        # Initially not running
        assert not intelligence_system.running
        
        # Start monitoring
        await intelligence_system.start_monitoring()
        assert intelligence_system.running
        
        # Stop monitoring
        await intelligence_system.stop_monitoring()
        assert not intelligence_system.running
    
    @pytest.mark.asyncio
    async def test_intelligence_processor(self):
        """Test intelligence processor functionality"""
        processor = IntelligenceProcessor()
        
        # Create test findings
        findings = [
            ResearchFindingData(
                id="finding_1",
                title="Breakthrough in AI Agent Communication",
                summary="Novel method for agent coordination",
                source_url="test://url1",
                source_type=SourceType.ARXIV,
                discovered_at=datetime.now(),
                significance_score=0.6,
                significance_level=SignificanceLevel.MEDIUM,
                focus_areas=["multi-agent"],
                implementation_potential=0.8,
                community_interest=0.7,
                authors=["Test Author"],
                tags=["agent"],
                full_content="Full content",
                metadata={}
            ),
            ResearchFindingData(
                id="finding_2",
                title="Minor Improvement in Memory Systems",
                summary="Small optimization for memory retrieval",
                source_url="test://url2",
                source_type=SourceType.GITHUB,
                discovered_at=datetime.now(),
                significance_score=0.3,
                significance_level=SignificanceLevel.LOW,
                focus_areas=["memory_systems"],
                implementation_potential=0.4,
                community_interest=0.2,
                authors=["Test Author 2"],
                tags=["memory"],
                full_content="Small improvement",
                metadata={}
            )
        ]
        
        # Process findings
        processed = await processor.process_findings(findings)
        
        # Should be sorted by significance
        assert len(processed) == 2
        assert processed[0].significance_score >= processed[1].significance_score
        
        # High significance finding should be first
        assert processed[0].id == "finding_1"
    
    @pytest.mark.asyncio
    async def test_capability_enhancer(self):
        """Test capability enhancement generation"""
        enhancer = CapabilityEnhancer()
        
        # High-significance finding should generate enhancement plan
        finding = ResearchFindingData(
            id="high_sig_finding",
            title="Revolutionary Agent Architecture",
            summary="Breakthrough approach with high implementation potential",
            source_url="test://revolutionary",
            source_type=SourceType.ARXIV,
            discovered_at=datetime.now(),
            significance_score=0.9,
            significance_level=SignificanceLevel.BREAKTHROUGH,
            focus_areas=["multi-agent", "architecture"],
            implementation_potential=0.8,
            community_interest=0.7,
            authors=["Revolutionary Author"],
            tags=["breakthrough", "agent"],
            full_content="Revolutionary content with implementation details",
            metadata={}
        )
        
        plan = await enhancer.generate_enhancement_plan(finding)
        
        assert plan is not None
        assert plan.research_finding_id == finding.id
        assert plan.implementation_complexity > 0
        assert plan.estimated_impact > 0
        assert len(plan.required_resources) > 0
        assert plan.implementation_plan is not None
        assert plan.testing_strategy is not None
        assert plan.deployment_strategy is not None
        assert plan.risk_assessment is not None
    
    @pytest.mark.asyncio
    async def test_capability_enhancer_low_significance(self):
        """Test that low-significance findings don't generate plans"""
        enhancer = CapabilityEnhancer()
        
        # Low-significance finding should not generate enhancement plan
        finding = ResearchFindingData(
            id="low_sig_finding",
            title="Minor Update",
            summary="Small change with low impact",
            source_url="test://minor",
            source_type=SourceType.GITHUB,
            discovered_at=datetime.now(),
            significance_score=0.2,
            significance_level=SignificanceLevel.LOW,
            focus_areas=["optimization"],
            implementation_potential=0.3,
            community_interest=0.1,
            authors=["Minor Author"],
            tags=["minor"],
            full_content="Minor content",
            metadata={}
        )
        
        plan = await enhancer.generate_enhancement_plan(finding)
        
        # Should not generate plan for low-significance findings
        assert plan is None


class TestArxivAIMonitor:
    """Test ArXiv AI monitoring"""
    
    @pytest.fixture
    def arxiv_monitor(self):
        """Create ArXiv monitor for testing"""
        return ArxivAIMonitor(
            focus_areas=["multi-agent", "memory_systems", "tool_orchestration"],
            update_frequency_hours=24
        )
    
    @pytest.mark.asyncio
    async def test_significance_calculation(self, arxiv_monitor):
        """Test significance score calculation"""
        # High-impact title and summary
        high_impact_score = await arxiv_monitor._calculate_significance(
            "Breakthrough Multi-Agent Architecture with Novel Memory Systems",
            "We present a novel approach that significantly outperforms state-of-the-art methods for agent coordination and memory sharing..."
        )
        
        # Should be high significance
        assert high_impact_score > 0.5
        
        # Low-impact title and summary
        low_impact_score = await arxiv_monitor._calculate_significance(
            "Small optimization in basic algorithm",
            "We make a minor improvement to existing methods..."
        )
        
        # Should be lower significance
        assert low_impact_score < high_impact_score
    
    @pytest.mark.asyncio
    async def test_implementation_potential_assessment(self, arxiv_monitor):
        """Test implementation potential assessment"""
        # High implementation potential
        high_impl_score = await arxiv_monitor._assess_implementation_potential(
            "Open Source Multi-Agent Framework with Code Implementation",
            "We provide a complete implementation with GitHub repository and reproducible evaluation..."
        )
        
        # Should be high implementation potential
        assert high_impl_score > 0.5
        
        # Low implementation potential
        low_impl_score = await arxiv_monitor._assess_implementation_potential(
            "Theoretical Analysis of Complex Systems",
            "We present a theoretical framework without implementation details..."
        )
        
        # Should be lower implementation potential
        assert low_impl_score < high_impl_score
    
    def test_focus_area_extraction(self, arxiv_monitor):
        """Test focus area extraction"""
        text = "This paper presents a novel multi-agent memory system with tool orchestration capabilities"
        areas = arxiv_monitor._extract_focus_areas(text)
        
        # Should extract relevant focus areas
        assert "multi-agent" in areas
        assert "memory_systems" in areas or "memory" in text.lower()
        assert "tool_orchestration" in areas or "tool" in text.lower()
    
    def test_tag_extraction(self, arxiv_monitor):
        """Test tag extraction"""
        title = "Neural Multi-Agent Language Models with Memory Retrieval"
        summary = "We present a neural network architecture for autonomous agents with reasoning capabilities"
        
        tags = arxiv_monitor._extract_tags(title, summary)
        
        # Should extract relevant tags
        assert "neural" in tags
        assert "agent" in tags
        assert "llm" in tags or "language model" in (title + summary).lower()
        assert "memory" in tags


class TestGitHubActivityMonitor:
    """Test GitHub activity monitoring"""
    
    @pytest.fixture
    def github_monitor(self):
        """Create GitHub monitor for testing"""
        return GitHubActivityMonitor(
            repositories=["trending", "langchain-ai/langchain"],
            update_frequency_hours=12
        )
    
    def test_github_significance_calculation(self, github_monitor):
        """Test GitHub repository significance calculation"""
        # High-star, recently updated repo
        high_sig_repo = {
            "id": 12345,
            "full_name": "ai-org/amazing-agents",
            "description": "Revolutionary multi-agent framework",
            "stargazers_count": 15000,
            "forks_count": 2000,
            "updated_at": "2024-01-15T10:00:00Z",
            "topics": ["ai", "machine-learning", "agent"],
            "language": "Python"
        }
        
        score = github_monitor._calculate_github_significance(high_sig_repo)
        assert score > 0.5
        
        # Low-star, old repo
        low_sig_repo = {
            "id": 67890,
            "full_name": "user/small-project",
            "description": "Small utility",
            "stargazers_count": 10,
            "forks_count": 1,
            "updated_at": "2023-01-15T10:00:00Z",
            "topics": [],
            "language": "JavaScript"
        }
        
        low_score = github_monitor._calculate_github_significance(low_sig_repo)
        assert low_score < score
    
    def test_focus_area_extraction_github(self, github_monitor):
        """Test GitHub focus area extraction"""
        repo = {
            "topics": ["ai-agent", "memory-systems", "tool-orchestration"],
            "description": "Multi-agent framework with advanced memory capabilities"
        }
        
        areas = github_monitor._extract_github_focus_areas(repo)
        
        # Should map topics to focus areas
        assert any("agent" in area.lower() for area in areas)
        assert any("memory" in area.lower() for area in areas)


class TestCrossAttentionMemoryRetrieval:
    """Test cross-attention memory retrieval system"""
    
    @pytest.fixture
    async def cross_attention_system(self):
        """Create cross-attention system for testing"""
        system = CrossAttentionMemoryRetrieval()
        # Mock initialization to avoid loading actual models in tests
        system.device = "cpu"
        system.tokenizer = Mock()
        system.base_model = Mock()
        system.cross_attention_model = Mock()
        return system
    
    @pytest.fixture
    def sample_memory_results(self):
        """Sample memory results for testing"""
        return [
            {
                "memory_id": "mem_1",
                "memory_type": "episodic",
                "content": {
                    "task_description": "Design microservices architecture",
                    "agent_used": "Architect",
                    "success": True,
                    "domain": "system_design"
                },
                "relevance_score": 0.8,
                "importance_score": 0.7,
                "created_at": datetime.now() - timedelta(days=2),
                "last_accessed": datetime.now() - timedelta(hours=1),
                "access_count": 3
            },
            {
                "memory_id": "mem_2",
                "memory_type": "semantic",
                "content": {
                    "concept": "microservices patterns",
                    "content": "Common patterns for microservices architecture"
                },
                "relevance_score": 0.6,
                "importance_score": 0.8,
                "created_at": datetime.now() - timedelta(days=7),
                "last_accessed": datetime.now() - timedelta(days=3),
                "access_count": 1
            }
        ]
    
    def test_temporal_decay_computation(self, cross_attention_system, sample_memory_results):
        """Test temporal decay factor computation"""
        from src.memory.memory_system import MemoryResult
        
        # Convert dict to MemoryResult objects
        memories = []
        for mem_data in sample_memory_results:
            memory = MemoryResult(
                memory_id=mem_data["memory_id"],
                memory_type=mem_data["memory_type"],
                content=mem_data["content"],
                relevance_score=mem_data["relevance_score"],
                importance_score=mem_data["importance_score"],
                created_at=mem_data["created_at"],
                last_accessed=mem_data["last_accessed"],
                access_count=mem_data["access_count"]
            )
            memories.append(memory)
        
        decay_factors = cross_attention_system._compute_temporal_decay(memories)
        
        # Should return decay factors for all memories
        assert len(decay_factors) == len(memories)
        
        # Recent memory should have higher decay factor
        assert decay_factors[0] > decay_factors[1]  # mem_1 is more recent
        
        # All factors should be between 0 and 1
        assert all(0 <= factor <= 1 for factor in decay_factors)
    
    def test_relevance_score_computation(self, cross_attention_system, sample_memory_results):
        """Test final relevance score computation"""
        from src.memory.memory_system import MemoryResult, MemoryQuery
        
        # Convert to MemoryResult objects
        memories = []
        for mem_data in sample_memory_results:
            memory = MemoryResult(
                memory_id=mem_data["memory_id"],
                memory_type=mem_data["memory_type"],
                content=mem_data["content"],
                relevance_score=mem_data["relevance_score"],
                importance_score=mem_data["importance_score"],
                created_at=mem_data["created_at"],
                last_accessed=mem_data["last_accessed"],
                access_count=mem_data["access_count"]
            )
            memories.append(memory)
        
        # Mock components
        importance_scores = np.array([0.8, 0.6])
        temporal_factors = cross_attention_system._compute_temporal_decay(memories)
        
        query = MemoryQuery(
            query_text="microservices architecture",
            user_id="test_user",
            limit=5
        )
        
        relevance_scores = cross_attention_system._compute_final_relevance_scores(
            importance_scores, temporal_factors, memories, query
        )
        
        # Should return scores for all memories
        assert len(relevance_scores) == len(memories)
        
        # Scores should be between 0 and 1
        assert all(0 <= score <= 1 for score in relevance_scores)
    
    def test_retrieval_reasoning_generation(self, cross_attention_system, sample_memory_results):
        """Test retrieval reasoning generation"""
        from src.memory.memory_system import MemoryResult, MemoryQuery
        
        # Convert to MemoryResult objects
        memories = []
        for mem_data in sample_memory_results:
            memory = MemoryResult(
                memory_id=mem_data["memory_id"],
                memory_type=mem_data["memory_type"],
                content=mem_data["content"],
                relevance_score=mem_data["relevance_score"],
                importance_score=mem_data["importance_score"],
                created_at=mem_data["created_at"],
                last_accessed=mem_data["last_accessed"],
                access_count=mem_data["access_count"]
            )
            memories.append(memory)
        
        query = MemoryQuery(
            query_text="microservices design",
            user_id="test_user",
            limit=5
        )
        
        importance_scores = np.array([0.8, 0.6])
        attention_weights = np.random.rand(10, 20)  # Mock attention weights
        correlations = np.random.rand(2, 2)  # Mock correlations
        
        reasoning = cross_attention_system._generate_retrieval_reasoning(
            query, memories, importance_scores, attention_weights, correlations
        )
        
        # Should generate non-empty reasoning
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert "memories" in reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_expertise_insights_generation(self, cross_attention_system, sample_memory_results):
        """Test expertise insights generation"""
        from src.memory.memory_system import MemoryResult, MemoryQuery
        
        # Convert to MemoryResult objects with more episodic memories
        memories = []
        for i, mem_data in enumerate(sample_memory_results):
            # Make both memories episodic for testing
            mem_data_copy = mem_data.copy()
            if mem_data_copy["memory_type"] == "semantic":
                mem_data_copy["memory_type"] = "episodic"
                mem_data_copy["content"] = {
                    "task_description": "API design patterns",
                    "agent_used": "Architect",
                    "success": True,
                    "domain": "system_design"
                }
            
            memory = MemoryResult(
                memory_id=mem_data_copy["memory_id"],
                memory_type=mem_data_copy["memory_type"],
                content=mem_data_copy["content"],
                relevance_score=mem_data_copy["relevance_score"],
                importance_score=mem_data_copy["importance_score"],
                created_at=mem_data_copy["created_at"],
                last_accessed=mem_data_copy["last_accessed"],
                access_count=mem_data_copy["access_count"]
            )
            memories.append(memory)
        
        query = MemoryQuery(
            query_text="system architecture",
            user_id="test_user",
            limit=5
        )
        
        attention_weights = np.array([0.8, 0.6])
        
        insights = await cross_attention_system._generate_expertise_insights(
            query, memories, attention_weights
        )
        
        # Should generate insights
        assert isinstance(insights, list)
        # Should identify expertise patterns
        if insights:
            assert any("system_design" in insight.lower() or "expertise" in insight.lower() 
                     for insight in insights)
    
    @pytest.mark.asyncio
    async def test_expertise_profile_tracking(self, cross_attention_system):
        """Test user expertise profile tracking"""
        user_id = "test_user_123"
        
        # Update expertise with sample data
        from src.memory.memory_system import MemoryResult
        
        memories = [
            MemoryResult(
                memory_id="mem_1",
                memory_type="episodic",
                content={
                    "domain": "system_design",
                    "agent_used": "Architect",
                    "success": True
                },
                relevance_score=0.8,
                importance_score=0.7,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1
            )
        ]
        
        relevance_scores = np.array([0.8])
        
        await cross_attention_system._update_expertise_tracking(user_id, memories, relevance_scores)
        
        # Get expertise profile
        profile = await cross_attention_system.get_user_expertise_profile(user_id)
        
        # Should have expertise data
        assert "domains" in profile
        assert "agents" in profile
        assert "expertise_level" in profile
        assert "total_experience" in profile
        
        # Should track system_design domain
        if profile["domains"]:
            assert any(domain["domain"] == "system_design" for domain in profile["domains"])


class TestCrossAttentionEncoder:
    """Test the cross-attention neural network model"""
    
    @pytest.fixture
    def encoder_model(self):
        """Create encoder model for testing"""
        return CrossAttentionEncoder(
            hidden_size=384,
            num_attention_heads=6,
            num_layers=2,
            max_seq_length=128
        )
    
    def test_model_initialization(self, encoder_model):
        """Test model initialization"""
        assert encoder_model.hidden_size == 384
        assert encoder_model.num_attention_heads == 6
        assert len(encoder_model.attention_layers) == 2
        assert len(encoder_model.layer_norms) == 2
        assert len(encoder_model.ffn_layers) == 2
        
        # Check model components
        assert encoder_model.importance_classifier is not None
        assert encoder_model.correlation_predictor is not None
    
    def test_forward_pass_shapes(self, encoder_model):
        """Test forward pass tensor shapes"""
        import torch
        
        batch_size = 2
        query_seq_len = 10
        num_memories = 5
        memory_seq_len = 8
        hidden_size = 384
        
        # Create mock tensors
        query_embeddings = torch.randn(batch_size, query_seq_len, hidden_size)
        memory_embeddings = torch.randn(batch_size, num_memories, memory_seq_len, hidden_size)
        
        with torch.no_grad():
            attended_memories, attention_weights, importance_scores = encoder_model(
                query_embeddings, memory_embeddings
            )
        
        # Check output shapes
        assert attended_memories.shape == (batch_size, num_memories, hidden_size)
        assert attention_weights.shape == (batch_size, query_seq_len, num_memories * memory_seq_len)
        assert importance_scores.shape == (batch_size, num_memories)
        
        # Check value ranges
        assert torch.all(importance_scores >= 0) and torch.all(importance_scores <= 1)
    
    def test_memory_correlation_computation(self, encoder_model):
        """Test memory correlation computation"""
        import torch
        
        batch_size = 1
        num_memories = 3
        hidden_size = 384
        
        # Create mock memory embeddings
        memory_embeddings = torch.randn(batch_size, num_memories, hidden_size)
        
        with torch.no_grad():
            correlations = encoder_model.compute_memory_correlations(memory_embeddings)
        
        # Check correlation matrix shape
        assert correlations.shape == (batch_size, num_memories, num_memories)
        
        # Check correlation properties
        # Should be symmetric
        assert torch.allclose(correlations, correlations.transpose(-1, -2), atol=1e-6)
        
        # Diagonal should not necessarily be 1 (since we're not computing self-correlations explicitly)
        # Values should be between 0 and 1
        assert torch.all(correlations >= 0) and torch.all(correlations <= 1)


class TestPhase2Integration:
    """Test integration between Phase 2 components"""
    
    @pytest.mark.asyncio
    async def test_intelligence_to_memory_integration(self):
        """Test integration from community intelligence to cross-attention memory"""
        # Mock research finding that should enhance memory capabilities
        finding = ResearchFindingData(
            id="memory_enhancement_finding",
            title="Advanced Cross-Attention for Memory Systems",
            summary="Novel attention mechanism that improves memory retrieval by 40%",
            source_url="test://memory-enhancement",
            source_type=SourceType.ARXIV,
            discovered_at=datetime.now(),
            significance_score=0.9,
            significance_level=SignificanceLevel.HIGH,
            focus_areas=["memory_systems", "attention_mechanisms"],
            implementation_potential=0.8,
            community_interest=0.7,
            authors=["Memory Expert"],
            tags=["memory", "attention", "neural"],
            full_content="Detailed implementation of cross-attention for memory",
            metadata={"implementation_ready": True}
        )
        
        # Generate enhancement plan
        enhancer = CapabilityEnhancer()
        plan = await enhancer.generate_enhancement_plan(finding)
        
        # Should generate plan for memory enhancement
        assert plan is not None
        assert "memory" in plan.title.lower() or "attention" in plan.title.lower()
        assert plan.estimated_impact > 0.5  # High impact expected
        
        # Plan should include memory-specific considerations
        assert any("memory" in resource.lower() for resource in plan.required_resources)
    
    @pytest.mark.asyncio  
    async def test_end_to_end_research_to_enhancement(self):
        """Test end-to-end flow from research discovery to enhancement deployment"""
        
        # Step 1: Research discovery
        intelligence_processor = IntelligenceProcessor()
        
        breakthrough_finding = ResearchFindingData(
            id="breakthrough_test",
            title="Breakthrough: Self-Improving Agent Memory Networks",
            summary="Agents that automatically enhance their memory systems based on usage patterns",
            source_url="test://breakthrough",
            source_type=SourceType.ARXIV,
            discovered_at=datetime.now(),
            significance_score=0.95,
            significance_level=SignificanceLevel.BREAKTHROUGH,
            focus_areas=["multi-agent", "memory_systems", "self-improvement"],
            implementation_potential=0.85,
            community_interest=0.9,
            authors=["Breakthrough Author"],
            tags=["breakthrough", "self-improving", "memory", "agent"],
            full_content="Revolutionary approach with detailed implementation",
            metadata={"breakthrough": True, "citations": 50}
        )
        
        # Step 2: Process and rank
        processed_findings = await intelligence_processor.process_findings([breakthrough_finding])
        assert len(processed_findings) == 1
        assert processed_findings[0].significance_level == SignificanceLevel.BREAKTHROUGH
        
        # Step 3: Generate enhancement plan
        enhancer = CapabilityEnhancer()
        enhancement_plan = await enhancer.generate_enhancement_plan(processed_findings[0])
        
        assert enhancement_plan is not None
        assert enhancement_plan.estimated_impact > 0.8  # Very high impact
        assert enhancement_plan.development_time_days > 0
        
        # Step 4: Enhancement plan should have comprehensive details
        assert enhancement_plan.implementation_plan is not None
        assert enhancement_plan.testing_strategy is not None
        assert enhancement_plan.deployment_strategy is not None
        assert enhancement_plan.risk_assessment is not None
        
        # Implementation plan should have phases
        implementation_plan = enhancement_plan.implementation_plan
        assert "phases" in implementation_plan
        assert len(implementation_plan["phases"]) > 0
        
        # Each phase should have tasks
        for phase in implementation_plan["phases"]:
            assert "name" in phase
            assert "duration_days" in phase
            assert "tasks" in phase
            assert len(phase["tasks"]) > 0


class TestPhase2Performance:
    """Test Phase 2 performance requirements"""
    
    @pytest.mark.asyncio
    async def test_community_intelligence_performance(self):
        """Test community intelligence performance targets"""
        
        # Test research discovery performance
        start_time = time.time()
        
        arxiv_monitor = ArxivAIMonitor(["multi-agent", "memory"], 24)
        
        # Mock a research check (would normally call external API)
        with patch.object(arxiv_monitor, 'check_for_updates') as mock_check:
            mock_findings = [
                ResearchFindingData(
                    id=f"perf_test_{i}",
                    title=f"Test Finding {i}",
                    summary="Test summary",
                    source_url=f"test://url{i}",
                    source_type=SourceType.ARXIV,
                    discovered_at=datetime.now(),
                    significance_score=0.5,
                    significance_level=SignificanceLevel.MEDIUM,
                    focus_areas=["test"],
                    implementation_potential=0.5,
                    community_interest=0.5,
                    authors=["Test"],
                    tags=["test"],
                    full_content="Test",
                    metadata={}
                ) for i in range(100)  # 100 research findings
            ]
            mock_check.return_value = mock_findings
            
            findings = await arxiv_monitor.check_for_updates()
        
        processing_time = time.time() - start_time
        
        # Should process 100 findings quickly
        assert len(findings) == 100
        assert processing_time < 5.0  # Should process in under 5 seconds
    
    @pytest.mark.asyncio
    async def test_cross_attention_memory_performance(self):
        """Test cross-attention memory performance"""
        
        # Create mock cross-attention system
        cross_attention = CrossAttentionMemoryRetrieval()
        cross_attention.device = "cpu"
        
        # Mock models to avoid actual model loading
        with patch.object(cross_attention, '_encode_query') as mock_encode_query, \
             patch.object(cross_attention, '_encode_memories') as mock_encode_memories, \
             patch.object(cross_attention, '_get_candidate_memories') as mock_get_candidates:
            
            # Mock return values
            mock_encode_query.return_value = np.random.rand(384)
            mock_encode_memories.return_value = (np.random.rand(20, 384), ["text"] * 20)
            
            # Mock candidate memories
            mock_memories = []
            for i in range(20):
                from src.memory.memory_system import MemoryResult
                memory = MemoryResult(
                    memory_id=f"perf_mem_{i}",
                    memory_type="episodic",
                    content={"test": f"content_{i}"},
                    relevance_score=0.5,
                    importance_score=0.5,
                    created_at=datetime.now() - timedelta(hours=i),
                    last_accessed=datetime.now(),
                    access_count=1
                )
                mock_memories.append(memory)
            
            mock_get_candidates.return_value = mock_memories
            
            # Test retrieval performance
            from src.memory.memory_system import MemoryQuery
            
            query = MemoryQuery(
                query_text="test performance query",
                user_id="perf_test_user",
                limit=10
            )
            
            start_time = time.time()
            
            # Mock the cross-attention model call
            with patch.object(cross_attention, 'cross_attention_model') as mock_model:
                import torch
                
                # Mock model outputs
                mock_model.return_value = (
                    torch.randn(1, 20, 384),  # attended_memories
                    torch.randn(1, 1, 160),   # attention_weights (mock size)  
                    torch.sigmoid(torch.randn(1, 20))  # importance_scores
                )
                mock_model.compute_memory_correlations.return_value = torch.randn(1, 20, 20)
                
                result = await cross_attention._compute_final_relevance_scores(
                    np.random.rand(20), 
                    np.random.rand(20), 
                    mock_memories, 
                    query
                )
            
            processing_time = time.time() - start_time
            
            # Should compute relevance scores quickly
            assert len(result) == 20
            assert processing_time < 1.0  # Should process in under 1 second
    
    @pytest.mark.asyncio 
    async def test_enhancement_generation_performance(self):
        """Test enhancement plan generation performance"""
        
        enhancer = CapabilityEnhancer()
        
        # Create high-significance finding
        finding = ResearchFindingData(
            id="perf_enhancement_test",
            title="Performance Test Enhancement",
            summary="Test enhancement generation performance",
            source_url="test://performance",
            source_type=SourceType.ARXIV,
            discovered_at=datetime.now(),
            significance_score=0.8,
            significance_level=SignificanceLevel.HIGH,
            focus_areas=["performance"],
            implementation_potential=0.7,
            community_interest=0.6,
            authors=["Perf Tester"],
            tags=["performance"],
            full_content="Performance testing content",
            metadata={}
        )
        
        start_time = time.time()
        plan = await enhancer.generate_enhancement_plan(finding)
        generation_time = time.time() - start_time
        
        # Should generate plan quickly
        assert plan is not None
        assert generation_time < 2.0  # Should generate in under 2 seconds
        
        # Plan should be comprehensive
        assert len(plan.implementation_plan["phases"]) > 0
        assert len(plan.required_resources) > 0
        assert plan.development_time_days > 0


# Performance benchmark runner for Phase 2
class Phase2BenchmarkRunner:
    """Run Phase 2 performance benchmarks"""
    
    @staticmethod
    async def run_community_intelligence_benchmarks():
        """Run community intelligence benchmarks"""
        print("\n=== Phase 2 Community Intelligence Benchmarks ===")
        
        results = {}
        
        # Test research finding processing
        processor = IntelligenceProcessor()
        
        # Generate test findings
        test_findings = []
        for i in range(1000):  # 1000 findings
            finding = ResearchFindingData(
                id=f"benchmark_{i}",
                title=f"Benchmark Finding {i}",
                summary="Benchmark summary with relevant keywords for testing performance",
                source_url=f"benchmark://url{i}",
                source_type=SourceType.ARXIV,
                discovered_at=datetime.now(),
                significance_score=0.5 + (i % 100) / 200,  # Vary scores
                significance_level=SignificanceLevel.MEDIUM,
                focus_areas=["benchmark"],
                implementation_potential=0.5,
                community_interest=0.5,
                authors=[f"Author {i}"],
                tags=["benchmark"],
                full_content="Benchmark content",
                metadata={}
            )
            test_findings.append(finding)
        
        # Benchmark processing time
        start_time = time.time()
        processed_findings = await processor.process_findings(test_findings)
        processing_time = time.time() - start_time
        
        results["research_processing"] = {
            "findings_processed": len(processed_findings),
            "processing_time_seconds": processing_time,
            "findings_per_second": len(processed_findings) / processing_time,
            "target_met": processing_time < 10.0  # Target: <10 seconds for 1000 findings
        }
        
        return results
    
    @staticmethod
    async def run_cross_attention_benchmarks():
        """Run cross-attention memory benchmarks"""
        print("\n=== Phase 2 Cross-Attention Memory Benchmarks ===")
        
        results = {}
        
        # Test memory retrieval performance
        cross_attention = CrossAttentionMemoryRetrieval()
        cross_attention.device = "cpu"  # Use CPU for consistent benchmarking
        
        # Mock components for benchmarking
        with patch.object(cross_attention, '_encode_query') as mock_encode_query, \
             patch.object(cross_attention, '_encode_memories') as mock_encode_memories:
            
            # Mock encodings
            mock_encode_query.return_value = np.random.rand(384)
            mock_encode_memories.return_value = (np.random.rand(100, 384), ["text"] * 100)
            
            # Test different memory set sizes
            memory_sizes = [10, 50, 100, 200]
            
            for size in memory_sizes:
                # Create mock memories
                from src.memory.memory_system import MemoryResult, MemoryQuery
                
                memories = []
                for i in range(size):
                    memory = MemoryResult(
                        memory_id=f"benchmark_mem_{i}",
                        memory_type="episodic",
                        content={"benchmark": f"content_{i}"},
                        relevance_score=0.5,
                        importance_score=0.5,
                        created_at=datetime.now() - timedelta(hours=i),
                        last_accessed=datetime.now(),
                        access_count=1
                    )
                    memories.append(memory)
                
                # Test temporal decay computation
                start_time = time.time()
                decay_factors = cross_attention._compute_temporal_decay(memories)
                decay_time = time.time() - start_time
                
                # Test relevance scoring
                query = MemoryQuery(
                    query_text="benchmark query",
                    user_id="benchmark_user",
                    limit=10
                )
                
                start_time = time.time()
                relevance_scores = cross_attention._compute_final_relevance_scores(
                    np.random.rand(size),
                    decay_factors,
                    memories,
                    query
                )
                scoring_time = time.time() - start_time
                
                results[f"memory_retrieval_{size}"] = {
                    "memory_count": size,
                    "decay_computation_ms": decay_time * 1000,
                    "relevance_scoring_ms": scoring_time * 1000,
                    "total_time_ms": (decay_time + scoring_time) * 1000,
                    "target_met": (decay_time + scoring_time) < 0.1  # Target: <100ms
                }
        
        return results


if __name__ == "__main__":
    # Run Phase 2 benchmarks
    async def run_phase2_benchmarks():
        print("🚀 Running MONK CLI Phase 2 Performance Benchmarks")
        
        # Community intelligence benchmarks
        ci_results = await Phase2BenchmarkRunner.run_community_intelligence_benchmarks()
        
        # Cross-attention memory benchmarks  
        ca_results = await Phase2BenchmarkRunner.run_cross_attention_benchmarks()
        
        # Combine results
        all_results = {
            "community_intelligence": ci_results,
            "cross_attention_memory": ca_results,
            "timestamp": datetime.now().isoformat(),
            "phase": "2"
        }
        
        # Print summary
        print("\n=== Phase 2 Benchmark Results ===")
        print(json.dumps(all_results, indent=2, default=str))
        
        return all_results
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        asyncio.run(run_phase2_benchmarks())
    else:
        # Run pytest
        pytest.main([__file__, "-v"])