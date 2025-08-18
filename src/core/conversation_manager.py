"""
Conversation Manager
Maintains conversation context and memory across sessions for Monk CLI
"""

import json
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib


@dataclass
class Message:
    """Represents a single message in conversation history"""
    timestamp: float
    role: str  # 'user', 'assistant', 'system'
    content: str
    metadata: Dict[str, Any]
    session_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ProjectContext:
    """Represents the current project context"""
    project_path: str
    project_type: str
    git_status: Dict[str, Any]
    relevant_files: List[str]
    dependencies: List[str]
    last_updated: float
    summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectContext':
        """Create from dictionary"""
        return cls(**data)


class ConversationManager:
    """
    Manages conversation history, context, and session persistence
    """
    
    def __init__(self, session_dir: Optional[Path] = None):
        self.session_dir = session_dir or Path.home() / ".monk-sessions"
        self.session_dir.mkdir(exist_ok=True)
        
        # Current session state
        self.current_session_id = self._generate_session_id()
        self.conversation_history: List[Message] = []
        self.project_context: Optional[ProjectContext] = None
        self.context_window = 10  # Number of messages to keep in active context
        
        # Session persistence
        self.session_file = self.session_dir / f"{self.current_session_id}.json"
        self.global_history_file = self.session_dir / "global_history.json"
        
        # Load existing session if available
        self._load_session_state()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = str(time.time())
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a message to conversation history"""
        message = Message(
            timestamp=time.time(),
            role=role,
            content=content,
            metadata=metadata or {},
            session_id=self.current_session_id
        )
        
        self.conversation_history.append(message)
        
        # Persist immediately for durability
        self._save_session_state()
        
        return message
    
    def get_context_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get recent messages for context (within context window)"""
        limit = limit or self.context_window
        return self.conversation_history[-limit:] if self.conversation_history else []
    
    def get_conversation_summary(self) -> str:
        """Generate a summary of the current conversation"""
        if not self.conversation_history:
            return "No conversation history"
        
        recent_messages = self.get_context_messages()
        
        summary_parts = []
        if self.project_context:
            summary_parts.append(f"Project: {self.project_context.project_type} at {self.project_context.project_path}")
        
        summary_parts.append(f"Conversation: {len(recent_messages)} recent messages")
        
        # Add key topics from recent messages
        user_messages = [msg.content for msg in recent_messages if msg.role == 'user']
        if user_messages:
            summary_parts.append(f"Recent topics: {', '.join(user_messages[-3:])}")
        
        return " | ".join(summary_parts)
    
    def set_project_context(self, context: ProjectContext):
        """Set the current project context"""
        self.project_context = context
        self._save_session_state()
    
    def get_project_context(self) -> Optional[ProjectContext]:
        """Get current project context"""
        return self.project_context
    
    def clear_conversation(self):
        """Clear current conversation but keep project context"""
        self.conversation_history = []
        self._save_session_state()
    
    def start_new_session(self) -> str:
        """Start a new conversation session"""
        # Save current session
        self._save_session_state()
        
        # Start new session
        old_session_id = self.current_session_id
        self.current_session_id = self._generate_session_id()
        self.conversation_history = []
        self.session_file = self.session_dir / f"{self.current_session_id}.json"
        
        return old_session_id
    
    def load_session(self, session_id: str) -> bool:
        """Load a specific session by ID"""
        session_file = self.session_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return False
        
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            self.current_session_id = session_id
            self.session_file = session_file
            
            # Load conversation history
            self.conversation_history = [
                Message.from_dict(msg_data) 
                for msg_data in data.get('conversation_history', [])
            ]
            
            # Load project context
            if 'project_context' in data and data['project_context']:
                self.project_context = ProjectContext.from_dict(data['project_context'])
            
            return True
            
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List available sessions with metadata"""
        sessions = []
        
        for session_file in self.session_dir.glob("*.json"):
            if session_file.name == "global_history.json":
                continue
            
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                session_info = {
                    'session_id': session_file.stem,
                    'created': data.get('created', 0),
                    'last_updated': data.get('last_updated', 0),
                    'message_count': len(data.get('conversation_history', [])),
                    'project_path': data.get('project_context', {}).get('project_path', 'Unknown'),
                    'project_type': data.get('project_context', {}).get('project_type', 'Unknown')
                }
                
                sessions.append(session_info)
                
            except Exception as e:
                print(f"Error reading session {session_file}: {e}")
        
        # Sort by last updated
        sessions.sort(key=lambda x: x['last_updated'], reverse=True)
        return sessions
    
    def search_history(self, query: str, limit: int = 10) -> List[Message]:
        """Search conversation history for messages containing query"""
        query_lower = query.lower()
        matches = []
        
        for message in self.conversation_history:
            if query_lower in message.content.lower():
                matches.append(message)
        
        return matches[-limit:] if matches else []
    
    def get_context_for_ai(self) -> Dict[str, Any]:
        """Get formatted context for AI providers"""
        context = {
            'session_id': self.current_session_id,
            'timestamp': time.time(),
            'conversation_summary': self.get_conversation_summary(),
            'recent_messages': [msg.to_dict() for msg in self.get_context_messages()],
        }
        
        if self.project_context:
            context['project'] = self.project_context.to_dict()
        
        return context
    
    def _save_session_state(self):
        """Save current session state to disk"""
        try:
            session_data = {
                'session_id': self.current_session_id,
                'created': time.time() if not self.session_file.exists() else None,
                'last_updated': time.time(),
                'conversation_history': [msg.to_dict() for msg in self.conversation_history],
                'project_context': self.project_context.to_dict() if self.project_context else None
            }
            
            # Update created time if loading existing session
            if self.session_file.exists():
                try:
                    with open(self.session_file, 'r') as f:
                        existing_data = json.load(f)
                        session_data['created'] = existing_data.get('created', time.time())
                except:
                    session_data['created'] = time.time()
            else:
                session_data['created'] = time.time()
            
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save session state: {e}")
    
    def _load_session_state(self):
        """Load session state from disk if available"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                
                # Load conversation history
                self.conversation_history = [
                    Message.from_dict(msg_data) 
                    for msg_data in data.get('conversation_history', [])
                ]
                
                # Load project context
                if 'project_context' in data and data['project_context']:
                    self.project_context = ProjectContext.from_dict(data['project_context'])
                    
            except Exception as e:
                print(f"Warning: Failed to load session state: {e}")
                # Continue with empty state
    
    def export_conversation(self, format: str = 'json') -> str:
        """Export conversation to various formats"""
        if format == 'json':
            return json.dumps({
                'session_id': self.current_session_id,
                'conversation_history': [msg.to_dict() for msg in self.conversation_history],
                'project_context': self.project_context.to_dict() if self.project_context else None,
                'exported_at': time.time()
            }, indent=2)
        
        elif format == 'markdown':
            lines = [f"# Conversation Export - {self.current_session_id}"]
            lines.append(f"Exported: {datetime.now().isoformat()}")
            
            if self.project_context:
                lines.append(f"\n## Project Context")
                lines.append(f"- **Path**: {self.project_context.project_path}")
                lines.append(f"- **Type**: {self.project_context.project_type}")
                lines.append(f"- **Summary**: {self.project_context.summary}")
            
            lines.append(f"\n## Conversation History")
            
            for msg in self.conversation_history:
                timestamp = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M:%S")
                lines.append(f"\n### {msg.role.title()} [{timestamp}]")
                lines.append(msg.content)
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global conversation manager instance
conversation_manager = ConversationManager()


# Example usage and testing
if __name__ == "__main__":
    # Test conversation manager
    cm = ConversationManager()
    
    # Add some test messages
    cm.add_message("user", "Hello, I need help with my Python project")
    cm.add_message("assistant", "I'd be happy to help! What specific issues are you facing?")
    cm.add_message("user", "/analyze src/")
    
    # Test project context
    context = ProjectContext(
        project_path="/home/user/my-project",
        project_type="Python",
        git_status={"branch": "main", "clean": True},
        relevant_files=["main.py", "requirements.txt"],
        dependencies=["flask", "requests"],
        last_updated=time.time(),
        summary="Flask web application with REST API"
    )
    cm.set_project_context(context)
    
    # Test context retrieval
    ai_context = cm.get_context_for_ai()
    print("AI Context:")
    print(json.dumps(ai_context, indent=2))
    
    # Test export
    markdown_export = cm.export_conversation('markdown')
    print("\nMarkdown Export:")
    print(markdown_export)