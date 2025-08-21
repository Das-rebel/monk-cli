"""
MONK CLI Phase 2.5 - Collaborative Development Platform Integration
Open source collaborative development platform with real-time collaboration
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import hashlib
import websockets
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading

from ...core.config import config
from ...memory.memory_system import MemorySystem, MemoryQuery
from ..bridges.treequest_smolagent_bridge import TreeQuestTask, TreeQuestSmolagentBridge
from ..smolagents.multi_agent_system import MONKMultiAgentSystem, AgentTask
from ..lsp.tree_sitter_explorer import TreeSitterLSPExplorer, CodeSymbol, CodeTree
from ...core.database import get_db_session

logger = logging.getLogger(__name__)


class CollaborationEventType(Enum):
    """Types of collaboration events"""
    CODE_CHANGE = "code_change"
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    CURSOR_MOVE = "cursor_move"
    SELECTION_CHANGE = "selection_change"
    COMMENT_ADD = "comment_add"
    COMMENT_RESOLVE = "comment_resolve"
    TASK_ASSIGN = "task_assign"
    TASK_COMPLETE = "task_complete"
    AGENT_SUGGESTION = "agent_suggestion"
    FILE_LOCK = "file_lock"
    FILE_UNLOCK = "file_unlock"
    BREAKPOINT_SET = "breakpoint_set"
    BREAKPOINT_REMOVE = "breakpoint_remove"


class UserRole(Enum):
    """User roles in collaborative session"""
    OWNER = "owner"
    MAINTAINER = "maintainer"
    CONTRIBUTOR = "contributor"
    REVIEWER = "reviewer"
    OBSERVER = "observer"


@dataclass
class CollaborationUser:
    """User in collaborative session"""
    user_id: str
    name: str
    email: str
    role: UserRole
    avatar_url: Optional[str] = None
    status: str = "active"  # active, idle, away, offline
    current_file: Optional[str] = None
    cursor_position: Optional[Dict[str, int]] = None
    selection: Optional[Dict[str, Any]] = None
    joined_at: datetime = None
    last_activity: datetime = None
    
    def __post_init__(self):
        if self.joined_at is None:
            self.joined_at = datetime.now()
        if self.last_activity is None:
            self.last_activity = datetime.now()


@dataclass
class CollaborationEvent:
    """Collaboration event data"""
    event_id: str
    event_type: CollaborationEventType
    user_id: str
    session_id: str
    data: Dict[str, Any]
    timestamp: datetime
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CodeComment:
    """Code comment in collaborative session"""
    comment_id: str
    user_id: str
    file_path: str
    line_number: int
    content: str
    thread_id: Optional[str] = None
    resolved: bool = False
    created_at: datetime = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    replies: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.replies is None:
            self.replies = []


@dataclass
class CollaborationSession:
    """Collaborative development session"""
    session_id: str
    name: str
    description: str
    workspace_path: str
    owner_id: str
    users: Dict[str, CollaborationUser]
    active_files: Set[str]
    file_locks: Dict[str, str]  # file_path -> user_id
    comments: Dict[str, List[CodeComment]]  # file_path -> comments
    shared_terminals: Dict[str, Dict[str, Any]]
    agent_suggestions: List[Dict[str, Any]]
    created_at: datetime
    last_activity: datetime
    settings: Dict[str, Any]
    
    def __post_init__(self):
        if not hasattr(self, 'users') or self.users is None:
            self.users = {}
        if not hasattr(self, 'active_files') or self.active_files is None:
            self.active_files = set()
        if not hasattr(self, 'file_locks') or self.file_locks is None:
            self.file_locks = {}
        if not hasattr(self, 'comments') or self.comments is None:
            self.comments = {}
        if not hasattr(self, 'shared_terminals') or self.shared_terminals is None:
            self.shared_terminals = {}
        if not hasattr(self, 'agent_suggestions') or self.agent_suggestions is None:
            self.agent_suggestions = []


class CollaborativePlatform:
    """Open source collaborative development platform"""
    
    def __init__(self):
        # Core systems integration
        self.memory_system = MemorySystem()
        self.treequest_bridge = TreeQuestSmolagentBridge()
        self.multi_agent_system = MONKMultiAgentSystem()
        self.tree_sitter_explorer = TreeSitterLSPExplorer()
        
        # Collaboration state
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.event_history: List[CollaborationEvent] = []
        
        # Real-time communication
        self.websocket_server = None
        self.websocket_port = 8001
        self.http_server_port = 8002
        
        # Background services
        self.event_processor = None
        self.session_monitor = None
        self.auto_save_service = None
        
        # Performance metrics
        self.metrics = {
            "active_sessions": 0,
            "total_users": 0,
            "events_processed": 0,
            "files_collaborated": 0,
            "agent_interactions": 0,
            "average_response_time": 0.0
        }
        
    async def initialize(self):
        """Initialize collaborative platform"""
        try:
            logger.info("Initializing Collaborative Development Platform")
            
            # Initialize core systems
            await self.memory_system.initialize()
            await self.treequest_bridge.initialize()
            await self.multi_agent_system.initialize()
            await self.tree_sitter_explorer.initialize()
            
            # Start WebSocket server for real-time communication
            await self.start_websocket_server()
            
            # Start HTTP API server
            await self.start_http_server()
            
            # Start background services
            await self.start_background_services()
            
            logger.info("Collaborative Platform initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize collaborative platform: {e}")
            return False
    
    async def start_websocket_server(self):
        """Start WebSocket server for real-time collaboration"""
        try:
            self.websocket_server = await websockets.serve(
                self.handle_websocket_connection,
                "localhost",
                self.websocket_port,
                ping_interval=20,
                ping_timeout=10
            )
            
            logger.info(f"WebSocket server started on ws://localhost:{self.websocket_port}")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise
    
    async def handle_websocket_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        connection_id = f"conn_{datetime.now().timestamp()}_{id(websocket)}"
        self.websocket_connections[connection_id] = websocket
        
        logger.info(f"New WebSocket connection: {connection_id}")
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat()
            }))
            
            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_websocket_message(connection_id, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {connection_id}: {message}")
                except Exception as e:
                    logger.error(f"Error handling message from {connection_id}: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket connection {connection_id}: {e}")
        finally:
            # Clean up connection
            if connection_id in self.websocket_connections:
                del self.websocket_connections[connection_id]
    
    async def handle_websocket_message(self, connection_id: str, data: Dict[str, Any]):
        """Handle WebSocket message"""
        message_type = data.get("type")
        
        if message_type == "join_session":
            await self.handle_join_session(connection_id, data)
        elif message_type == "leave_session":
            await self.handle_leave_session(connection_id, data)
        elif message_type == "code_change":
            await self.handle_code_change(connection_id, data)
        elif message_type == "cursor_move":
            await self.handle_cursor_move(connection_id, data)
        elif message_type == "file_lock":
            await self.handle_file_lock(connection_id, data)
        elif message_type == "add_comment":
            await self.handle_add_comment(connection_id, data)
        elif message_type == "request_agent_help":
            await self.handle_agent_help_request(connection_id, data)
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    async def handle_join_session(self, connection_id: str, data: Dict[str, Any]):
        """Handle user joining collaboration session"""
        session_id = data.get("session_id")
        user_info = data.get("user")
        
        if not session_id or not user_info:
            await self.send_error(connection_id, "Missing session_id or user info")
            return
        
        # Get or create session
        session = await self.get_or_create_session(session_id, data.get("session_config", {}))
        
        # Create user object
        user = CollaborationUser(
            user_id=user_info["user_id"],
            name=user_info["name"],
            email=user_info["email"],
            role=UserRole(user_info.get("role", "contributor")),
            avatar_url=user_info.get("avatar_url")
        )
        
        # Add user to session
        session.users[user.user_id] = user
        self.user_sessions[user.user_id] = session_id
        
        # Create join event
        event = CollaborationEvent(
            event_id=self.generate_event_id(),
            event_type=CollaborationEventType.USER_JOIN,
            user_id=user.user_id,
            session_id=session_id,
            data={"user": asdict(user)},
            timestamp=datetime.now()
        )
        
        await self.process_event(event)
        
        # Send session state to new user
        await self.send_session_state(connection_id, session)
        
        # Broadcast user join to other users
        await self.broadcast_to_session(session_id, {
            "type": "user_joined",
            "user": asdict(user),
            "timestamp": datetime.now().isoformat()
        }, exclude_connection=connection_id)
        
        logger.info(f"User {user.name} joined session {session_id}")
    
    async def handle_leave_session(self, connection_id: str, data: Dict[str, Any]):
        """Handle user leaving collaboration session"""
        user_id = data.get("user_id")
        
        if user_id in self.user_sessions:
            session_id = self.user_sessions[user_id]
            session = self.sessions.get(session_id)
            
            if session and user_id in session.users:
                user = session.users[user_id]
                
                # Remove user locks
                files_to_unlock = [f for f, u in session.file_locks.items() if u == user_id]
                for file_path in files_to_unlock:
                    del session.file_locks[file_path]
                
                # Remove user from session
                del session.users[user_id]
                del self.user_sessions[user_id]
                
                # Create leave event
                event = CollaborationEvent(
                    event_id=self.generate_event_id(),
                    event_type=CollaborationEventType.USER_LEAVE,
                    user_id=user_id,
                    session_id=session_id,
                    data={"files_unlocked": files_to_unlock},
                    timestamp=datetime.now()
                )
                
                await self.process_event(event)
                
                # Broadcast user leave
                await self.broadcast_to_session(session_id, {
                    "type": "user_left",
                    "user_id": user_id,
                    "files_unlocked": files_to_unlock,
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"User {user.name} left session {session_id}")
    
    async def handle_code_change(self, connection_id: str, data: Dict[str, Any]):
        """Handle code change event"""
        session_id = data.get("session_id")
        user_id = data.get("user_id")
        file_path = data.get("file_path")
        changes = data.get("changes")
        
        if not all([session_id, user_id, file_path, changes]):
            await self.send_error(connection_id, "Missing required fields for code change")
            return
        
        session = self.sessions.get(session_id)
        if not session:
            await self.send_error(connection_id, "Session not found")
            return
        
        # Check if user has permission to edit
        if not await self.can_user_edit_file(user_id, file_path, session):
            await self.send_error(connection_id, "Permission denied")
            return
        
        # Apply changes using tree-sitter for analysis
        try:
            # Parse the file to understand the changes
            await self.tree_sitter_explorer.parse_file(file_path, force_reparse=True)
            
            # Create change event
            event = CollaborationEvent(
                event_id=self.generate_event_id(),
                event_type=CollaborationEventType.CODE_CHANGE,
                user_id=user_id,
                session_id=session_id,
                data={
                    "file_path": file_path,
                    "changes": changes,
                    "content_hash": hashlib.md5(str(changes).encode()).hexdigest()
                },
                timestamp=datetime.now(),
                file_path=file_path
            )
            
            await self.process_event(event)
            
            # Get AI suggestions if enabled
            suggestions = await self.get_ai_suggestions_for_changes(file_path, changes, session)
            
            # Broadcast changes to other users
            await self.broadcast_to_session(session_id, {
                "type": "code_changed",
                "user_id": user_id,
                "file_path": file_path,
                "changes": changes,
                "suggestions": suggestions,
                "timestamp": datetime.now().isoformat()
            }, exclude_connection=connection_id)
            
        except Exception as e:
            logger.error(f"Error handling code change: {e}")
            await self.send_error(connection_id, f"Error processing code change: {e}")
    
    async def handle_cursor_move(self, connection_id: str, data: Dict[str, Any]):
        """Handle cursor movement event"""
        session_id = data.get("session_id")
        user_id = data.get("user_id")
        file_path = data.get("file_path")
        position = data.get("position")
        
        session = self.sessions.get(session_id)
        if session and user_id in session.users:
            user = session.users[user_id]
            user.current_file = file_path
            user.cursor_position = position
            user.last_activity = datetime.now()
            
            # Broadcast cursor movement (throttled)
            await self.broadcast_to_session(session_id, {
                "type": "cursor_moved",
                "user_id": user_id,
                "file_path": file_path,
                "position": position,
                "timestamp": datetime.now().isoformat()
            }, exclude_connection=connection_id, throttle_ms=100)
    
    async def handle_file_lock(self, connection_id: str, data: Dict[str, Any]):
        """Handle file locking/unlocking"""
        session_id = data.get("session_id")
        user_id = data.get("user_id")
        file_path = data.get("file_path")
        action = data.get("action")  # "lock" or "unlock"
        
        session = self.sessions.get(session_id)
        if not session:
            await self.send_error(connection_id, "Session not found")
            return
        
        if action == "lock":
            if file_path in session.file_locks:
                await self.send_error(connection_id, f"File already locked by {session.file_locks[file_path]}")
                return
            
            session.file_locks[file_path] = user_id
            event_type = CollaborationEventType.FILE_LOCK
            
        elif action == "unlock":
            if file_path not in session.file_locks or session.file_locks[file_path] != user_id:
                await self.send_error(connection_id, "Cannot unlock file not locked by you")
                return
            
            del session.file_locks[file_path]
            event_type = CollaborationEventType.FILE_UNLOCK
        
        else:
            await self.send_error(connection_id, "Invalid action")
            return
        
        # Create event
        event = CollaborationEvent(
            event_id=self.generate_event_id(),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            data={"file_path": file_path, "action": action},
            timestamp=datetime.now(),
            file_path=file_path
        )
        
        await self.process_event(event)
        
        # Broadcast file lock status
        await self.broadcast_to_session(session_id, {
            "type": "file_lock_changed",
            "file_path": file_path,
            "locked_by": user_id if action == "lock" else None,
            "timestamp": datetime.now().isoformat()
        })
    
    async def handle_add_comment(self, connection_id: str, data: Dict[str, Any]):
        """Handle adding code comment"""
        session_id = data.get("session_id")
        user_id = data.get("user_id")
        file_path = data.get("file_path")
        line_number = data.get("line_number")
        content = data.get("content")
        thread_id = data.get("thread_id")
        
        session = self.sessions.get(session_id)
        if not session:
            await self.send_error(connection_id, "Session not found")
            return
        
        comment = CodeComment(
            comment_id=self.generate_comment_id(),
            user_id=user_id,
            file_path=file_path,
            line_number=line_number,
            content=content,
            thread_id=thread_id
        )
        
        # Add comment to session
        if file_path not in session.comments:
            session.comments[file_path] = []
        session.comments[file_path].append(comment)
        
        # Create event
        event = CollaborationEvent(
            event_id=self.generate_event_id(),
            event_type=CollaborationEventType.COMMENT_ADD,
            user_id=user_id,
            session_id=session_id,
            data=asdict(comment),
            timestamp=datetime.now(),
            file_path=file_path,
            line_number=line_number
        )
        
        await self.process_event(event)
        
        # Broadcast comment
        await self.broadcast_to_session(session_id, {
            "type": "comment_added",
            "comment": asdict(comment),
            "timestamp": datetime.now().isoformat()
        })
    
    async def handle_agent_help_request(self, connection_id: str, data: Dict[str, Any]):
        """Handle request for AI agent help"""
        session_id = data.get("session_id")
        user_id = data.get("user_id")
        file_path = data.get("file_path")
        request_type = data.get("request_type")  # "explain", "suggest", "fix", etc.
        context = data.get("context", {})
        
        session = self.sessions.get(session_id)
        if not session:
            await self.send_error(connection_id, "Session not found")
            return
        
        try:
            # Submit task to multi-agent system
            task_description = f"{request_type} code in {file_path}"
            
            task_id = await self.multi_agent_system.submit_task(
                description=task_description,
                input_data={
                    "file_path": file_path,
                    "request_type": request_type,
                    "context": context,
                    "session_id": session_id,
                    "user_id": user_id
                }
            )
            
            # Wait for result (with timeout)
            results = await self.multi_agent_system.wait_for_completion([task_id], timeout=30)
            
            if task_id in results and results[task_id]["status"] == "completed":
                suggestion = {
                    "suggestion_id": self.generate_suggestion_id(),
                    "type": request_type,
                    "file_path": file_path,
                    "content": results[task_id]["result"]["result"],
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "agent_used": results[task_id]["result"].get("agent_used"),
                    "confidence": results[task_id]["result"].get("confidence_score", 0.8)
                }
                
                session.agent_suggestions.append(suggestion)
                
                # Broadcast agent suggestion
                await self.broadcast_to_session(session_id, {
                    "type": "agent_suggestion",
                    "suggestion": suggestion
                })
            else:
                await self.send_error(connection_id, "Agent request failed or timed out")
                
        except Exception as e:
            logger.error(f"Error handling agent help request: {e}")
            await self.send_error(connection_id, f"Error processing agent request: {e}")
    
    async def get_or_create_session(self, session_id: str, config: Dict[str, Any]) -> CollaborationSession:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            session = CollaborationSession(
                session_id=session_id,
                name=config.get("name", f"Session {session_id}"),
                description=config.get("description", "Collaborative development session"),
                workspace_path=config.get("workspace_path", "/tmp"),
                owner_id=config.get("owner_id", "unknown"),
                users={},
                active_files=set(),
                file_locks={},
                comments={},
                shared_terminals={},
                agent_suggestions=[],
                created_at=datetime.now(),
                last_activity=datetime.now(),
                settings=config.get("settings", {})
            )
            
            self.sessions[session_id] = session
            self.metrics["active_sessions"] = len(self.sessions)
            
            logger.info(f"Created new collaboration session: {session_id}")
        
        return self.sessions[session_id]
    
    async def can_user_edit_file(self, user_id: str, file_path: str, session: CollaborationSession) -> bool:
        """Check if user can edit file"""
        # Check if file is locked by someone else
        if file_path in session.file_locks and session.file_locks[file_path] != user_id:
            return False
        
        # Check user permissions
        if user_id not in session.users:
            return False
        
        user = session.users[user_id]
        if user.role in [UserRole.OWNER, UserRole.MAINTAINER, UserRole.CONTRIBUTOR]:
            return True
        
        return False
    
    async def get_ai_suggestions_for_changes(self, file_path: str, changes: List[Dict[str, Any]], 
                                           session: CollaborationSession) -> List[Dict[str, Any]]:
        """Get AI suggestions for code changes"""
        suggestions = []
        
        try:
            # Analyze changes using tree-sitter
            code_tree = await self.tree_sitter_explorer.parse_file(file_path)
            
            if code_tree:
                # Submit analysis task to multi-agent system
                task_id = await self.multi_agent_system.submit_task(
                    description="Analyze code changes and provide suggestions",
                    input_data={
                        "file_path": file_path,
                        "changes": changes,
                        "code_tree": asdict(code_tree),
                        "session_id": session.session_id
                    },
                    priority=2  # Lower priority than user requests
                )
                
                # Wait briefly for suggestions (non-blocking)
                results = await self.multi_agent_system.wait_for_completion([task_id], timeout=5)
                
                if task_id in results and results[task_id]["status"] == "completed":
                    result = results[task_id]["result"]
                    if "suggestions" in result:
                        suggestions.extend(result["suggestions"])
                
        except Exception as e:
            logger.error(f"Error getting AI suggestions: {e}")
        
        return suggestions
    
    async def process_event(self, event: CollaborationEvent):
        """Process collaboration event"""
        try:
            self.event_history.append(event)
            self.metrics["events_processed"] += 1
            
            # Store in memory system for future analysis
            await self.memory_system.store_memory(
                content=f"Collaboration event: {event.event_type.value}",
                memory_type="collaboration_event",
                metadata=asdict(event),
                context_tags=[
                    f"session:{event.session_id}",
                    f"user:{event.user_id}",
                    f"type:{event.event_type.value}"
                ]
            )
            
            # Update session activity
            if event.session_id in self.sessions:
                self.sessions[event.session_id].last_activity = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing event: {e}")
    
    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any], 
                                 exclude_connection: str = None, throttle_ms: int = None):
        """Broadcast message to all users in session"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        # Add throttling for high-frequency events
        if throttle_ms:
            # Simple throttling implementation
            pass  # In practice, implement proper throttling
        
        message_json = json.dumps(message)
        
        # Send to all connections for users in this session
        for user_id in session.users.keys():
            for conn_id, websocket in self.websocket_connections.items():
                if conn_id != exclude_connection:
                    try:
                        await websocket.send(message_json)
                    except websockets.exceptions.ConnectionClosed:
                        # Connection is closed, will be cleaned up
                        pass
                    except Exception as e:
                        logger.error(f"Error broadcasting to {conn_id}: {e}")
    
    async def send_session_state(self, connection_id: str, session: CollaborationSession):
        """Send current session state to connection"""
        websocket = self.websocket_connections.get(connection_id)
        if not websocket:
            return
        
        try:
            # Prepare session state
            session_state = {
                "type": "session_state",
                "session": {
                    "session_id": session.session_id,
                    "name": session.name,
                    "description": session.description,
                    "users": {uid: asdict(user) for uid, user in session.users.items()},
                    "active_files": list(session.active_files),
                    "file_locks": session.file_locks,
                    "comments": {
                        file_path: [asdict(comment) for comment in comments]
                        for file_path, comments in session.comments.items()
                    },
                    "recent_suggestions": session.agent_suggestions[-10:]  # Last 10 suggestions
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send(json.dumps(session_state))
            
        except Exception as e:
            logger.error(f"Error sending session state to {connection_id}: {e}")
    
    async def send_error(self, connection_id: str, error_message: str):
        """Send error message to connection"""
        websocket = self.websocket_connections.get(connection_id)
        if websocket:
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": error_message,
                    "timestamp": datetime.now().isoformat()
                }))
            except Exception as e:
                logger.error(f"Error sending error to {connection_id}: {e}")
    
    async def start_http_server(self):
        """Start HTTP API server for REST endpoints"""
        from aiohttp import web, web_runner
        
        app = web.Application()
        
        # API routes
        app.router.add_get('/api/sessions', self.api_get_sessions)
        app.router.add_post('/api/sessions', self.api_create_session)
        app.router.add_get('/api/sessions/{session_id}', self.api_get_session)
        app.router.add_delete('/api/sessions/{session_id}', self.api_delete_session)
        app.router.add_get('/api/sessions/{session_id}/events', self.api_get_session_events)
        app.router.add_get('/api/metrics', self.api_get_metrics)
        
        # CORS middleware
        app.middlewares.append(self.cors_middleware)
        
        runner = web_runner.AppRunner(app)
        await runner.setup()
        
        site = web_runner.TCPSite(runner, 'localhost', self.http_server_port)
        await site.start()
        
        logger.info(f"HTTP API server started on http://localhost:{self.http_server_port}")
    
    async def cors_middleware(self, request, handler):
        """CORS middleware for HTTP API"""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    async def api_get_sessions(self, request):
        """API: Get all collaboration sessions"""
        from aiohttp import web
        
        sessions = [
            {
                "session_id": session.session_id,
                "name": session.name,
                "description": session.description,
                "user_count": len(session.users),
                "active_files": len(session.active_files),
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat()
            }
            for session in self.sessions.values()
        ]
        
        return web.json_response({"sessions": sessions})
    
    async def api_create_session(self, request):
        """API: Create new collaboration session"""
        from aiohttp import web
        
        try:
            data = await request.json()
            session_id = data.get("session_id", f"session_{datetime.now().timestamp()}")
            
            session = await self.get_or_create_session(session_id, data)
            
            return web.json_response({
                "session_id": session.session_id,
                "websocket_url": f"ws://localhost:{self.websocket_port}",
                "created": True
            })
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
    
    async def api_get_session(self, request):
        """API: Get specific session details"""
        from aiohttp import web
        
        session_id = request.match_info['session_id']
        session = self.sessions.get(session_id)
        
        if not session:
            return web.json_response({"error": "Session not found"}, status=404)
        
        return web.json_response({
            "session_id": session.session_id,
            "name": session.name,
            "description": session.description,
            "users": [asdict(user) for user in session.users.values()],
            "active_files": list(session.active_files),
            "file_locks": session.file_locks,
            "comment_count": sum(len(comments) for comments in session.comments.values()),
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat()
        })
    
    async def api_delete_session(self, request):
        """API: Delete collaboration session"""
        from aiohttp import web
        
        session_id = request.match_info['session_id']
        
        if session_id in self.sessions:
            # Notify users and close connections
            await self.broadcast_to_session(session_id, {
                "type": "session_ended",
                "timestamp": datetime.now().isoformat()
            })
            
            del self.sessions[session_id]
            self.metrics["active_sessions"] = len(self.sessions)
            
            return web.json_response({"deleted": True})
        else:
            return web.json_response({"error": "Session not found"}, status=404)
    
    async def api_get_session_events(self, request):
        """API: Get events for session"""
        from aiohttp import web
        
        session_id = request.match_info['session_id']
        limit = int(request.query.get('limit', 100))
        
        session_events = [
            asdict(event) for event in self.event_history
            if event.session_id == session_id
        ][-limit:]
        
        return web.json_response({"events": session_events})
    
    async def api_get_metrics(self, request):
        """API: Get platform metrics"""
        from aiohttp import web
        
        # Update metrics
        self.metrics["total_users"] = sum(len(s.users) for s in self.sessions.values())
        self.metrics["files_collaborated"] = sum(len(s.active_files) for s in self.sessions.values())
        
        return web.json_response(self.metrics)
    
    async def start_background_services(self):
        """Start background services"""
        # Event processor
        self.event_processor = asyncio.create_task(self.event_processor_loop())
        
        # Session monitor
        self.session_monitor = asyncio.create_task(self.session_monitor_loop())
        
        # Auto-save service
        self.auto_save_service = asyncio.create_task(self.auto_save_loop())
        
        logger.info("Background services started")
    
    async def event_processor_loop(self):
        """Background event processing loop"""
        while True:
            try:
                # Process recent events for patterns, analytics, etc.
                # This is where you'd implement event analysis, notifications, etc.
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in event processor loop: {e}")
                await asyncio.sleep(10)
    
    async def session_monitor_loop(self):
        """Background session monitoring loop"""
        while True:
            try:
                current_time = datetime.now()
                
                # Clean up inactive sessions
                inactive_sessions = []
                for session_id, session in self.sessions.items():
                    if current_time - session.last_activity > timedelta(hours=24):
                        if not session.users:  # No active users
                            inactive_sessions.append(session_id)
                
                for session_id in inactive_sessions:
                    del self.sessions[session_id]
                    logger.info(f"Cleaned up inactive session: {session_id}")
                
                # Update metrics
                self.metrics["active_sessions"] = len(self.sessions)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in session monitor loop: {e}")
                await asyncio.sleep(60)
    
    async def auto_save_loop(self):
        """Background auto-save loop"""
        while True:
            try:
                # Auto-save session state, events, etc.
                # In practice, save to persistent storage
                await asyncio.sleep(300)  # Save every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in auto-save loop: {e}")
                await asyncio.sleep(300)
    
    def generate_event_id(self) -> str:
        """Generate unique event ID"""
        return f"event_{datetime.now().timestamp()}_{hash(threading.current_thread())}"
    
    def generate_comment_id(self) -> str:
        """Generate unique comment ID"""
        return f"comment_{datetime.now().timestamp()}_{hash(threading.current_thread())}"
    
    def generate_suggestion_id(self) -> str:
        """Generate unique suggestion ID"""
        return f"suggestion_{datetime.now().timestamp()}_{hash(threading.current_thread())}"
    
    async def shutdown(self):
        """Gracefully shutdown collaborative platform"""
        logger.info("Shutting down Collaborative Platform")
        
        # Notify all users
        for session_id in self.sessions.keys():
            await self.broadcast_to_session(session_id, {
                "type": "platform_shutdown",
                "timestamp": datetime.now().isoformat()
            })
        
        # Close WebSocket connections
        for websocket in self.websocket_connections.values():
            await websocket.close()
        
        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Cancel background services
        if self.event_processor:
            self.event_processor.cancel()
        if self.session_monitor:
            self.session_monitor.cancel()
        if self.auto_save_service:
            self.auto_save_service.cancel()
        
        # Shutdown integrated systems
        await self.multi_agent_system.shutdown()
        
        logger.info("Collaborative Platform shutdown complete")


# Global instance
collaborative_platform = CollaborativePlatform()