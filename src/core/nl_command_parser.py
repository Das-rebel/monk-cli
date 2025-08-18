"""
Natural Language Command Parser
Converts natural language input to structured commands
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

@dataclass
class ParsedIntent:
    """Parsed intent from natural language"""
    intent: str
    confidence: float
    command: Optional[str]
    parameters: Dict[str, Any]
    original_text: str

class NLCommandParser:
    """
    Converts natural language to structured commands
    """
    
    def __init__(self):
        # Intent patterns with associated commands
        self.intent_patterns = {
            'analyze_project': {
                'patterns': [
                    r'analyze.*project',
                    r'check.*project',
                    r'scan.*project',
                    r'look.*at.*project',
                    r'examine.*project'
                ],
                'command': '/analyze',
                'confidence': 0.9
            },
            'analyze_file': {
                'patterns': [
                    r'analyze.*file',
                    r'check.*file',
                    r'look.*at.*file',
                    r'examine.*file',
                    r'review.*file'
                ],
                'command': '/analyze',
                'confidence': 0.85
            },
            'show_file': {
                'patterns': [
                    r'show.*file',
                    r'display.*file',
                    r'cat.*file',
                    r'read.*file',
                    r'view.*file'
                ],
                'command': '/cat',
                'confidence': 0.9
            },
            'list_directory': {
                'patterns': [
                    r'list.*files?',
                    r'show.*files?',
                    r'ls.*',
                    r'what.*files?.*in',
                    r'files?.*in.*directory'
                ],
                'command': '/ls',
                'confidence': 0.85
            },
            'show_tree': {
                'patterns': [
                    r'show.*tree',
                    r'directory.*tree',
                    r'project.*structure',
                    r'folder.*structure'
                ],
                'command': '/tree',
                'confidence': 0.9
            },
            'git_status': {
                'patterns': [
                    r'git.*status',
                    r'check.*status',
                    r'repo.*status',
                    r'what.*changed',
                    r'working.*directory.*status'
                ],
                'command': '/status',
                'confidence': 0.9
            },
            'git_diff': {
                'patterns': [
                    r'git.*diff',
                    r'show.*diff',
                    r'what.*different',
                    r'changes.*made',
                    r'modifications'
                ],
                'command': '/diff',
                'confidence': 0.85
            },
            'workspace_info': {
                'patterns': [
                    r'current.*workspace',
                    r'workspace.*info',
                    r'where.*am.*i',
                    r'project.*info'
                ],
                'command': '/workspace',
                'confidence': 0.9
            },
            'clear_history': {
                'patterns': [
                    r'clear.*history',
                    r'clear.*conversation',
                    r'start.*fresh',
                    r'reset.*chat'
                ],
                'command': '/clear',
                'confidence': 0.95
            },
            'help': {
                'patterns': [
                    r'help',
                    r'commands?.*available',
                    r'what.*can.*do',
                    r'how.*use',
                    r'guide'
                ],
                'command': '/help',
                'confidence': 0.85
            }
        }
        
        # Parameter extraction patterns
        self.parameter_patterns = {
            'file_path': r'([/\w.-]+\.[\w]+)|(["\'][^"\']*["\'])',
            'directory_path': r'([/\w.-]+/?)|(["\'][^"\']*["\'])',
            'command_args': r'--?([\w-]+)(?:=([\w.-]+))?'
        }
    
    async def parse(self, text: str) -> ParsedIntent:
        """Parse natural language text to structured intent"""
        text_lower = text.lower().strip()
        
        # If already a slash command, return as-is
        if text.startswith('/'):
            return ParsedIntent(
                intent='slash_command',
                confidence=1.0,
                command=text,
                parameters={},
                original_text=text
            )
        
        # Find best matching intent
        best_match = None
        highest_confidence = 0.0
        
        for intent_name, intent_data in self.intent_patterns.items():
            for pattern in intent_data['patterns']:
                if re.search(pattern, text_lower):
                    confidence = intent_data['confidence']
                    # Boost confidence for exact matches
                    if pattern == text_lower:
                        confidence += 0.1
                    
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_match = {
                            'intent': intent_name,
                            'command': intent_data['command'],
                            'confidence': confidence
                        }
        
        if best_match:
            # Extract parameters
            parameters = self._extract_parameters(text, best_match['intent'])
            
            # Build command with parameters
            command = self._build_command(best_match['command'], parameters)
            
            return ParsedIntent(
                intent=best_match['intent'],
                confidence=best_match['confidence'],
                command=command,
                parameters=parameters,
                original_text=text
            )
        
        # No match found - general query
        return ParsedIntent(
            intent='general_query',
            confidence=0.5,
            command=None,
            parameters={'query': text},
            original_text=text
        )
    
    def _extract_parameters(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract parameters based on intent type"""
        parameters = {}
        
        if intent in ['analyze_file', 'show_file']:
            # Extract file path
            file_match = re.search(self.parameter_patterns['file_path'], text)
            if file_match:
                file_path = file_match.group(1) or file_match.group(2)
                parameters['file'] = file_path.strip('\'"')
        
        elif intent in ['list_directory', 'show_tree']:
            # Extract directory path
            dir_match = re.search(self.parameter_patterns['directory_path'], text)
            if dir_match:
                dir_path = dir_match.group(1) or dir_match.group(2)
                parameters['directory'] = dir_path.strip('\'"')
        
        elif intent == 'help':
            # Extract specific help topic
            help_topics = ['analyze', 'git', 'files', 'workspace']
            for topic in help_topics:
                if topic in text.lower():
                    parameters['topic'] = topic
                    break
        
        # Extract general command arguments
        arg_matches = re.findall(self.parameter_patterns['command_args'], text)
        for arg_name, arg_value in arg_matches:
            parameters[arg_name] = arg_value if arg_value else True
        
        return parameters
    
    def _build_command(self, base_command: str, parameters: Dict[str, Any]) -> str:
        """Build final command string with parameters"""
        command_parts = [base_command]
        
        # Add positional parameters
        if 'file' in parameters:
            command_parts.append(parameters['file'])
        elif 'directory' in parameters:
            command_parts.append(parameters['directory'])
        elif 'topic' in parameters:
            command_parts.append(parameters['topic'])
        
        # Add flag parameters
        for key, value in parameters.items():
            if key not in ['file', 'directory', 'topic', 'query']:
                if value is True:
                    command_parts.append(f'--{key}')
                else:
                    command_parts.append(f'--{key}={value}')
        
        return ' '.join(command_parts)
    
    def get_suggestions(self, partial_text: str) -> List[str]:
        """Get command suggestions for partial input"""
        partial_lower = partial_text.lower()
        suggestions = []
        
        # Check for partial matches in intent patterns
        for intent_name, intent_data in self.intent_patterns.items():
            command = intent_data['command']
            
            # Check if any pattern words match partial input
            for pattern in intent_data['patterns']:
                pattern_words = re.findall(r'\w+', pattern)
                for word in pattern_words:
                    if word.startswith(partial_lower) or partial_lower in word:
                        suggestion = f"{command} (from: {partial_text})"
                        if suggestion not in suggestions:
                            suggestions.append(suggestion)
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def explain_parsing(self, parsed: ParsedIntent) -> str:
        """Provide explanation of parsing result"""
        if parsed.intent == 'slash_command':
            return f"✅ Recognized as slash command: {parsed.command}"
        elif parsed.intent == 'general_query':
            return f"❓ Interpreted as general query (no specific command pattern matched)"
        else:
            explanation = f"🧠 **Parsed Intent**: {parsed.intent}\n"
            explanation += f"- **Confidence**: {parsed.confidence:.1%}\n"
            explanation += f"- **Command**: {parsed.command}\n"
            if parsed.parameters:
                params = ", ".join(f"{k}={v}" for k, v in parsed.parameters.items())
                explanation += f"- **Parameters**: {params}"
            return explanation

# Global NL command parser
nl_parser = NLCommandParser()


# Example usage and testing
if __name__ == "__main__":
    async def test_parser():
        test_cases = [
            "analyze this project",
            "show me the main.py file", 
            "what files are in src/",
            "git status",
            "clear conversation",
            "/help analyze"
        ]
        
        parser = NLCommandParser()
        
        for test in test_cases:
            result = await parser.parse(test)
            print(f"Input: {test}")
            print(f"Intent: {result.intent} ({result.confidence:.1%})")
            print(f"Command: {result.command}")
            print(f"Parameters: {result.parameters}")
            print("-" * 50)
    
    asyncio.run(test_parser())
