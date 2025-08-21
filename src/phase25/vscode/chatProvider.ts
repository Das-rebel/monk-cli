/**
 * MONK CLI Phase 2.5 - Chat Provider with Open Source Integration
 * Handles AI chat interactions with personality-based responses
 */

import * as vscode from 'vscode';
import axios from 'axios';

interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    metadata?: {
        personality?: string;
        context?: any;
        tokens?: number;
        executionTime?: number;
    };
}

interface ChatSession {
    id: string;
    messages: ChatMessage[];
    personality: string;
    context: any;
    created: Date;
    lastActivity: Date;
}

interface PersonalityConfig {
    id: string;
    name: string;
    description: string;
    systemPrompt: string;
    characteristics: string[];
    examples: string[];
}

export class MONKChatProvider {
    private config: any;
    private currentSession: ChatSession;
    private chatHistory: ChatSession[] = [];
    private personalities: Map<string, PersonalityConfig> = new Map();
    private currentPersonality: string = 'analytical';
    
    constructor(config: any) {
        this.config = config;
        this.initializePersonalities();
        this.createNewSession();
    }
    
    public async initialize(): Promise<void> {
        try {
            // Test connection to MONK CLI API
            const response = await axios.get(`${this.config.apiEndpoint}/health`, {
                timeout: 5000
            });
            
            if (response.status === 200) {
                console.log('Connected to MONK CLI API successfully');
            }
        } catch (error) {
            console.warn('Could not connect to MONK CLI API, using fallback mode:', error);
        }
        
        // Load any saved chat history
        await this.loadChatHistory();
    }
    
    private initializePersonalities(): void {
        const personalities: PersonalityConfig[] = [
            {
                id: 'analytical',
                name: 'Analytical',
                description: 'Logical, systematic, and detail-oriented approach to problem solving',
                systemPrompt: 'You are an analytical AI assistant focused on logical reasoning, systematic analysis, and providing detailed technical explanations. Break down complex problems into smaller components and provide step-by-step solutions.',
                characteristics: [
                    'Systematic approach',
                    'Data-driven insights',
                    'Logical reasoning',
                    'Detailed explanations',
                    'Evidence-based conclusions'
                ],
                examples: [
                    'Let me analyze this step by step...',
                    'Based on the data, I can see that...',
                    'The logical approach would be to...',
                    'Here\'s a systematic breakdown...'
                ]
            },
            {
                id: 'creative',
                name: 'Creative',
                description: 'Innovative and imaginative approach to finding unique solutions',
                systemPrompt: 'You are a creative AI assistant who thinks outside the box and provides innovative solutions. Explore alternative approaches, suggest creative patterns, and help brainstorm unique ideas.',
                characteristics: [
                    'Innovative thinking',
                    'Alternative approaches',
                    'Pattern recognition',
                    'Brainstorming',
                    'Unique perspectives'
                ],
                examples: [
                    'Here\'s an interesting alternative approach...',
                    'What if we tried a different pattern?',
                    'I see some creative possibilities here...',
                    'Let\'s brainstorm some innovative solutions...'
                ]
            },
            {
                id: 'detail-oriented',
                name: 'Detail-Oriented',
                description: 'Meticulous attention to code quality, testing, and documentation',
                systemPrompt: 'You are a detail-oriented AI assistant focused on code quality, thorough testing, proper documentation, and best practices. Pay attention to edge cases, error handling, and maintainability.',
                characteristics: [
                    'Code quality focus',
                    'Thorough testing',
                    'Comprehensive documentation',
                    'Best practices',
                    'Edge case consideration'
                ],
                examples: [
                    'Let me check all the details...',
                    'We should consider these edge cases...',
                    'Here\'s how to properly document this...',
                    'Don\'t forget to add error handling for...'
                ]
            },
            {
                id: 'collaborative',
                name: 'Collaborative',
                description: 'Team-oriented approach focusing on communication and coordination',
                systemPrompt: 'You are a collaborative AI assistant who excels at communication, coordination, and team-oriented solutions. Focus on clear explanations, shared understanding, and facilitating collaboration.',
                characteristics: [
                    'Clear communication',
                    'Team coordination',
                    'Shared understanding',
                    'Inclusive approach',
                    'Conflict resolution'
                ],
                examples: [
                    'Let me explain this clearly...',
                    'Here\'s how the team can work together...',
                    'To ensure everyone understands...',
                    'This approach will help coordinate...'
                ]
            }
        ];
        
        personalities.forEach(personality => {
            this.personalities.set(personality.id, personality);
        });
    }
    
    private createNewSession(): void {
        this.currentSession = {
            id: this.generateSessionId(),
            messages: [],
            personality: this.currentPersonality,
            context: {},
            created: new Date(),
            lastActivity: new Date()
        };
    }
    
    private generateSessionId(): string {
        return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    public async sendMessage(content: string, context?: any): Promise<string> {
        try {
            // Add user message to session
            const userMessage: ChatMessage = {
                id: this.generateMessageId(),
                role: 'user',
                content: content,
                timestamp: new Date(),
                metadata: {
                    personality: this.currentPersonality,
                    context: context
                }
            };
            
            this.currentSession.messages.push(userMessage);
            this.currentSession.lastActivity = new Date();
            
            // Get AI response
            const response = await this.getAIResponse(content, context);
            
            // Add assistant response to session
            const assistantMessage: ChatMessage = {
                id: this.generateMessageId(),
                role: 'assistant',
                content: response,
                timestamp: new Date(),
                metadata: {
                    personality: this.currentPersonality,
                    context: context
                }
            };
            
            this.currentSession.messages.push(assistantMessage);
            
            return response;
            
        } catch (error) {
            console.error('Error sending message:', error);
            return 'I apologize, but I encountered an error processing your request. Please try again.';
        }
    }
    
    private async getAIResponse(content: string, context?: any): Promise<string> {
        try {
            // Try MONK CLI API first
            const response = await this.getMONKAPIResponse(content, context);
            if (response) return response;
        } catch (error) {
            console.warn('MONK API unavailable, using fallback:', error);
        }
        
        // Fallback to built-in responses
        return this.getFallbackResponse(content, context);
    }
    
    private async getMONKAPIResponse(content: string, context?: any): Promise<string | null> {
        try {
            const personality = this.personalities.get(this.currentPersonality);
            if (!personality) {
                throw new Error('Unknown personality');
            }
            
            const requestData = {
                message: content,
                personality: this.currentPersonality,
                context: context,
                history: this.currentSession.messages.slice(-10), // Last 10 messages for context
                system_prompt: personality.systemPrompt
            };
            
            const response = await axios.post(
                `${this.config.apiEndpoint}/api/chat`,
                requestData,
                {
                    timeout: 30000,
                    headers: {
                        'Content-Type': 'application/json'
                    }
                }
            );
            
            if (response.status === 200 && response.data.response) {
                return response.data.response;
            }
            
            return null;
        } catch (error) {
            console.error('Error calling MONK API:', error);
            return null;
        }
    }
    
    private getFallbackResponse(content: string, context?: any): string {
        const personality = this.personalities.get(this.currentPersonality);
        if (!personality) {
            return 'I apologize, but I\'m having trouble with my personality configuration.';
        }
        
        const lowerContent = content.toLowerCase();
        
        // Code explanation requests
        if (lowerContent.includes('explain') && context?.selectedText) {
            return this.getCodeExplanationFallback(context.selectedText, personality);
        }
        
        // Code generation requests
        if (lowerContent.includes('generate') || lowerContent.includes('create')) {
            return this.getCodeGenerationFallback(content, personality);
        }
        
        // Refactoring requests
        if (lowerContent.includes('refactor') || lowerContent.includes('improve')) {
            return this.getRefactoringFallback(content, personality);
        }
        
        // Bug finding requests
        if (lowerContent.includes('bug') || lowerContent.includes('error') || lowerContent.includes('issue')) {
            return this.getBugAnalysisFallback(content, personality);
        }
        
        // General questions
        return this.getGeneralResponseFallback(content, personality);
    }
    
    private getCodeExplanationFallback(code: string, personality: PersonalityConfig): string {
        const lines = code.split('\n').length;
        const hasComments = code.includes('//') || code.includes('#') || code.includes('/*');
        
        switch (personality.id) {
            case 'analytical':
                return `Let me analyze this code systematically:

**Structure Analysis:**
- This code snippet contains ${lines} lines
- ${hasComments ? 'Includes comments' : 'No comments found'}

**Functionality:**
The code appears to define functionality that processes data through a series of operations. Let me break down the key components:

1. **Input processing**: The code likely accepts input parameters
2. **Logic execution**: Core business logic is implemented
3. **Output generation**: Results are produced and returned

**Recommendations:**
- Consider adding more descriptive comments
- Ensure proper error handling is in place
- Review variable naming conventions

Would you like me to analyze any specific aspect in more detail?`;

            case 'creative':
                return `Interesting code! Let me look at this from different angles:

**Creative Patterns I Notice:**
This code demonstrates some interesting approaches to problem-solving. The structure suggests a creative solution that could be enhanced in several ways:

**Alternative Approaches:**
- Could be refactored using a more functional approach
- Might benefit from a design pattern like Strategy or Observer
- Consider using modern language features for cleaner syntax

**Enhancement Ideas:**
🚀 Add configuration options for flexibility
🔧 Implement a plugin system for extensibility
📊 Consider adding metrics collection
🎨 Use more expressive variable names

What kind of creative improvements are you thinking about?`;

            case 'detail-oriented':
                return `Here's a detailed analysis of your code:

**Code Quality Assessment:**
- **Lines of code**: ${lines}
- **Documentation**: ${hasComments ? '✅ Has comments' : '❌ Missing comments'}
- **Structure**: Appears to follow standard conventions

**Detailed Review:**
1. **Variable naming**: Review for clarity and consistency
2. **Error handling**: Ensure all edge cases are covered
3. **Performance**: Check for potential optimization opportunities
4. **Testing**: Consider adding unit tests
5. **Documentation**: Add comprehensive docstrings

**Best Practices Checklist:**
- [ ] Input validation
- [ ] Error handling
- [ ] Proper logging
- [ ] Type hints (if applicable)
- [ ] Unit tests
- [ ] Code comments

Would you like me to focus on any specific quality aspect?`;

            case 'collaborative':
                return `Let me explain this code in a way that's easy to understand and share with your team:

**Code Overview for Team Discussion:**
This code snippet implements functionality that your team can build upon. Here's what it does in simple terms:

**Key Components:**
1. **Purpose**: Solves a specific problem in your application
2. **Approach**: Uses a straightforward methodology
3. **Integration**: Can be connected with other system components

**Team Collaboration Points:**
- Share this with your team for review
- Consider pair programming for enhancements
- Document any assumptions for future developers
- Plan integration with existing codebase

**Communication Summary:**
This code provides a foundation that the team can discuss and improve together. It's ready for code review and collaborative enhancement.

How would you like to share this with your team?`;

            default:
                return `Here's an explanation of the selected code:\n\n${code}\n\nThis code performs specific operations based on the language and context. Would you like me to explain any particular aspect in more detail?`;
        }
    }
    
    private getCodeGenerationFallback(request: string, personality: PersonalityConfig): string {
        switch (personality.id) {
            case 'analytical':
                return `I'll help you generate code systematically. To provide the most accurate solution, I need to analyze your requirements:

**Requirements Analysis:**
- **Functionality needed**: ${request}
- **Programming language**: Please specify (Python, JavaScript, etc.)
- **Framework/libraries**: Any specific requirements?
- **Performance requirements**: Any specific needs?

**Systematic Approach:**
1. **Define inputs and outputs**
2. **Plan the algorithm**
3. **Implement core logic**
4. **Add error handling**
5. **Include documentation**

Could you provide more specific details about what you'd like to generate?`;

            case 'creative':
                return `Great! Let's create something innovative together! 🚀

**Creative Code Generation Ideas:**
Based on your request "${request}", here are some creative approaches we could take:

**Modern Patterns to Consider:**
- Functional programming approach
- Reactive programming style
- Microservices architecture
- Event-driven design

**Innovation Opportunities:**
- Use cutting-edge language features
- Implement creative algorithms
- Add extensibility points
- Consider future scalability

**Next Steps:**
Tell me more about your vision, and I'll help you craft something unique and effective!

What programming language and style are you envisioning?`;

            case 'detail-oriented':
                return `I'll help you generate comprehensive, well-documented code. Let me ensure we cover all the important details:

**Detailed Requirements Gathering:**
- **Exact functionality**: ${request}
- **Input specifications**: Data types, validation rules
- **Output requirements**: Format, error handling
- **Edge cases**: What should happen in unusual situations?
- **Performance criteria**: Speed, memory usage requirements
- **Testing strategy**: How should we validate the code?

**Code Quality Standards:**
- Full documentation with docstrings
- Comprehensive error handling
- Input validation
- Type hints (where applicable)
- Unit test examples
- Code comments explaining complex logic

Please provide specific details so I can generate production-ready code.`;

            case 'collaborative':
                return `Let's work together to generate code that your whole team can understand and maintain!

**Collaborative Code Generation:**
For "${request}", let's create something that promotes team collaboration:

**Team-Friendly Approach:**
- Clear, readable code structure
- Comprehensive documentation
- Standard naming conventions
- Modular design for easy collaboration

**Knowledge Sharing:**
- Code comments explain the "why", not just "what"
- Examples and usage patterns included
- Easy to extend and modify
- Follows team coding standards

**Next Steps for Team Success:**
1. Define the requirements together
2. Choose coding standards
3. Plan testing approach
4. Consider code review process

What programming language does your team prefer, and are there any specific standards I should follow?`;

            default:
                return `I'd be happy to help generate code for: ${request}\n\nPlease specify:\n- Programming language\n- Specific requirements\n- Any constraints or preferences\n\nThis will help me provide the most relevant solution.`;
        }
    }
    
    private getRefactoringFallback(request: string, personality: PersonalityConfig): string {
        return `I can help refactor your code with a ${personality.name.toLowerCase()} approach. Please share the code you'd like to improve, and I'll provide suggestions based on ${personality.description.toLowerCase()}.`;
    }
    
    private getBugAnalysisFallback(request: string, personality: PersonalityConfig): string {
        return `I'll analyze your code for potential issues using my ${personality.name.toLowerCase()} approach. Please share the code, and I'll help identify and fix any bugs or problems.`;
    }
    
    private getGeneralResponseFallback(content: string, personality: PersonalityConfig): string {
        const example = personality.examples[Math.floor(Math.random() * personality.examples.length)];
        
        return `${example}

I'm here to help with your development needs using my ${personality.name.toLowerCase()} approach. I can assist with:

- Code explanation and analysis
- Code generation and implementation
- Refactoring and optimization
- Bug detection and debugging
- Architecture and design decisions

What specific challenge are you working on? Please feel free to share code, ask questions, or describe what you're trying to accomplish.`;
    }
    
    private generateMessageId(): string {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    public async setPersonality(personalityId: string): Promise<void> {
        if (this.personalities.has(personalityId)) {
            this.currentPersonality = personalityId;
            this.currentSession.personality = personalityId;
            
            // Add system message about personality change
            const systemMessage: ChatMessage = {
                id: this.generateMessageId(),
                role: 'system',
                content: `Personality changed to ${this.personalities.get(personalityId)?.name}`,
                timestamp: new Date(),
                metadata: {
                    personality: personalityId
                }
            };
            
            this.currentSession.messages.push(systemMessage);
        }
    }
    
    public getCurrentPersonality(): PersonalityConfig | undefined {
        return this.personalities.get(this.currentPersonality);
    }
    
    public getAvailablePersonalities(): PersonalityConfig[] {
        return Array.from(this.personalities.values());
    }
    
    public async clearHistory(): Promise<void> {
        this.currentSession.messages = [];
    }
    
    public async getHistory(): Promise<ChatSession> {
        return { ...this.currentSession };
    }
    
    public async getAllSessions(): Promise<ChatSession[]> {
        return [...this.chatHistory, this.currentSession];
    }
    
    public async startNewSession(): Promise<void> {
        // Save current session to history
        this.chatHistory.push({ ...this.currentSession });
        
        // Create new session
        this.createNewSession();
    }
    
    public async loadSession(sessionId: string): Promise<boolean> {
        const session = this.chatHistory.find(s => s.id === sessionId);
        if (session) {
            // Save current session
            const currentIndex = this.chatHistory.findIndex(s => s.id === this.currentSession.id);
            if (currentIndex >= 0) {
                this.chatHistory[currentIndex] = { ...this.currentSession };
            } else {
                this.chatHistory.push({ ...this.currentSession });
            }
            
            // Load requested session
            this.currentSession = { ...session };
            this.currentPersonality = session.personality;
            
            return true;
        }
        
        return false;
    }
    
    private async loadChatHistory(): Promise<void> {
        try {
            // In a real implementation, load from persistent storage
            // For now, this is just a placeholder
            console.log('Chat history loading is ready for implementation');
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }
    
    public async saveChatHistory(): Promise<void> {
        try {
            // In a real implementation, save to persistent storage
            // For now, this is just a placeholder
            console.log('Chat history saving is ready for implementation');
        } catch (error) {
            console.error('Error saving chat history:', error);
        }
    }
    
    public getMetrics(): any {
        const totalMessages = this.currentSession.messages.length;
        const userMessages = this.currentSession.messages.filter(m => m.role === 'user').length;
        const assistantMessages = this.currentSession.messages.filter(m => m.role === 'assistant').length;
        
        return {
            currentSession: {
                id: this.currentSession.id,
                totalMessages: totalMessages,
                userMessages: userMessages,
                assistantMessages: assistantMessages,
                personality: this.currentPersonality,
                created: this.currentSession.created,
                lastActivity: this.currentSession.lastActivity
            },
            totalSessions: this.chatHistory.length + 1,
            availablePersonalities: this.personalities.size,
            apiEndpoint: this.config.apiEndpoint
        };
    }
}