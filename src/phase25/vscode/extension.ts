/**
 * MONK CLI Phase 2.5 - VS Code Extension with Open Source GitHub Copilot Chat
 * Enhanced development experience with AI-powered assistance
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { TreeSitterExplorer } from './treeSitterIntegration';
import { MONKChatProvider } from './chatProvider';
import { CodeAnalysisProvider } from './codeAnalysisProvider';
import { PersonalitySelector } from './personalitySelector';

interface MONKConfiguration {
    apiEndpoint: string;
    enablePersonalitySelection: boolean;
    defaultPersonality: string;
    enableCodeAnalysis: boolean;
    enableTreeSitterParsing: boolean;
    maxContextLength: number;
    autoSaveInterval: number;
}

interface CodeContext {
    activeFile?: string;
    selectedText?: string;
    cursorPosition?: vscode.Position;
    workspaceRoot?: string;
    recentFiles: string[];
    projectType?: string;
}

class MONKExtension {
    private context: vscode.ExtensionContext;
    private treeSitterExplorer: TreeSitterExplorer;
    private chatProvider: MONKChatProvider;
    private codeAnalysisProvider: CodeAnalysisProvider;
    private personalitySelector: PersonalitySelector;
    private config: MONKConfiguration;
    
    // Status bar items
    private statusBarItem: vscode.StatusBarItem;
    private personalityStatusItem: vscode.StatusBarItem;
    
    // Output channels
    private outputChannel: vscode.OutputChannel;
    
    // Webview panels
    private chatPanel?: vscode.WebviewPanel;
    private analysisPanel?: vscode.WebviewPanel;
    
    constructor(context: vscode.ExtensionContext) {
        this.context = context;
        this.loadConfiguration();
        
        // Initialize components
        this.treeSitterExplorer = new TreeSitterExplorer(this.config);
        this.chatProvider = new MONKChatProvider(this.config);
        this.codeAnalysisProvider = new CodeAnalysisProvider(this.treeSitterExplorer);
        this.personalitySelector = new PersonalitySelector();
        
        // Create output channel
        this.outputChannel = vscode.window.createOutputChannel('MONK CLI');
        
        // Create status bar items
        this.createStatusBarItems();
    }
    
    private loadConfiguration() {
        const config = vscode.workspace.getConfiguration('monk');
        
        this.config = {
            apiEndpoint: config.get('apiEndpoint', 'http://localhost:8000'),
            enablePersonalitySelection: config.get('enablePersonalitySelection', true),
            defaultPersonality: config.get('defaultPersonality', 'analytical'),
            enableCodeAnalysis: config.get('enableCodeAnalysis', true),
            enableTreeSitterParsing: config.get('enableTreeSitterParsing', true),
            maxContextLength: config.get('maxContextLength', 8192),
            autoSaveInterval: config.get('autoSaveInterval', 30000)
        };
    }
    
    private createStatusBarItems() {
        // Main MONK status
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left, 
            100
        );
        this.statusBarItem.text = '$(robot) MONK';
        this.statusBarItem.tooltip = 'MONK CLI Assistant - Click to open chat';
        this.statusBarItem.command = 'monk.openChat';
        this.statusBarItem.show();
        
        // Personality selector
        if (this.config.enablePersonalitySelection) {
            this.personalityStatusItem = vscode.window.createStatusBarItem(
                vscode.StatusBarAlignment.Left,
                99
            );
            this.personalityStatusItem.text = `$(person) ${this.config.defaultPersonality}`;
            this.personalityStatusItem.tooltip = 'Current AI personality - Click to change';
            this.personalityStatusItem.command = 'monk.selectPersonality';
            this.personalityStatusItem.show();
        }
    }
    
    public async activate() {
        this.outputChannel.appendLine('Activating MONK CLI Extension...');
        
        try {
            // Initialize tree-sitter explorer
            if (this.config.enableTreeSitterParsing) {
                await this.treeSitterExplorer.initialize();
                this.outputChannel.appendLine('Tree-sitter explorer initialized');
            }
            
            // Initialize chat provider
            await this.chatProvider.initialize();
            this.outputChannel.appendLine('Chat provider initialized');
            
            // Register commands
            this.registerCommands();
            
            // Register event listeners
            this.registerEventListeners();
            
            // Start background services
            this.startBackgroundServices();
            
            this.outputChannel.appendLine('MONK CLI Extension activated successfully');
            vscode.window.showInformationMessage('MONK CLI Extension is now active!');
            
        } catch (error) {
            this.outputChannel.appendLine(`Error activating extension: ${error}`);
            vscode.window.showErrorMessage(`Failed to activate MONK CLI: ${error}`);
        }
    }
    
    private registerCommands() {
        const commands = [
            // Chat commands
            vscode.commands.registerCommand('monk.openChat', () => this.openChatPanel()),
            vscode.commands.registerCommand('monk.askQuestion', () => this.askQuestion()),
            vscode.commands.registerCommand('monk.explainCode', () => this.explainSelectedCode()),
            vscode.commands.registerCommand('monk.generateCode', () => this.generateCode()),
            vscode.commands.registerCommand('monk.refactorCode', () => this.refactorCode()),
            
            // Analysis commands
            vscode.commands.registerCommand('monk.analyzeFile', () => this.analyzeCurrentFile()),
            vscode.commands.registerCommand('monk.analyzeProject', () => this.analyzeProject()),
            vscode.commands.registerCommand('monk.findBugs', () => this.findBugs()),
            vscode.commands.registerCommand('monk.suggestImprovements', () => this.suggestImprovements()),
            
            // Personality commands
            vscode.commands.registerCommand('monk.selectPersonality', () => this.selectPersonality()),
            vscode.commands.registerCommand('monk.resetPersonality', () => this.resetPersonality()),
            
            // Utility commands
            vscode.commands.registerCommand('monk.clearChat', () => this.clearChatHistory()),
            vscode.commands.registerCommand('monk.exportChat', () => this.exportChatHistory()),
            vscode.commands.registerCommand('monk.showSettings', () => this.showSettings()),
            vscode.commands.registerCommand('monk.restartExtension', () => this.restartExtension()),
        ];
        
        commands.forEach(disposable => {
            this.context.subscriptions.push(disposable);
        });
    }
    
    private registerEventListeners() {
        // File change events
        const fileWatcher = vscode.workspace.createFileSystemWatcher('**/*.{py,js,ts,tsx,json}');
        
        fileWatcher.onDidChange(async (uri) => {
            if (this.config.enableTreeSitterParsing) {
                await this.treeSitterExplorer.parseFile(uri.fsPath, true);
            }
        });
        
        fileWatcher.onDidCreate(async (uri) => {
            if (this.config.enableTreeSitterParsing) {
                await this.treeSitterExplorer.parseFile(uri.fsPath);
            }
        });
        
        this.context.subscriptions.push(fileWatcher);
        
        // Active editor change
        const editorChangeDisposable = vscode.window.onDidChangeActiveTextEditor(
            (editor) => this.onActiveEditorChanged(editor)
        );
        this.context.subscriptions.push(editorChangeDisposable);
        
        // Selection change
        const selectionChangeDisposable = vscode.window.onDidChangeTextEditorSelection(
            (event) => this.onSelectionChanged(event)
        );
        this.context.subscriptions.push(selectionChangeDisposable);
        
        // Configuration change
        const configChangeDisposable = vscode.workspace.onDidChangeConfiguration(
            (event) => {
                if (event.affectsConfiguration('monk')) {
                    this.loadConfiguration();
                    this.updateStatusBar();
                }
            }
        );
        this.context.subscriptions.push(configChangeDisposable);
    }
    
    private startBackgroundServices() {
        // Auto-save chat history
        setInterval(() => {
            this.saveChatHistory();
        }, this.config.autoSaveInterval);
        
        // Parse workspace files if enabled
        if (this.config.enableTreeSitterParsing) {
            this.parseWorkspaceFiles();
        }
    }
    
    private async parseWorkspaceFiles() {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) return;
        
        try {
            for (const folder of workspaceFolders) {
                await this.treeSitterExplorer.parseDirectory(folder.uri.fsPath);
            }
            this.outputChannel.appendLine('Workspace parsing completed');
        } catch (error) {
            this.outputChannel.appendLine(`Error parsing workspace: ${error}`);
        }
    }
    
    private async openChatPanel() {
        if (this.chatPanel) {
            this.chatPanel.reveal(vscode.ViewColumn.Beside);
            return;
        }
        
        this.chatPanel = vscode.window.createWebviewPanel(
            'monkChat',
            'MONK AI Assistant',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [
                    vscode.Uri.joinPath(this.context.extensionUri, 'media'),
                    vscode.Uri.joinPath(this.context.extensionUri, 'dist')
                ]
            }
        );
        
        this.chatPanel.webview.html = await this.getChatPanelHtml();
        
        // Handle messages from webview
        this.chatPanel.webview.onDidReceiveMessage(
            async (message) => {
                await this.handleChatMessage(message);
            },
            undefined,
            this.context.subscriptions
        );
        
        // Clean up when panel is disposed
        this.chatPanel.onDidDispose(
            () => {
                this.chatPanel = undefined;
            },
            null,
            this.context.subscriptions
        );
        
        // Send initial context
        await this.sendContextToChat();
    }
    
    private async getChatPanelHtml(): Promise<string> {
        const scriptUri = this.chatPanel!.webview.asWebviewUri(
            vscode.Uri.joinPath(this.context.extensionUri, 'dist', 'chat.js')
        );
        const styleUri = this.chatPanel!.webview.asWebviewUri(
            vscode.Uri.joinPath(this.context.extensionUri, 'media', 'chat.css')
        );
        
        return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <title>MONK AI Assistant</title>
        </head>
        <body>
            <div id="app">
                <div class="chat-header">
                    <h2>🤖 MONK AI Assistant</h2>
                    <div class="personality-indicator">
                        <span id="currentPersonality">${this.config.defaultPersonality}</span>
                        <button id="changePersonality">Change</button>
                    </div>
                </div>
                
                <div class="chat-container">
                    <div id="chatMessages" class="chat-messages"></div>
                    <div class="chat-input-container">
                        <textarea 
                            id="chatInput" 
                            placeholder="Ask MONK anything about your code..."
                            rows="3"
                        ></textarea>
                        <div class="chat-actions">
                            <button id="sendButton" class="primary">Send</button>
                            <button id="clearButton">Clear</button>
                            <button id="contextButton">Add Context</button>
                        </div>
                    </div>
                </div>
                
                <div class="quick-actions">
                    <button class="quick-action" data-action="explain">Explain Selected Code</button>
                    <button class="quick-action" data-action="generate">Generate Code</button>
                    <button class="quick-action" data-action="refactor">Refactor Code</button>
                    <button class="quick-action" data-action="debug">Find Bugs</button>
                </div>
            </div>
            
            <script src="${scriptUri}"></script>
        </body>
        </html>
        `;
    }
    
    private async handleChatMessage(message: any) {
        switch (message.type) {
            case 'sendMessage':
                await this.processChatMessage(message.text);
                break;
            
            case 'changePersonality':
                await this.selectPersonality();
                break;
            
            case 'quickAction':
                await this.handleQuickAction(message.action);
                break;
            
            case 'addContext':
                await this.addContextToChat();
                break;
            
            case 'clearChat':
                await this.clearChatHistory();
                break;
        }
    }
    
    private async processChatMessage(userMessage: string) {
        try {
            // Get current context
            const context = await this.getCurrentContext();
            
            // Send message to chat provider
            const response = await this.chatProvider.sendMessage(userMessage, context);
            
            // Send response back to webview
            this.chatPanel?.webview.postMessage({
                type: 'chatResponse',
                response: response,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            this.outputChannel.appendLine(`Error processing chat message: ${error}`);
            
            this.chatPanel?.webview.postMessage({
                type: 'chatError',
                error: `Sorry, I encountered an error: ${error}`
            });
        }
    }
    
    private async getCurrentContext(): Promise<CodeContext> {
        const activeEditor = vscode.window.activeTextEditor;
        const workspaceFolders = vscode.workspace.workspaceFolders;
        
        const context: CodeContext = {
            recentFiles: []
        };
        
        if (activeEditor) {
            context.activeFile = activeEditor.document.fileName;
            context.cursorPosition = activeEditor.selection.active;
            
            // Get selected text if any
            const selection = activeEditor.selection;
            if (!selection.isEmpty) {
                context.selectedText = activeEditor.document.getText(selection);
            }
        }
        
        if (workspaceFolders && workspaceFolders.length > 0) {
            context.workspaceRoot = workspaceFolders[0].uri.fsPath;
            context.projectType = await this.detectProjectType(context.workspaceRoot);
        }
        
        // Get recently opened files
        context.recentFiles = await this.getRecentFiles();
        
        return context;
    }
    
    private async detectProjectType(workspaceRoot: string): Promise<string> {
        const files = ['package.json', 'requirements.txt', 'Cargo.toml', 'go.mod', 'pom.xml'];
        
        for (const file of files) {
            const filePath = path.join(workspaceRoot, file);
            if (fs.existsSync(filePath)) {
                switch (file) {
                    case 'package.json': return 'javascript/typescript';
                    case 'requirements.txt': return 'python';
                    case 'Cargo.toml': return 'rust';
                    case 'go.mod': return 'go';
                    case 'pom.xml': return 'java';
                }
            }
        }
        
        return 'unknown';
    }
    
    private async getRecentFiles(): Promise<string[]> {
        // Get recently opened files from VS Code
        const recentFiles: string[] = [];
        
        // This is a simplified implementation
        // In practice, you'd access VS Code's recent files API
        
        return recentFiles;
    }
    
    private async handleQuickAction(action: string) {
        switch (action) {
            case 'explain':
                await this.explainSelectedCode();
                break;
            case 'generate':
                await this.generateCode();
                break;
            case 'refactor':
                await this.refactorCode();
                break;
            case 'debug':
                await this.findBugs();
                break;
        }
    }
    
    private async askQuestion() {
        const question = await vscode.window.showInputBox({
            prompt: 'What would you like to ask MONK?',
            placeHolder: 'Enter your question...'
        });
        
        if (question) {
            await this.openChatPanel();
            await this.processChatMessage(question);
        }
    }
    
    private async explainSelectedCode() {
        const activeEditor = vscode.window.activeTextEditor;
        if (!activeEditor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }
        
        const selection = activeEditor.selection;
        if (selection.isEmpty) {
            vscode.window.showWarningMessage('Please select code to explain');
            return;
        }
        
        const selectedText = activeEditor.document.getText(selection);
        const language = activeEditor.document.languageId;
        
        const prompt = `Explain this ${language} code:\n\n${selectedText}`;
        
        await this.openChatPanel();
        await this.processChatMessage(prompt);
    }
    
    private async generateCode() {
        const requirement = await vscode.window.showInputBox({
            prompt: 'What code would you like me to generate?',
            placeHolder: 'Describe what you want to build...'
        });
        
        if (requirement) {
            const activeEditor = vscode.window.activeTextEditor;
            const language = activeEditor ? activeEditor.document.languageId : 'auto-detect';
            
            const prompt = `Generate ${language} code for: ${requirement}`;
            
            await this.openChatPanel();
            await this.processChatMessage(prompt);
        }
    }
    
    private async refactorCode() {
        const activeEditor = vscode.window.activeTextEditor;
        if (!activeEditor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }
        
        const selection = activeEditor.selection;
        if (selection.isEmpty) {
            vscode.window.showWarningMessage('Please select code to refactor');
            return;
        }
        
        const selectedText = activeEditor.document.getText(selection);
        const language = activeEditor.document.languageId;
        
        const refactorType = await vscode.window.showQuickPick([
            'Improve readability',
            'Optimize performance',
            'Extract functions',
            'Add error handling',
            'Simplify logic',
            'Custom refactoring'
        ], {
            placeHolder: 'What type of refactoring?'
        });
        
        if (refactorType) {
            let prompt = `Refactor this ${language} code to ${refactorType.toLowerCase()}:\n\n${selectedText}`;
            
            if (refactorType === 'Custom refactoring') {
                const customInstructions = await vscode.window.showInputBox({
                    prompt: 'Specify your refactoring requirements',
                    placeHolder: 'Describe how you want the code refactored...'
                });
                
                if (customInstructions) {
                    prompt = `Refactor this ${language} code: ${customInstructions}\n\n${selectedText}`;
                }
            }
            
            await this.openChatPanel();
            await this.processChatMessage(prompt);
        }
    }
    
    private async analyzeCurrentFile() {
        const activeEditor = vscode.window.activeTextEditor;
        if (!activeEditor) {
            vscode.window.showWarningMessage('No active file to analyze');
            return;
        }
        
        const filePath = activeEditor.document.fileName;
        
        try {
            const analysis = await this.codeAnalysisProvider.analyzeFile(filePath);
            await this.showAnalysisPanel(analysis);
        } catch (error) {
            vscode.window.showErrorMessage(`Error analyzing file: ${error}`);
        }
    }
    
    private async analyzeProject() {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) {
            vscode.window.showWarningMessage('No workspace folder found');
            return;
        }
        
        const projectPath = workspaceFolders[0].uri.fsPath;
        
        try {
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Analyzing project...',
                cancellable: true
            }, async (progress, token) => {
                const analysis = await this.codeAnalysisProvider.analyzeProject(
                    projectPath, 
                    (current, total) => {
                        progress.report({
                            increment: (current / total) * 100,
                            message: `Analyzing file ${current} of ${total}`
                        });
                    }
                );
                
                if (!token.isCancellationRequested) {
                    await this.showAnalysisPanel(analysis);
                }
            });
        } catch (error) {
            vscode.window.showErrorMessage(`Error analyzing project: ${error}`);
        }
    }
    
    private async findBugs() {
        const activeEditor = vscode.window.activeTextEditor;
        if (!activeEditor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }
        
        const document = activeEditor.document;
        const code = document.getText();
        const language = document.languageId;
        
        const prompt = `Analyze this ${language} code for potential bugs, issues, and improvements:\n\n${code}`;
        
        await this.openChatPanel();
        await this.processChatMessage(prompt);
    }
    
    private async suggestImprovements() {
        const activeEditor = vscode.window.activeTextEditor;
        if (!activeEditor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }
        
        const document = activeEditor.document;
        const code = document.getText();
        const language = document.languageId;
        
        const prompt = `Suggest improvements for this ${language} code (performance, readability, best practices):\n\n${code}`;
        
        await this.openChatPanel();
        await this.processChatMessage(prompt);
    }
    
    private async selectPersonality() {
        const personalities = await this.personalitySelector.getAvailablePersonalities();
        
        const selectedPersonality = await vscode.window.showQuickPick(personalities, {
            placeHolder: 'Select AI personality'
        });
        
        if (selectedPersonality) {
            this.config.defaultPersonality = selectedPersonality.id;
            await this.chatProvider.setPersonality(selectedPersonality.id);
            this.updatePersonalityStatus(selectedPersonality.name);
            
            vscode.window.showInformationMessage(
                `Switched to ${selectedPersonality.name} personality`
            );
        }
    }
    
    private async resetPersonality() {
        this.config.defaultPersonality = 'analytical';
        await this.chatProvider.setPersonality('analytical');
        this.updatePersonalityStatus('Analytical');
        
        vscode.window.showInformationMessage('Personality reset to Analytical');
    }
    
    private updatePersonalityStatus(personalityName: string) {
        if (this.personalityStatusItem) {
            this.personalityStatusItem.text = `$(person) ${personalityName}`;
        }
    }
    
    private async showAnalysisPanel(analysis: any) {
        if (this.analysisPanel) {
            this.analysisPanel.reveal(vscode.ViewColumn.Two);
        } else {
            this.analysisPanel = vscode.window.createWebviewPanel(
                'monkAnalysis',
                'Code Analysis',
                vscode.ViewColumn.Two,
                { enableScripts: true }
            );
            
            this.analysisPanel.onDidDispose(() => {
                this.analysisPanel = undefined;
            });
        }
        
        this.analysisPanel.webview.html = this.getAnalysisHtml(analysis);
    }
    
    private getAnalysisHtml(analysis: any): string {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                .metric { margin: 10px 0; }
                .metric-value { font-weight: bold; }
                .issue { margin: 10px 0; padding: 10px; border-left: 3px solid #ff6b6b; }
                .suggestion { margin: 10px 0; padding: 10px; border-left: 3px solid #51cf66; }
            </style>
        </head>
        <body>
            <h1>📊 Code Analysis Results</h1>
            <pre>${JSON.stringify(analysis, null, 2)}</pre>
        </body>
        </html>
        `;
    }
    
    private async sendContextToChat() {
        if (!this.chatPanel) return;
        
        const context = await this.getCurrentContext();
        
        this.chatPanel.webview.postMessage({
            type: 'updateContext',
            context: context
        });
    }
    
    private async addContextToChat() {
        await this.sendContextToChat();
        
        vscode.window.showInformationMessage('Context added to chat');
    }
    
    private async clearChatHistory() {
        if (this.chatPanel) {
            this.chatPanel.webview.postMessage({
                type: 'clearChat'
            });
        }
        
        await this.chatProvider.clearHistory();
        vscode.window.showInformationMessage('Chat history cleared');
    }
    
    private async exportChatHistory() {
        const history = await this.chatProvider.getHistory();
        
        const uri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file('monk-chat-history.json'),
            filters: { 'JSON': ['json'] }
        });
        
        if (uri) {
            fs.writeFileSync(uri.fsPath, JSON.stringify(history, null, 2));
            vscode.window.showInformationMessage('Chat history exported successfully');
        }
    }
    
    private async saveChatHistory() {
        try {
            const history = await this.chatProvider.getHistory();
            const historyPath = path.join(this.context.globalStorageUri.fsPath, 'chat-history.json');
            
            // Ensure directory exists
            const dir = path.dirname(historyPath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
            
            fs.writeFileSync(historyPath, JSON.stringify(history, null, 2));
        } catch (error) {
            this.outputChannel.appendLine(`Error saving chat history: ${error}`);
        }
    }
    
    private showSettings() {
        vscode.commands.executeCommand('workbench.action.openSettings', 'monk');
    }
    
    private async restartExtension() {
        await this.deactivate();
        await this.activate();
        vscode.window.showInformationMessage('MONK Extension restarted');
    }
    
    private updateStatusBar() {
        this.statusBarItem.text = '$(robot) MONK';
        
        if (this.personalityStatusItem) {
            this.personalityStatusItem.text = `$(person) ${this.config.defaultPersonality}`;
        }
    }
    
    private onActiveEditorChanged(editor: vscode.TextEditor | undefined) {
        if (editor && this.config.enableTreeSitterParsing) {
            // Parse the newly active file
            this.treeSitterExplorer.parseFile(editor.document.fileName);
        }
        
        // Update context in chat if open
        if (this.chatPanel) {
            this.sendContextToChat();
        }
    }
    
    private onSelectionChanged(event: vscode.TextEditorSelectionChangeEvent) {
        // Update context when selection changes
        if (this.chatPanel) {
            this.sendContextToChat();
        }
    }
    
    public async deactivate() {
        // Clean up resources
        this.statusBarItem.dispose();
        if (this.personalityStatusItem) {
            this.personalityStatusItem.dispose();
        }
        
        this.outputChannel.dispose();
        
        if (this.chatPanel) {
            this.chatPanel.dispose();
        }
        
        if (this.analysisPanel) {
            this.analysisPanel.dispose();
        }
        
        await this.saveChatHistory();
        
        this.outputChannel.appendLine('MONK CLI Extension deactivated');
    }
}

// Extension activation function
export function activate(context: vscode.ExtensionContext) {
    const extension = new MONKExtension(context);
    extension.activate();
    
    // Store extension instance for deactivation
    context.subscriptions.push({
        dispose: () => extension.deactivate()
    });
}

export function deactivate() {
    // Cleanup handled by extension instance
}