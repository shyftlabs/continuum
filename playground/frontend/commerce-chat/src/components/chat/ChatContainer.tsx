import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { Button } from '@/components/ui/button';
import { Plus, UserPlus, Brain, Users, Network, Clock } from 'lucide-react';
import type { ChatMessage as ChatMessageType } from '@/types/chat';
import { v4 as uuidv4 } from 'uuid';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8088';

// Generate a new random user_id on each frontend startup
// This ensures each session gets a fresh user ID
const generateNewUserId = (): string => {
  const userId = `user_${uuidv4().substring(0, 8)}`;
  console.log(`🆔 Generated new user_id: ${userId}`);
  return userId;
};

interface Agent {
  id: string;
  name: string;
  description: string;
  icon: string;
  color: string;
  api_base?: string;
  endpoints?: {
    chat?: string;
    chat_stream?: string;
    info?: string;
    clear_session?: string;
  };
  features?: {
    streaming?: boolean;
    widgets?: boolean;
    session?: boolean;
    multi_agent?: boolean;
  };
}

interface MemoryInfo {
  enabled: boolean;
  isolation_mode: string | null;
  mode_display_name: string;
  provider?: string;
  search_limit?: number;
}

interface ChatContainerProps {}

export const ChatContainer: React.FC<ChatContainerProps> = () => {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  // Generate a new user ID on component mount (frontend startup)
  const [userId, setUserId] = useState<string>(() => generateNewUserId());
  const [memoryInfo, setMemoryInfo] = useState<MemoryInfo | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load agents from agents.json
  useEffect(() => {
    const loadAgents = async () => {
      try {
        // Try to load from public folder first, then fallback to ui folder
        const response = await fetch('/agents.json').catch(() => 
          fetch('../ui/agents.json')
        );
        if (response && response.ok) {
          const data = await response.json();
          const loadedAgents = data.agents || [];
          setAgents(loadedAgents);
          
          // Set default agent from config or first agent
          const defaultAgentId = data.ui_config?.default_agent || loadedAgents[0]?.id;
          const defaultAgent = loadedAgents.find((a: Agent) => a.id === defaultAgentId) || loadedAgents[0];
          if (defaultAgent) {
            setSelectedAgent(defaultAgent);
          }
        } else {
          // Fallback: use default agent
          const fallbackAgent: Agent = {
            id: 'petco',
            name: 'Petco Agent',
            description: 'AI-powered shopping assistant for Petco',
            icon: '🛒',
            color: '#FF6B35',
            api_base: API_BASE_URL,
            endpoints: {
              chat: '/chat',
              chat_stream: '/chat/stream',
              info: '/info',
              clear_session: '/session'
            }
          };
          setAgents([fallbackAgent]);
          setSelectedAgent(fallbackAgent);
        }
      } catch (error) {
        console.error('Failed to load agents:', error);
        // Fallback: use default agent
        const fallbackAgent: Agent = {
          id: 'petco',
          name: 'Petco Agent',
          description: 'AI-powered shopping assistant for Petco',
          icon: '🛒',
          color: '#FF6B35',
          api_base: API_BASE_URL,
          endpoints: {
            chat: '/chat',
            chat_stream: '/chat/stream',
            info: '/info',
            clear_session: '/session'
          }
        };
        setAgents([fallbackAgent]);
        setSelectedAgent(fallbackAgent);
      }
    };
    loadAgents();
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Fetch memory info when agent is selected
  useEffect(() => {
    const fetchMemoryInfo = async () => {
      if (!selectedAgent) return;
      
      try {
        const agentApiBase = selectedAgent.api_base || API_BASE_URL;
        const response = await fetch(`${agentApiBase}/memory/info`);
        if (response.ok) {
          const data = await response.json();
          setMemoryInfo(data);
          console.log('Memory info:', data);
        }
      } catch (error) {
        console.error('Failed to fetch memory info:', error);
      }
    };
    
    fetchMemoryInfo();
  }, [selectedAgent]);

  const createNewChat = async () => {
    if (!memoryInfo || !memoryInfo.enabled) {
      // If memory disabled, just clear messages
      setMessages([]);
      return;
    }
    
    const isolationMode = memoryInfo.isolation_mode;
    
    if (isolationMode === 'run') {
      // RUN mode: Create new session (no previous memory)
      try {
        const agentApiBase = selectedAgent?.api_base || API_BASE_URL;
        const response = await fetch(`${agentApiBase}/session/new?user_id=${userId}`, {
          method: 'POST',
        });
        if (response.ok) {
          console.log('New session created for RUN mode');
        }
      } catch (error) {
        console.error('Failed to create new session:', error);
      }
    } else {
      // USER/AGENT/SHARED: Just clear session context, keep user_id
      try {
        const agentApiBase = selectedAgent?.api_base || API_BASE_URL;
        const clearEndpoint = selectedAgent?.endpoints?.clear_session || '/session';
        await fetch(`${agentApiBase}${clearEndpoint}?user_id=${userId}`, {
          method: 'DELETE',
        });
      } catch (error) {
        console.error('Failed to clear session:', error);
      }
    }
    
    // Clear local messages
    setMessages([]);
    
    // Clear window.openai context (for widgets)
    if (typeof window !== 'undefined' && window.openai) {
      (window.openai as any).context = {};
      (window.openai as any).toolOutput = null;
    }
    
    console.log('New chat started - session cleared');
  };

  const createNewUser = async () => {
    // Only show for USER mode
    const newUserId = generateNewUserId();
    setUserId(newUserId);
    setMessages([]);
    
    // Clear session
    try {
      const agentApiBase = selectedAgent?.api_base || API_BASE_URL;
      const clearEndpoint = selectedAgent?.endpoints?.clear_session || '/session';
      await fetch(`${agentApiBase}${clearEndpoint}?user_id=${newUserId}`, {
        method: 'DELETE',
      });
    } catch (error) {
      console.error('Failed to clear session:', error);
    }
    
    // Clear window.openai context
    if (typeof window !== 'undefined' && window.openai) {
      (window.openai as any).context = {};
      (window.openai as any).toolOutput = null;
    }
    
    console.log(`🆔 New user created: ${newUserId}`);
  };

  const extractWidgetName = (templateUrl: string): string => {
    try {
      const withoutProtocol = templateUrl.replace(/^ui:\/\/widget\//, '');
      const filename = withoutProtocol.split('/').pop() || '';
      return filename.replace(/\.(html|jsx|tsx)$/, '');
    } catch {
      return '';
    }
  };

  const extractWidgetsFromResponse = (runArtifacts: any, fallbackSessionId: string): any[] => {
    if (!runArtifacts || !runArtifacts.tool_artifacts) {
      return [];
    }

    const widgets: any[] = [];

    for (const artifact of runArtifacts.tool_artifacts) {
      if (artifact.meta && artifact.meta['openai/outputTemplate']) {
        const templateUrl = artifact.meta['openai/outputTemplate'];
        const widgetName = extractWidgetName(templateUrl);

        if (widgetName && artifact.structured_content) {
          // Use MCP session_id from backend if available, otherwise fallback
          const structuredContent = { ...artifact.structured_content };
          
          // Prefer MCP session_id (from create_session tool) over SDK session_id
          const mcpSessionId = structuredContent.mcp_session_id || 
                              structuredContent.session_id || 
                              structuredContent.sessionId;
          const effectiveSessionId = mcpSessionId || fallbackSessionId;
          
          // Always set all three variants for compatibility
          structuredContent.session_id = effectiveSessionId;
          structuredContent.sessionId = effectiveSessionId;
          structuredContent.mcp_session_id = effectiveSessionId;
          
          console.log(`ChatContainer - Widget session_id: ${effectiveSessionId.substring(0, 8)}... (MCP: ${!!mcpSessionId})`);
          
          widgets.push({
            widgetName,
            templateUrl,
            structuredContent,
            meta: artifact.meta,
          });
        }
      }
    }

    return widgets;
  };

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return;

    // Add user message
    const userMessage: ChatMessageType = {
      id: uuidv4(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    // Assistant message will be created when first content arrives
    const assistantMessageId = uuidv4();
    let assistantMessageCreated = false;

    try {
      // Get endpoint from selected agent or use default
      if (!selectedAgent) {
        throw new Error('No agent selected');
      }
      
      const agentApiBase = selectedAgent.api_base || API_BASE_URL;
      const streamEndpoint = selectedAgent.endpoints?.chat_stream || '/chat/stream';
      const endpoint = `${agentApiBase}${streamEndpoint}`;
      
      console.log(`Using endpoint: ${endpoint} for agent: ${selectedAgent.name}`);
      
      // Use streaming endpoint
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: content,
          user_id: userId,
          // NO session_id - backend handles it!
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      // Handle SSE streaming
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let sessionId: string | null = null;
      let accumulatedContent = '';

      if (!reader) {
        throw new Error('Response body is not readable');
      }

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          // Process any remaining data in buffer
          if (buffer.trim()) {
            const lines = buffer.split('\n');
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.slice(6));
                  processSSEData(data);
                } catch (parseError) {
                  console.error('Error parsing SSE data:', parseError, line);
                }
              }
            }
          }
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        
        // SSE format: events are separated by double newlines (\n\n)
        const events = buffer.split('\n\n');
        buffer = events.pop() || ''; // Keep incomplete event in buffer

        for (const event of events) {
          if (!event.trim()) continue;
          
          const lines = event.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                processSSEData(data);
              } catch (parseError) {
                console.error('Error parsing SSE data:', parseError, line);
              }
            }
          }
        }
      }

      function processSSEData(data: any) {
        if (data.type === 'start') {
          sessionId = data.session_id || null;
          if (sessionId) {
            console.log(`ChatContainer - Backend session_id: ${sessionId.substring(0, 8)}...`);
          }
        } else if (data.type === 'session_update') {
          // Update session_id if MCP session_id becomes available
          if (data.mcp_session_id) {
            sessionId = data.mcp_session_id;
            if (sessionId) {
              console.log(`ChatContainer - Updated to MCP session_id: ${sessionId.substring(0, 8)}...`);
            }
          } else if (data.session_id) {
            sessionId = data.session_id;
            if (sessionId) {
              console.log(`ChatContainer - Updated session_id: ${sessionId.substring(0, 8)}...`);
            }
          }
        } else if (data.type === 'content') {
          // Accumulate content chunks
          accumulatedContent += data.content || '';
          
          // Create assistant message on first content chunk
          if (!assistantMessageCreated && accumulatedContent.trim()) {
            assistantMessageCreated = true;
            const assistantMessage: ChatMessageType = {
              id: assistantMessageId,
              role: 'assistant',
              content: accumulatedContent,
              timestamp: new Date(),
            };
            setMessages((prev) => [...prev, assistantMessage]);
            setIsLoading(false); // Hide loading indicator once we have content
          } else if (assistantMessageCreated) {
            // Update the assistant message with accumulated content
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId
                  ? { ...msg, content: accumulatedContent }
                  : msg
              )
            );
          }
        } else if (data.type === 'artifacts') {
          // Extract widgets with session_id
          // Use sessionId from start event (which should be MCP session_id if available)
          const effectiveSessionId = sessionId || '';
          const widgets = extractWidgetsFromResponse(data.artifacts, effectiveSessionId);
          
          // Create assistant message if it doesn't exist yet (for widgets-only responses)
          if (!assistantMessageCreated) {
            assistantMessageCreated = true;
            const assistantMessage: ChatMessageType = {
              id: assistantMessageId,
              role: 'assistant',
              content: accumulatedContent || '',
              widgets: widgets.length > 0 ? widgets : undefined,
              timestamp: new Date(),
            };
            setMessages((prev) => [...prev, assistantMessage]);
            setIsLoading(false);
          } else {
            // Update the assistant message with widgets
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId
                  ? { ...msg, widgets: widgets.length > 0 ? widgets : undefined }
                  : msg
              )
            );
          }
        } else if (data.type === 'done') {
          // Streaming complete - ensure loading is off
          setIsLoading(false);
          console.log('Streaming completed');
        } else if (data.type === 'error') {
          throw new Error(data.error || 'Unknown error');
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setIsLoading(false);
      
      // Create error message if assistant message wasn't created yet
      if (!assistantMessageCreated) {
        const errorMessage: ChatMessageType = {
          id: assistantMessageId,
          role: 'assistant',
          content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMessage]);
      } else {
        // Update existing assistant message with error
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? {
                  ...msg,
                  content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
                }
              : msg
          )
        );
      }
    }
  };

  return (
    <div className="flex h-screen bg-background">
      {/* Left Sidebar */}
      <div className="w-64 border-r border-border/50 bg-gradient-to-b from-card via-card to-card/95 backdrop-blur-sm flex flex-col shadow-lg">
        {/* Cortex Layer Branding */}
        <div className="p-6 border-b border-border/50 bg-gradient-to-r from-primary/5 via-primary/5 to-transparent">
          <h1 className="text-xl font-bold text-foreground tracking-tight bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text">
            Cortex Layer
          </h1>
          <p className="text-xs text-muted-foreground mt-1.5 font-medium">AI Agent Platform</p>
        </div>

        {/* Agents List */}
        <div className="flex-1 overflow-y-auto p-4">
          <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4 px-1">
            Available Agents
          </h2>
          <div className="space-y-2.5">
            {agents.length === 0 ? (
              <div className="text-xs text-muted-foreground text-center py-8">
                <div className="animate-pulse">Loading agents...</div>
              </div>
            ) : (
              agents.map((agent) => {
                const isSelected = selectedAgent?.id === agent.id;
                return (
                <div
                  key={agent.id}
                    onClick={() => {
                      setSelectedAgent(agent);
                      console.log(`Selected agent: ${agent.name} (${agent.id})`);
                    }}
                    className={`p-3.5 rounded-xl border transition-all duration-200 cursor-pointer group ${
                      isSelected
                        ? 'border-primary/50 bg-gradient-to-br from-primary/10 via-primary/5 to-transparent shadow-md shadow-primary/10'
                        : 'border-border/50 bg-background/50 hover:bg-accent/30 hover:border-border hover:shadow-sm'
                    }`}
                >
                  <div className="flex items-center gap-3">
                    <div 
                      className="text-2xl transition-transform duration-200 group-hover:scale-110"
                      style={{ color: agent.color }}
                    >
                      {agent.icon}
                    </div>
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                      <div className={`text-sm font-semibold truncate transition-colors ${
                        isSelected ? 'text-primary' : 'text-foreground'
                      }`}>
                        {agent.name}
                          </div>
                          {isSelected && (
                            <div className="w-2 h-2 rounded-full bg-primary animate-pulse shadow-sm shadow-primary/50" />
                          )}
                      </div>
                      <div className="text-xs text-muted-foreground truncate mt-0.5">
                        {agent.description}
                        </div>
                        {agent.features?.multi_agent && (
                          <div className="text-xs text-primary mt-1.5 font-semibold px-1.5 py-0.5 rounded bg-primary/10 inline-block">
                            Multi-Agent
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-background">
        {/* Header with Memory Badge */}
        {selectedAgent && (
          <div className="border-b border-border bg-gradient-to-r from-card via-card to-card/95 backdrop-blur-sm">
            <div className="px-6 py-4 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-3">
                  <div 
                    className="text-2xl"
                    style={{ color: selectedAgent.color }}
                  >
                    {selectedAgent.icon}
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-foreground">
                      {selectedAgent.name}
                    </h2>
                    <p className="text-xs text-muted-foreground">
                      {userId.substring(0, 12)}...
                    </p>
                  </div>
                </div>
                
                {/* Memory Mode Badge */}
                {memoryInfo && memoryInfo.enabled && (
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 border border-primary/20">
                    {memoryInfo.isolation_mode === 'user' && <UserPlus className="h-3.5 w-3.5 text-primary" />}
                    {memoryInfo.isolation_mode === 'run' && <Clock className="h-3.5 w-3.5 text-primary" />}
                    {memoryInfo.isolation_mode === 'agent' && <Network className="h-3.5 w-3.5 text-primary" />}
                    {memoryInfo.isolation_mode === 'shared' && <Users className="h-3.5 w-3.5 text-primary" />}
                    {!memoryInfo.isolation_mode && <Brain className="h-3.5 w-3.5 text-primary" />}
                    <span className="text-xs font-medium text-primary">
                      {memoryInfo.mode_display_name}
                    </span>
                  </div>
                )}
              </div>
              
              {/* Action Buttons */}
              <div className="flex items-center gap-2">
                {memoryInfo && memoryInfo.enabled && memoryInfo.isolation_mode === 'user' && (
                  <Button
                    onClick={createNewUser}
                    variant="outline"
                    size="sm"
                    className="gap-2"
                  >
                    <UserPlus className="h-4 w-4" />
                    New User
                  </Button>
                )}
                <Button
                  onClick={createNewChat}
                  variant="default"
                  size="sm"
                  className="gap-2"
                >
                  <Plus className="h-4 w-4" />
                  New Chat
                </Button>
              </div>
            </div>
          </div>
        )}
        
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full -mt-16 animate-in fade-in">
              <div className="relative mb-8">
                <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-primary/5 rounded-3xl blur-2xl"></div>
                <img 
                  src="/assests/petco.png" 
                  alt="Petco" 
                  className="relative w-48 h-48 object-contain opacity-90"
                  onError={(e) => {
                    // Fallback if image doesn't load
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              </div>
              <p className="text-xl font-bold text-foreground mb-2 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text">
                Welcome to Petco Chat
              </p>
              <p className="text-sm text-muted-foreground max-w-md text-center">
                Start a conversation by typing a message below
              </p>
            </div>
          )}
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {isLoading && (
            <div className="flex justify-start mb-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
              <div className="max-w-[80%] bg-gradient-to-br from-card via-card/95 to-card/80 text-foreground rounded-2xl px-6 py-4 border border-border/50 shadow-xl backdrop-blur-sm relative overflow-hidden">
                {/* Animated background gradient */}
                <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-primary/10 to-primary/5 animate-shimmer opacity-50"></div>
                
                <div className="relative flex items-center gap-4">
                  {/* Animated icon with pulse effect */}
                  <div className="relative flex-shrink-0">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary/30 via-primary/20 to-primary/10 border-2 border-primary/40 flex items-center justify-center shadow-lg shadow-primary/20">
                      {selectedAgent && (
                        <span className="text-lg" style={{ color: selectedAgent.color }}>
                          {selectedAgent.icon}
                        </span>
                      )}
                    </div>
                    {/* Pulsing rings */}
                    <div className="absolute inset-0 w-10 h-10 rounded-full border-2 border-primary/30 animate-ping"></div>
                    <div className="absolute inset-0 w-10 h-10 rounded-full border-2 border-primary/20 animate-ping" style={{ animationDelay: '0.5s' }}></div>
                  </div>
                  
                  {/* Animated typing indicator */}
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center gap-1.5">
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms', animationDuration: '1.4s' }} />
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '200ms', animationDuration: '1.4s' }} />
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '400ms', animationDuration: '1.4s' }} />
                    </div>
                    <p className="text-xs text-muted-foreground font-semibold tracking-wide">
                      <span className="inline-block animate-pulse">Processing</span>
                      <span className="inline-block animate-pulse" style={{ animationDelay: '0.2s' }}>.</span>
                      <span className="inline-block animate-pulse" style={{ animationDelay: '0.4s' }}>.</span>
                      <span className="inline-block animate-pulse" style={{ animationDelay: '0.6s' }}>.</span>
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Agent Selection Warning */}
        {!selectedAgent && agents.length > 0 && (
          <div className="px-6 py-3 bg-gradient-to-r from-yellow-500/10 via-yellow-500/5 to-transparent border-t border-yellow-500/20 backdrop-blur-sm">
            <p className="text-sm text-yellow-600 dark:text-yellow-400 font-medium">
              Please select an agent from the sidebar to start chatting.
            </p>
          </div>
        )}

        {/* Input */}
        <ChatInput 
          onSendMessage={handleSendMessage} 
          disabled={isLoading || !selectedAgent} 
          placeholder={!selectedAgent ? "Select an agent to start chatting..." : undefined}
        />
      </div>
    </div>
  );
};
