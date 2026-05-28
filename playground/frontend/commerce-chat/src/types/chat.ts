export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  widgets?: WidgetData[];
}

export interface WidgetData {
  widgetName: string;
  templateUrl: string;
  structuredContent: Record<string, any>;
  meta?: Record<string, any>;
}

export interface ChatSession {
  id: string;
  createdAt: Date;
  messages: ChatMessage[];
}
