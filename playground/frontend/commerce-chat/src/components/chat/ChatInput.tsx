import React, { useState, type KeyboardEvent } from 'react';
import { Send } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  disabled = false,
  placeholder = "Type your message...",
}) => {
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim() && !disabled) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex items-end gap-3 p-6 border-t border-border/50 bg-gradient-to-t from-background via-background to-background/95 backdrop-blur-sm">
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
        className="flex-1 min-h-[52px] max-h-32 px-5 py-3.5 rounded-2xl border border-border/50 bg-card/80 text-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary/50 focus:ring-offset-2 focus:border-primary/50 disabled:opacity-50 placeholder:text-muted-foreground transition-all duration-200 shadow-sm hover:shadow-md focus:shadow-lg backdrop-blur-sm"
      />
      <Button
        onClick={handleSend}
        disabled={disabled || !input.trim()}
        size="icon"
        className="h-[52px] w-[52px] rounded-2xl shrink-0 shadow-lg hover:shadow-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <Send className="h-5 w-5" />
      </Button>
    </div>
  );
};
