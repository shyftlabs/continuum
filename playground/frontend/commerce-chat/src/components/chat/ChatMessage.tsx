import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatMessage as ChatMessageType } from '@/types/chat';
import { WidgetRenderer } from './WidgetRenderer';

interface ChatMessageProps {
  message: ChatMessageType;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const hasWidgets = message.widgets && message.widgets.length > 0;
  const isEmpty = !message.content || message.content.trim() === '';

  // Don't render empty assistant messages (they're handled by loading indicator)
  if (!isUser && isEmpty && !hasWidgets) {
    return null;
  }

  return (
    <div className="mb-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
      {/* Message bubble */}
      <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
        <div
          className={`max-w-[80%] rounded-2xl px-5 py-3.5 transition-all duration-200 ${
            isUser
              ? 'bg-gradient-to-br from-primary to-primary/90 text-primary-foreground shadow-lg shadow-primary/20 hover:shadow-xl hover:shadow-primary/30'
              : 'bg-card text-foreground border border-border/50 shadow-md hover:shadow-lg backdrop-blur-sm'
          }`}
        >
          {!isEmpty && (
          <div className="text-sm leading-relaxed markdown-content">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                em: ({ children }) => <em className="italic">{children}</em>,
                ul: ({ children }) => <ul className="list-disc mb-2 ml-4 space-y-1.5">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal mb-2 ml-4 space-y-1.5">{children}</ol>,
                li: ({ children }) => <li className="pl-1">{children}</li>,
                h1: ({ children }) => <h1 className="text-lg font-bold mb-2 mt-3 first:mt-0">{children}</h1>,
                h2: ({ children }) => <h2 className="text-base font-bold mb-2 mt-3 first:mt-0">{children}</h2>,
                h3: ({ children }) => <h3 className="text-sm font-bold mb-1 mt-2 first:mt-0">{children}</h3>,
                code: ({ children, className }) => {
                  const isInline = !className;
                  return isInline ? (
                    <code className="bg-black/10 dark:bg-white/10 px-1.5 py-0.5 rounded text-xs font-mono">{children}</code>
                  ) : (
                    <code className={className}>{children}</code>
                  );
                },
                pre: ({ children }) => (
                  <pre className="bg-black/10 dark:bg-white/10 p-3 rounded-lg text-xs font-mono overflow-x-auto mb-2">
                    {children}
                  </pre>
                ),
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-current/20 pl-4 my-2 italic">
                    {children}
                  </blockquote>
                ),
                hr: () => <hr className="my-3 border-t border-current/20" />,
                a: ({ children, href }) => (
                  <a href={href} className="underline hover:no-underline" target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
          )}
        </div>
      </div>
      
      {/* Render widgets OUTSIDE the message bubble for full width */}
      {hasWidgets && (
        <div className="mt-4 space-y-4 w-full">
          {message.widgets!.map((widget, index) => (
            <WidgetRenderer key={`${widget.widgetName}-${index}`} widget={widget} />
          ))}
        </div>
      )}
    </div>
  );
};
