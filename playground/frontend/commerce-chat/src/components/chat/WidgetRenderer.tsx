import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import type { Root } from 'react-dom/client';
import type { WidgetData } from '@/types/chat';

interface WidgetRendererProps {
  widget: WidgetData;
}

/**
 * Extracts widget name from template URL
 * Example: "ui://widget/cart.html" -> "cart"
 */
function extractWidgetName(templateUrl: string): string {
  try {
    // Remove protocol prefix
    const withoutProtocol = templateUrl.replace(/^ui:\/\/widget\//, '');
    // Extract filename without extension
    const filename = withoutProtocol.split('/').pop() || '';
    return filename.replace(/\.(html|jsx|tsx)$/, '');
  } catch {
    return '';
  }
}

/**
 * Maps widget names from template URLs to actual directory names
 * Handles variations like "products-list" -> "product-list"
 */
function normalizeWidgetName(widgetName: string): string {
  const nameMap: Record<string, string> = {
    // Product widgets
    'product': 'product-info',        // Single product detail (ui://widget/product.html)
    'product-info': 'product-info',
    'product-detail': 'product-info',
    'products': 'products',           // Products carousel
    'product-list': 'product-list',   // Product list with carousel
    'products-list': 'product-list',
    // Category & Store widgets
    'category-list': 'category-list',
    'store-search': 'store-search',
    'stores': 'store-search',         // Alias for store search
    // Appointment widgets
    'book-appointment': 'book-appointment',
    'appointment-booking': 'book-appointment',  // Backend uses this name
    'appointment-list': 'appointment-list',
    'appointments': 'appointment-list',         // Alias
    // Order & Cart widgets
    'pizzaz-order': 'pizzaz-order',
    'order': 'pizzaz-order',                    // Alias
    'cart': 'cart',
    'shopping-cart': 'cart',                    // Alias
  };
  
  return nameMap[widgetName] || widgetName;
}

/**
 * Dynamically loads and renders a widget component
 * Uses React's createRoot to properly render widgets with their data
 */
export const WidgetRenderer: React.FC<WidgetRendererProps> = ({ widget }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rootRef = useRef<Root | null>(null);
  const [error, setError] = useState<string | null>(null);
  const extractedName = extractWidgetName(widget.templateUrl);
  const widgetName = normalizeWidgetName(extractedName);

  useEffect(() => {
    if (!containerRef.current || !widgetName) {
      console.log(`WidgetRenderer [${widgetName}] - Missing container or widgetName`);
      return;
    }

    const container = containerRef.current;
    console.log(`WidgetRenderer [${widgetName}] - Loading widget...`);
    console.log(`WidgetRenderer [${widgetName}] - Template URL: ${widget.templateUrl}`);

    // Initialize window.openai if it doesn't exist
    if (typeof window !== 'undefined') {
      if (!window.openai) {
        window.openai = {
          theme: 'light',
          userAgent: {
            device: { type: 'unknown' },
            capabilities: { hover: true, touch: false },
          },
          locale: 'en',
          maxHeight: 600,
          displayMode: 'inline',
          safeArea: {
            insets: { top: 0, bottom: 0, left: 0, right: 0 },
          },
          toolInput: {},
          toolOutput: null,
          toolResponseMetadata: widget.meta || null,
          widgetState: null,
          setWidgetState: async () => {},
          callTool: async () => ({ result: '' }),
          sendFollowUpMessage: async () => {},
          openExternal: () => {},
          requestDisplayMode: async () => ({ mode: 'inline' }),
        } as any;
      }

      // Set toolOutput to the structured content - widgets read this
      window.openai.toolOutput = widget.structuredContent;
      window.openai.toolResponseMetadata = widget.meta || null;
      
      // Also store as widgetProps for direct access
      (window as any).widgetProps = {
        structuredContent: widget.structuredContent,
        meta: widget.meta,
      };
      
      // Get session_id from structured content (backend-provided)
      const sessionId = widget.structuredContent?.mcp_session_id ||
                        widget.structuredContent?.session_id || 
                        widget.structuredContent?.sessionId;
      
      if (sessionId) {
        console.log(`WidgetRenderer [${widgetName}] - session_id: ${sessionId.substring(0, 8)}...`);
        
        // Set in context for widgets to access
        const openaiAny = window.openai as any;
        if (!openaiAny.context) {
          openaiAny.context = {};
        }
        openaiAny.context.session_id = sessionId;
        openaiAny.context.sessionId = sessionId;
        openaiAny.context.mcp_session_id = sessionId;
      }
      
      // Log the data being passed
      console.log(`WidgetRenderer [${widgetName}] - Data:`, widget.structuredContent);
    }

    // Dynamically import and render the widget component
    const loadWidget = async () => {
      try {
        let WidgetComponent: React.ComponentType<any> | null = null;
        
        console.log(`WidgetRenderer [${widgetName}] - Importing module...`);
        
        // Import the widget module and get the App component
        switch (widgetName) {
          case 'cart': {
            const module = await import('@/widgets/cart/index.jsx');
            WidgetComponent = module.App;
            break;
          }
          case 'product-list': {
            const module = await import('@/widgets/product-list/index.jsx');
            WidgetComponent = module.App;
            break;
          }
          case 'product-info': {
            const module = await import('@/widgets/product-info/index.jsx');
            WidgetComponent = module.App || module.default;
            break;
          }
          case 'products': {
            const module = await import('@/widgets/products/index.jsx');
            WidgetComponent = module.App;
            break;
          }
          case 'category-list': {
            const module = await import('@/widgets/category-list/index.jsx');
            WidgetComponent = module.App || module.default;
            break;
          }
          case 'store-search': {
            const module = await import('@/widgets/store-search/index.jsx');
            WidgetComponent = module.App || module.default;
            break;
          }
          case 'book-appointment': {
            const module = await import('@/widgets/book-appointment/index.jsx');
            WidgetComponent = module.App || module.default;
            break;
          }
          case 'appointment-list': {
            const module = await import('@/widgets/appointment-list/index.jsx');
            WidgetComponent = module.App || module.default;
            break;
          }
          case 'pizzaz-order': {
            const module = await import('@/widgets/pizzaz-order/index.jsx');
            WidgetComponent = module.App || module.default;
            break;
          }
          default:
            console.error(`WidgetRenderer - Unknown widget: ${widgetName}`);
            throw new Error(`Unknown widget: ${widgetName}. Extracted from: ${widget.templateUrl}`);
        }

        if (!WidgetComponent) {
          throw new Error(`Widget ${widgetName} does not export App or default component`);
        }

        console.log(`WidgetRenderer [${widgetName}] - Component loaded, rendering...`);

        // Clean up existing root before creating new one
        if (rootRef.current) {
          rootRef.current.unmount();
          rootRef.current = null;
        }
        
        // Create a new React root and render the widget
        rootRef.current = createRoot(container);
        rootRef.current.render(<WidgetComponent />);
        console.log(`WidgetRenderer [${widgetName}] - Rendered successfully`);
        
      } catch (err) {
        console.error(`Failed to load widget ${widgetName}:`, err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    };

    loadWidget();

    // Cleanup
    return () => {
      if (rootRef.current) {
        rootRef.current.unmount();
        rootRef.current = null;
      }
    };
  // Re-run when widget name changes or when structured content changes
  // We use JSON.stringify to do a deep comparison since structuredContent is an object
  }, [widgetName, widget.templateUrl, JSON.stringify(widget.structuredContent)]);

  if (!widgetName) {
    return (
      <div className="p-4 bg-yellow-50 border border-yellow-200 rounded">
        <p className="text-yellow-600">
          Invalid widget template URL: {widget.templateUrl}
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded">
        <p className="text-red-600">Failed to load widget: {widgetName}</p>
        <p className="text-red-500 text-sm mt-2">{error}</p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="widget-container my-4 w-full"
      data-widget-name={widgetName}
    />
  );
};
