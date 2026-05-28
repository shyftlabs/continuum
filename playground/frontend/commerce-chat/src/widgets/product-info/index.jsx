import React, { useState, useMemo, useEffect } from "react";
import { createRoot } from "react-dom/client";
import useEmblaCarousel from "embla-carousel-react";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { useWidgetProps } from "../use-widget-props";
import PlaceCard from "./PlaceCard";

function App() {
  const widgetProps = useWidgetProps(() => ({}));
  const [itemsData, setItemsData] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [addedToCart, setAddedToCart] = useState(null);
  const [productQuantities, setProductQuantities] = useState({});
  const [cartItemIds, setCartItemIds] = useState({});

  // Poll for data injection (similar to other widgets)
  useEffect(() => {
    const MAX_ATTEMPTS = 60;
    const DELAY = 150;
    let attempts = 0;

    function pullData() {
      const openai = window.openai || {};
      const toolOutput = openai.toolOutput;

      // PRIORITY 1: Get MCP session_id from window.openai.context (set by WidgetRenderer)
      let mcpSessionIdFromContext = null;
      if (openai.context) {
        mcpSessionIdFromContext = openai.context.mcp_session_id || 
                                  openai.context.sessionId || 
                                  openai.context.session_id;
        if (mcpSessionIdFromContext) {
          console.log("product-info - Using MCP session_id from context:", mcpSessionIdFromContext.substring(0, 8) + "...");
          setSessionId(mcpSessionIdFromContext);
        }
      }

      // Handle nested structure: value.structuredContent
      if (toolOutput && typeof toolOutput === 'object') {
        let data = null;
        
        // Check for nested value.structuredContent structure
        if (toolOutput.value && toolOutput.value.structuredContent) {
          data = toolOutput.value.structuredContent;
        } else if (toolOutput.structuredContent) {
          data = toolOutput.structuredContent;
        } else if (Object.keys(toolOutput).length > 0) {
          data = toolOutput;
        }
        
        if (data) {
          setItemsData(data);
          
          // Only use sessionId from data if we don't have one from context
          if (!mcpSessionIdFromContext) {
            const foundSessionId = data.mcp_session_id ||
                                   data.sessionId || 
                                   data.session_id || 
                                   data.session;
            
            if (foundSessionId) {
              console.log("product-info - Using session_id from toolOutput:", foundSessionId.substring(0, 8) + "...");
              setSessionId(foundSessionId);
            }
          }
          
          return true;
        }
      }

      if (widgetProps && typeof widgetProps === 'object' && Object.keys(widgetProps).length > 0) {
        // Handle nested structure in widgetProps too
        let data = widgetProps;
        if (widgetProps.value && widgetProps.value.structuredContent) {
          data = widgetProps.value.structuredContent;
        } else if (widgetProps.structuredContent) {
          data = widgetProps.structuredContent;
        }
        
        setItemsData(data);
        
        if (!mcpSessionIdFromContext) {
          const foundSessionId = data.mcp_session_id ||
                                 data.sessionId || 
                                 data.session_id || 
                                 data.session;
          
          if (foundSessionId) {
            setSessionId(foundSessionId);
          }
        }
        
        return true;
      }

      if (window.widgetProps && typeof window.widgetProps === 'object' && Object.keys(window.widgetProps).length > 0) {
        let data = null;
        
        // Handle nested structure: value.structuredContent
        if (window.widgetProps.value && window.widgetProps.value.structuredContent) {
          data = window.widgetProps.value.structuredContent;
        } else if (window.widgetProps.structuredContent) {
          data = window.widgetProps.structuredContent;
        } else {
          data = window.widgetProps;
        }
        
        setItemsData(data);
        
        if (!mcpSessionIdFromContext) {
          const foundSessionId = data.mcp_session_id ||
                                 data.sessionId || 
                                 data.session_id || 
                                 data.session;
          
          if (foundSessionId) {
            setSessionId(foundSessionId);
          }
        }
        
        return true;
      }
      
      return false;
    }

    function hydrate() {
      if (pullData()) {
        return;
      }
      if (attempts++ < MAX_ATTEMPTS) {
        setTimeout(hydrate, DELAY);
      } else {
        setItemsData({});
      }
    }

    hydrate();
  }, [widgetProps]);

  // Get places from the polled data - transform to places format
  const places = useMemo(() => {
    if (!itemsData) return [];
    
    let rawItems = null;
    if (Array.isArray(itemsData)) {
      rawItems = itemsData;
    } else if (Array.isArray(itemsData.items)) {
      rawItems = itemsData.items;
    } else if (Array.isArray(itemsData.products)) {
      rawItems = itemsData.products;
    } else if (itemsData.id) {
      // Single product object (like the structuredContent structure)
      rawItems = [itemsData];
    }
    
    if (!rawItems || !Array.isArray(rawItems)) return [];
    
    // Transform items to places format
    return rawItems.map((item) => ({
      id: item.id || item.product_id || "",
      name: item.name || item.product_name || "Item",
      thumbnail: item.thumbnail || item.image || item.image_url || (item.image_urls && item.image_urls[0]) || "",
      rating: item.rating || item.review_average || (item.attributes && item.attributes.rating) || 0,
      price: item.price || (item.price_cents ? item.price_cents / 100 : null),
      description: item.description || item.short_description || "",
      statusLabel: item.statusLabel || item.status_label || (item.is_active ? "In stock" : "Out of stock"),
      statusColor: item.statusColor || item.status_color || (item.is_active ? "success" : "error"),
      // Additional fields from the new structure
      available_qty: item.available_qty,
      attributes: item.attributes || {},
      currency: item.currency || "USD",
    }));
  }, [itemsData]);

  const handleAddToCart = (result) => {
    const productId = result?.product_id || result?.id;
    const productName = result?.product_name || result?.name || "Product";
    
    // Extract cart_item_id from result.items
    const cartItemId = result?.cart_item_id || 
                      result?.items?.find(item => 
                        String(item.product_id) === String(productId)
                      )?.cart_item_id;
    
    setAddedToCart({ name: productName, quantity: 1 });
    
    if (productId) {
      if (cartItemId) {
        setCartItemIds((prev) => ({
          ...prev,
          [productId]: cartItemId,
        }));
      }
      setProductQuantities((prev) => ({
        ...prev,
        [productId]: 1,
      }));
    }
    
    setTimeout(() => {
      setAddedToCart(null);
    }, 3000);
  };

  const handleUpdateQuantity = async (cartItemId, newQty, productId) => {
    if (!sessionId || !productId || !cartItemId) {
      return;
    }

    try {
      // Get API base URL from context or use default
      const openai = window.openai || {};
      let apiBaseUrl = openai.toolOutput?.apiBaseUrl || 
                      openai.context?.apiBaseUrl ||
                      import.meta.env.VITE_BASE_URL || 
                      'https://api.omnity.shyftops.io';
      // Ensure /api/v1 is in the path
      if (!apiBaseUrl.includes('/api/v1')) {
        apiBaseUrl = `${apiBaseUrl}/api/v1`;
      }
      const apiUrl = `${apiBaseUrl}/sessions/${sessionId}/cart/${cartItemId}`;
      
      const response = await fetch(apiUrl, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ qty: newQty }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      if (newQty === 0) {
        setCartItemIds((prev) => {
          const updated = { ...prev };
          delete updated[productId];
          return updated;
        });
        setProductQuantities((prev) => ({
          ...prev,
          [productId]: 1,
        }));
      } else {
        setProductQuantities((prev) => ({
          ...prev,
          [productId]: newQty,
        }));
      }
    } catch (error) {
      // Error handling
    }
  };
  const [emblaRef, emblaApi] = useEmblaCarousel({
    align: "center",
    loop: false,
    containScroll: "trimSnaps",
    slidesToScroll: "auto",
    dragFree: false,
  });
  const [canPrev, setCanPrev] = React.useState(false);
  const [canNext, setCanNext] = React.useState(false);

  React.useEffect(() => {
    if (!emblaApi) return;
    const updateButtons = () => {
      setCanPrev(emblaApi.canScrollPrev());
      setCanNext(emblaApi.canScrollNext());
    };
    updateButtons();
    emblaApi.on("select", updateButtons);
    emblaApi.on("reInit", updateButtons);
    return () => {
      emblaApi.off("select", updateButtons);
      emblaApi.off("reInit", updateButtons);
    };
  }, [emblaApi]);

  if (!places.length) {
    return null;
  }

  return (
    <section className="mx-auto max-w-6xl space-y-5 px-4 py-8 w-full">
      {/* Success Notification */}
      {addedToCart && (
        <div className="fixed top-4 right-4 z-50 animate-in slide-in-from-top-2">
          <div className="bg-green-500 text-white px-6 py-4 rounded-lg shadow-lg flex items-center gap-3 min-w-[300px] max-w-[400px]">
            <div className="flex-1">
              <p className="font-semibold">Added to cart!</p>
              <p className="text-sm opacity-90">{addedToCart.name}</p>
            </div>
            <button
              onClick={() => setAddedToCart(null)}
              className="text-white/80 hover:text-white transition-colors flex-shrink-0"
              aria-label="Close notification"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      <div className="relative w-full">
        <div className="overflow-hidden" ref={emblaRef}>
          <div className="flex gap-6">
            {places.map((place) => (
              <PlaceCard
                key={place.id}
                place={place}
                sessionId={sessionId}
                onAddToCart={handleAddToCart}
                cartItemId={cartItemIds[place.id]}
                onUpdateQuantity={handleUpdateQuantity}
                quantity={productQuantities[place.id] ?? 1}
              />
            ))}
          </div>
        </div>
        {/* Previous/Next buttons */}
        {canPrev && (
          <div className="absolute left-0 top-1/2 -translate-y-1/2 z-10 -translate-x-4">
            <button
              aria-label="Previous"
              className="flex items-center justify-center w-10 h-10 rounded-full bg-black/20 backdrop-blur-sm hover:bg-black/30 active:bg-black/40 transition-colors"
              onClick={() => emblaApi && emblaApi.scrollPrev()}
              type="button"
            >
              <ArrowLeft className="h-4 w-4 text-white" strokeWidth={2.5} />
            </button>
          </div>
        )}
        {canNext && (
          <div className="absolute right-0 top-1/2 -translate-y-1/2 z-10 translate-x-4">
            <button
              aria-label="Next"
              className="flex items-center justify-center w-10 h-10 rounded-full bg-black/20 backdrop-blur-sm hover:bg-black/30 active:bg-black/40 transition-colors"
              onClick={() => emblaApi && emblaApi.scrollNext()}
              type="button"
            >
              <ArrowRight className="h-4 w-4 text-white" strokeWidth={2.5} />
            </button>
          </div>
        )}
      </div>
    </section>
  );
}

export { App };

const rootElement = document.getElementById("product-info-root");
if (rootElement) {
  createRoot(rootElement).render(<App />);
}
