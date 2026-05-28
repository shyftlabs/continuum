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

  // Poll for data injection
  useEffect(() => {
    const MAX_ATTEMPTS = 60;
    const DELAY = 150;
    let attempts = 0;

    function pullData() {
      const openai = window.openai || {};
      const toolOutput = openai.toolOutput;

      if (toolOutput && typeof toolOutput === 'object' && Object.keys(toolOutput).length > 0) {
        setItemsData(toolOutput);
        
        // Get session_id from toolOutput (injected by backend)
        const foundSessionId = toolOutput.mcp_session_id ||
                               toolOutput.session_id || 
                               toolOutput.sessionId;
        
        if (foundSessionId) {
          console.log("product-list - Using session_id from toolOutput:", foundSessionId.substring(0, 8) + "...");
          setSessionId(foundSessionId);
          
          // Also set in window.openai.context for other components
          if (!openai.context) {
            openai.context = {};
          }
          openai.context.session_id = foundSessionId;
          openai.context.sessionId = foundSessionId;
          openai.context.mcp_session_id = foundSessionId;
        }
        
        return true;
      }

      if (widgetProps && typeof widgetProps === 'object' && Object.keys(widgetProps).length > 0) {
        setItemsData(widgetProps);
        
        const foundSessionId = widgetProps.mcp_session_id ||
                               widgetProps.session_id || 
                               widgetProps.sessionId;
        
        if (foundSessionId) {
          console.log("product-list - Using session_id from widgetProps:", foundSessionId.substring(0, 8) + "...");
          setSessionId(foundSessionId);
        }
        
        return true;
      }

      if (window.widgetProps && typeof window.widgetProps === 'object' && Object.keys(window.widgetProps).length > 0) {
        const data = window.widgetProps.structuredContent || window.widgetProps;
        setItemsData(data);
        
        const foundSessionId = data.mcp_session_id ||
                               data.session_id || 
                               data.sessionId;
        
        if (foundSessionId) {
          console.log("product-list - Using session_id from window.widgetProps:", foundSessionId.substring(0, 8) + "...");
          setSessionId(foundSessionId);
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
    }
    
    if (!rawItems || !Array.isArray(rawItems)) return [];
    
    // Transform items to places format
    return rawItems.map((item) => {
      // Extract rating from various possible fields
      let rating = 0;
      if (item.rating !== undefined && item.rating !== null) {
        rating = typeof item.rating === 'number' ? item.rating : parseFloat(item.rating) || 0;
      } else if (item.review_average !== undefined && item.review_average !== null) {
        rating = typeof item.review_average === 'number' ? item.review_average : parseFloat(item.review_average) || 0;
      } else if (item.review_rating !== undefined && item.review_rating !== null) {
        rating = typeof item.review_rating === 'number' ? item.review_rating : parseFloat(item.review_rating) || 0;
      } else if (item.star_rating !== undefined && item.star_rating !== null) {
        rating = typeof item.star_rating === 'number' ? item.star_rating : parseFloat(item.star_rating) || 0;
      } else if (item.attributes?.rating !== undefined && item.attributes?.rating !== null) {
        rating = typeof item.attributes.rating === 'number' ? item.attributes.rating : parseFloat(item.attributes.rating) || 0;
      }
      
      return {
        id: item.id || item.product_id || "",
        name: item.name || item.product_name || "Item",
        thumbnail: (item?.image_urls && item?.image_urls[0]) || item?.image || item?.image_url || item?.thumbnail || "https://placehold.co/600x400",
        rating: rating,
        price: item.price || (item.price_cents ? item.price_cents / 100 : null),
        description: item.description || item.short_description || "",
        statusLabel: item.statusLabel || item.status_label || "In stock",
        statusColor: item.statusColor || item.status_color || "success",
      };
    });
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
      const openai = window.openai || {};
      let baseUrl = openai.toolOutput?.apiBaseUrl || 
                   openai.context?.apiBaseUrl ||
                   import.meta.env.VITE_BASE_URL || 
                   'https://api.omnity.shyftops.io';
      // Ensure /api/v1 is in the path
      if (!baseUrl.includes('/api/v1')) {
        baseUrl = `${baseUrl}/api/v1`;
      }
      const apiUrl = `${baseUrl}/sessions/${sessionId}/cart/${cartItemId}`;
      
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
