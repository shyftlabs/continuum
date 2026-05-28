import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { ProductCard, CARD_WIDTH } from "./ProductCard";
import { useWidgetProps } from "../use-widget-props";

const initializeQuantities = (items) =>
  items.reduce(
    (acc, item) => {
      acc[item.id] = acc[item.id] ?? 1;
      return acc;
    },
    {},
  );

const CARDS_PER_SLIDE = 3;
const CARD_GAP = 16; // gap-4 = 16px

function App() {
  const widgetProps = useWidgetProps(() => ({}));
  const [itemsData, setItemsData] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [addedToCart, setAddedToCart] = useState(null);
  const [productQuantities, setProductQuantities] = useState({}); // Map product.id -> quantity
  const [productErrors, setProductErrors] = useState({}); // Map product.id -> error message
  const [cartItemIds, setCartItemIds] = useState({}); // Map product.id -> cart_item_id

  // Poll for data injection and sessionId
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
          console.log("products - Using MCP session_id from context:", mcpSessionIdFromContext.substring(0, 8) + "...");
          setSessionId(mcpSessionIdFromContext);
        }
      }

      if (toolOutput && typeof toolOutput === 'object' && Object.keys(toolOutput).length > 0) {
        setItemsData(toolOutput);
        
        // Only use sessionId from toolOutput if we don't have one from context
        if (!mcpSessionIdFromContext) {
          const foundSessionId = toolOutput.mcp_session_id ||
                                 toolOutput.sessionId || 
                                 toolOutput.session_id || 
                                 toolOutput.session;
          
          if (foundSessionId) {
            console.log("products - Using session_id from toolOutput:", foundSessionId.substring(0, 8) + "...");
            setSessionId(foundSessionId);
          }
        }
        
        return true;
      }

      if (widgetProps && typeof widgetProps === 'object' && Object.keys(widgetProps).length > 0) {
        setItemsData(widgetProps);
        
        if (!mcpSessionIdFromContext) {
          const foundSessionId = widgetProps.mcp_session_id ||
                                 widgetProps.sessionId || 
                                 widgetProps.session_id || 
                                 widgetProps.session;
          
          if (foundSessionId) {
            setSessionId(foundSessionId);
          }
        }
        
        return true;
      }

      // Also check window.widgetProps if available
      if (window.widgetProps && typeof window.widgetProps === 'object' && Object.keys(window.widgetProps).length > 0) {
        const data = window.widgetProps.structuredContent || window.widgetProps;
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

  // Transform injected data to products format
  const products = useMemo(() => {
    if (!itemsData) {
      return [];
    }

    let rawProducts = null;

    // Check for different data formats
    if (Array.isArray(itemsData)) {
      rawProducts = itemsData;
    } else if (Array.isArray(itemsData.items)) {
      rawProducts = itemsData.items;
    } else if (Array.isArray(itemsData.products)) {
      rawProducts = itemsData.products;
    }

    if (rawProducts && Array.isArray(rawProducts) && rawProducts.length > 0) {
      // Normalize products to ensure they have the correct structure
      return rawProducts.map((item, index) => {
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
          id: item.id || item.product_id || `product-${index}`,
          category: item.category || "",
          name: item.name || item.product_name || "Product",
          statusLabel: item.statusLabel || item.status_label || "In stock",
          statusColor: item.statusColor || item.status_color || "success",
          price: item.price ||(item.price_cents ?'$' + item.price_cents / 100 : null)|| "$0",
          rating: rating,
          reviews: item.review_count || item.reviews || item.review_count || 0,
          description: item.description || item.short_description || "",
          image: item.image || item.image_url || item.thumbnail || (item?.image_urls && item?.image_urls[0]) || "https://placehold.co/600x400",
        };
      });
    }

    return [];
  }, [itemsData]);

  // Custom carousel state
  const [currentPage, setCurrentPage] = useState(0);
  const pageCount = Math.ceil(products.length / CARDS_PER_SLIDE);

  // Calculate which products to show on current page
  const startIndex = currentPage * CARDS_PER_SLIDE;
  const endIndex = startIndex + CARDS_PER_SLIDE;
  const visibleProducts = products.slice(startIndex, endIndex);

  const canScrollPrev = currentPage > 0;
  const canScrollNext = currentPage < pageCount - 1;

  const scrollPrev = () => {
    if (canScrollPrev) {
      setCurrentPage((prev) => Math.max(0, prev - 1));
    }
  };

  const scrollNext = () => {
    if (canScrollNext) {
      setCurrentPage((prev) => Math.min(pageCount - 1, prev + 1));
    }
  };

  const scrollTo = (pageIndex) => {
    setCurrentPage(Math.max(0, Math.min(pageCount - 1, pageIndex)));
  };


  // Initialize quantities for products not yet in cart
  useEffect(() => {
    setProductQuantities((prev) => {
      const base = initializeQuantities(products);
      Object.keys(base).forEach((id) => {
        if (!(id in prev)) {
          base[id] = 1;
        } else {
          base[id] = prev[id];
        }
      });
      return base;
    });
  }, [products]);

  const handleAddToCart = (result) => {
    const productId = result?.product_id || result?.id;
    const productName = result?.product_name || result?.name || "Product";
    const qty = result?.quantity || result?.qty || 1;
    
    // Extract cart_item_id from result.items or direct result
    const cartItemId = result?.cart_item_id || 
                      result?.items?.find(item => 
                        String(item.product_id) === String(productId)
                      )?.cart_item_id;
    
    // Clear any errors for this product on successful add
    if (productId) {
      handleProductError(productId, null);
    }
    
    setAddedToCart({ name: productName, quantity: qty });
    
    // Store cart_item_id and set quantity to 1 for the added product
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
    
    // Hide notification after 3 seconds
    setTimeout(() => {
      setAddedToCart(null);
    }, 3000);
  };

  const handleProductError = (productId, errorMessage) => {
    if (errorMessage === null || errorMessage === undefined) {
      // Clear error
      setProductErrors((prev) => {
        const updated = { ...prev };
        delete updated[productId];
        return updated;
      });
    } else {
      // Set error
      setProductErrors((prev) => ({
        ...prev,
        [productId]: errorMessage,
      }));
    }
  };

  const handleUpdateQuantity = async (cartItemId, newQty, productId) => {
    if (!sessionId) {
      handleProductError(productId, "No session ID available");
      return;
    }

    if (!productId) {
      return;
    }

    if (!cartItemId) {
      handleProductError(productId, "Cart item ID is missing");
      return;
    }

    // Clear any previous error
    handleProductError(productId, null);

    try {
      // Call API to update quantity using cart_item_id (like cart widget)
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
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.message || `HTTP ${response.status}: ${response.statusText}`;
        handleProductError(productId, errorMsg);
        throw new Error(errorMsg);
      }

      // API call successful - update local quantity
      handleProductError(productId, null);
      
      // If quantity is 0, remove cartItemId to show "Add to cart" button again
      if (newQty === 0) {
        setCartItemIds((prev) => {
          const updated = { ...prev };
          delete updated[productId];
          return updated;
        });
        setProductQuantities((prev) => ({
          ...prev,
          [productId]: 1, // Reset to 1 for next add
        }));
      } else {
        // Update local quantity
        setProductQuantities((prev) => ({
          ...prev,
          [productId]: newQty,
        }));
      }
    } catch (error) {
      handleProductError(productId, error.message || 'Unknown error');
    }
  };

  if (!products.length) {
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
              <p className="text-sm opacity-90">
                {addedToCart.quantity > 1 
                  ? `${addedToCart.quantity}x ${addedToCart.name}`
                  : addedToCart.name}
              </p>
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

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-black/60 text-sm">Featured products</p>
          <h1 className="text-2xl font-semibold text-black">PETCO</h1>
        </div>
      </div>

      <div className="relative w-full">
        {/* Previous/Next buttons in the middle */}
        <div className="absolute left-0 top-1/2 -translate-y-1/2 z-10 -translate-x-4">
          <button
            type="button"
            onClick={scrollPrev}
            disabled={!canScrollPrev}
            aria-label="Previous products"
            className="flex items-center justify-center w-10 h-10 rounded-full bg-black/20 backdrop-blur-sm hover:bg-black/30 active:bg-black/40 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="h-4 w-4 text-white" strokeWidth={2.5} />
          </button>
        </div>
        <div className="absolute right-0 top-1/2 -translate-y-1/2 z-10 translate-x-4">
          <button
            type="button"
            onClick={scrollNext}
            disabled={!canScrollNext}
            aria-label="Next products"
            className="flex items-center justify-center w-10 h-10 rounded-full bg-black/20 backdrop-blur-sm hover:bg-black/30 active:bg-black/40 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronRight className="h-4 w-4 text-white" strokeWidth={2.5} />
          </button>
        </div>

        <div 
          className="overflow-hidden rounded-3xl border border-transparent mx-auto" 
          style={{
            width: `${(CARDS_PER_SLIDE * CARD_WIDTH) + ((CARDS_PER_SLIDE - 1) * CARD_GAP)}px`,
            maxWidth: '100%',
          }}
        >
          <div className="flex transition-transform duration-500 ease-out">
            {visibleProducts.map((product) => (
              <div
                key={product.id}
                className="flex-shrink-0"
                  style={{
                  width: `${CARD_WIDTH}px`,
                  marginRight: `${CARD_GAP}px`,
                  }}
                >
                    <ProductCard
                      product={product}
                      quantity={productQuantities[product.id] ?? 1}
                      sessionId={sessionId}
                      onAddToCart={handleAddToCart}
                      onUpdateQuantity={handleUpdateQuantity}
                      errorMessage={productErrors[product.id]}
                      onError={(errorMsg) => handleProductError(product.id, errorMsg)}
                      cartItemId={cartItemIds[product.id]}
                    />
              </div>
            ))}
          </div>
        </div>

        <div className="mt-4 flex items-center justify-center">
          <div className="flex gap-1.5">
            {Array.from({ length: pageCount }).map((_, index) => (
              <button
                key={`dot-${index}`}
                type="button"
                  aria-label={`Go to page ${index + 1}`}
                  onClick={() => scrollTo(index)}
                className={`h-2.5 rounded-full transition-all ${
                    currentPage === index ? "w-6 bg-black" : "w-2.5 bg-gray-300"
                }`}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

export { App };

const rootElement = document.getElementById("products-root");
if (rootElement) {
  createRoot(rootElement).render(<App />);
}
