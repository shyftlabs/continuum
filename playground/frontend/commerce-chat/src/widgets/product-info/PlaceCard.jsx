import React, { useState } from "react";
import { Star, Plus, Minus, ShoppingCart } from "lucide-react";
import { Badge } from "@openai/apps-sdk-ui/components/Badge";

export default function PlaceCard({ place, sessionId, onAddToCart, cartItemId, onUpdateQuantity, quantity = 1 }) {
  if (!place) return null;

  const [isAdding, setIsAdding] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  
  // Show +/- buttons if item is in cart (has cartItemId) and quantity > 0
  const isInCart = !!cartItemId && quantity > 0;

  // Format price with currency
  const formatPrice = (price) => {
    if (!price && price !== 0) return null;
    if (typeof price === 'number') {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 2,
      }).format(price);
    }
    return price;
  };

  const formattedPrice = formatPrice(place.price);
  const roundedRating = Math.round(place.rating || 0);

  const handleDecrease = async (e) => {
    e.stopPropagation();
    
    if (!cartItemId) {
      return;
    }
    
    setIsUpdating(true);
    const newQty = Math.max(0, quantity - 1);
    if (onUpdateQuantity) {
      try {
        await onUpdateQuantity(cartItemId, newQty, place.id);
      } catch (error) {
        // Error handling
      } finally {
        setIsUpdating(false);
      }
    }
  };

  const handleIncrease = async (e) => {
    e.stopPropagation();
    
    if (!cartItemId) {
      return;
    }
    
    setIsUpdating(true);
    const newQty = quantity + 1;
    if (onUpdateQuantity) {
      try {
        await onUpdateQuantity(cartItemId, newQty, place.id);
      } catch (error) {
        // Error handling
      } finally {
        setIsUpdating(false);
      }
    }
  };

  const handleAddToCart = async (e) => {
    e.stopPropagation();
    
    if (!sessionId) {
      alert("Unable to add to cart: No session ID. Please start a new chat to create a session.");
      return;
    }

    if (!place.id) {
      alert("Unable to add to cart: Product ID is missing.");
      return;
    }

    setIsAdding(true);

    try {
      // Call the add_to_cart API - always add 1
      const productId = parseInt(place.id, 10);
      if (isNaN(productId)) {
        throw new Error("Invalid product ID");
      }

      // Use window.openai.requestTool if available, otherwise use fetch
      if (window.openai && window.openai.requestTool) {
        try {
          const result = await window.openai.requestTool({
            name: "add_to_cart",
            arguments: {
              session_id: sessionId,
              product_id: productId,
              qty: 1, // Always add 1
            },
          });
          
          // Extract cart_item_id from result.items
          const cartItemId = result.items?.find(item => 
            String(item.product_id) === String(productId)
          )?.cart_item_id;
          
          // Call the callback if provided
          if (onAddToCart) {
            onAddToCart({
              ...result,
              product_id: place.id,
              product_name: place.name,
              quantity: 1,
              cart_item_id: cartItemId,
            });
          }
        } catch (toolError) {
          throw toolError;
        }
      } else {
        // Fallback: Call HTTP endpoint directly
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
        const apiUrl = `${apiBaseUrl}/sessions/${sessionId}/cart`;
        console.log('PlaceCard - API URL:', apiUrl);
        
        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            product_id: productId,
            qty: 1, // Always add 1
          }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        
        // Extract cart_item_id from result.items
        const cartItemId = result.items?.find(item => 
          String(item.product_id) === String(productId)
        )?.cart_item_id;
        
        // Call the callback if provided
        if (onAddToCart) {
          onAddToCart({
            ...result,
            product_id: place.id,
            product_name: place.name,
            quantity: 1,
            cart_item_id: cartItemId,
          });
        }
      }
    } catch (error) {
      const errorMsg = error.message || 'Unknown error';
      alert(`Failed to add to cart: ${errorMsg}`);
    } finally {
      setIsAdding(false);
    }
  };

  return (
    <div className="min-w-full select-none flex-shrink-0 bg-white rounded-2xl border border-black/10 shadow-lg overflow-hidden">
      <div className="flex gap-6 p-6">
        {/* Big image on left */}
        <div className="w-64 h-64 flex-shrink-0">
          <img
            src={place.thumbnail}
            alt={place.name}
            className="w-full h-full rounded-xl object-cover"
            loading="lazy"
          />
        </div>

        {/* Content on right */}
        <div className="flex-1 flex flex-col justify-between min-w-0">
          <div>
            <div className="flex items-start justify-between gap-4 mb-3">
              <div className="flex-1 min-w-0">
                <h2 className="text-xl font-semibold text-black mb-2">{place.name}</h2>
                {place.description && (
                  <p className="text-sm text-black/70 mb-3 line-clamp-2">{place.description}</p>
                )}
              </div>
              {place.statusLabel && (
                <Badge color={place.statusColor || "success"} size="sm">
                  {place.statusLabel}
                </Badge>
              )}
            </div>

            <div className="flex items-center gap-4 mb-4">
              {formattedPrice && (
                <div className="text-2xl font-bold text-black">{formattedPrice}</div>
              )}
              {place.rating && (
                <div className="flex items-center gap-1">
                  <div className="flex items-center gap-1">
                    {Array.from({ length: 5 }).map((_, index) => (
                      <Star
                        key={`star-${place.id}-${index}`}
                        className={`h-4 w-4 ${
                          index < roundedRating
                            ? "fill-yellow-400 text-yellow-400"
                            : "text-gray-300"
                        }`}
                      />
                    ))}
                  </div>
                  <span className="text-sm text-black/60 ml-1">
                    {place.rating.toFixed(1)}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Add to cart section */}
          <div className="flex items-center gap-3 mt-4">
            {isInCart ? (
              // Show quantity counter if item is in cart
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={handleDecrease}
                  disabled={quantity <= 0 || isUpdating}
                  className="flex items-center justify-center w-10 h-10 rounded-lg bg-black/5 hover:bg-black/10 active:bg-black/15 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  aria-label="Decrease quantity"
                >
                  <Minus className="h-4 w-4 text-black" strokeWidth={2.5} />
                </button>
                <span className="text-lg font-semibold text-black min-w-[3rem] text-center">{quantity}</span>
                <button
                  type="button"
                  onClick={handleIncrease}
                  disabled={isUpdating}
                  className="flex items-center justify-center w-10 h-10 rounded-lg bg-black/5 hover:bg-black/10 active:bg-black/15 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  aria-label="Increase quantity"
                >
                  <Plus className="h-4 w-4 text-black" strokeWidth={2.5} />
                </button>
              </div>
            ) : (
              // Show "Add to cart" button if item is not in cart
              <button
                type="button"
                onClick={handleAddToCart}
                disabled={isAdding || !sessionId}
                className="inline-flex items-center gap-2 rounded-lg bg-[#00274A] text-white px-6 py-3 text-sm font-medium hover:bg-[#003d6b] active:bg-[#001f3a] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ShoppingCart className="h-4 w-4" strokeWidth={2.5} />
                <span>{isAdding ? "Adding..." : "Add to Cart"}</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
