import React, { useState } from "react";
import { Star, Plus, Minus, ShoppingCart } from "lucide-react";

export default function PlaceCard({ place, sessionId, onAddToCart }) {
  if (!place) return null;

  const [quantity, setQuantity] = useState(1);
  const [isAdding, setIsAdding] = useState(false);

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

  const handleDecrease = (e) => {
    e.stopPropagation();
    setQuantity((prev) => Math.max(1, prev - 1));
  };

  const handleIncrease = (e) => {
    e.stopPropagation();
    setQuantity((prev) => prev + 1);
  };

  const handleAddToCart = async (e) => {
    e.stopPropagation();
    
    // Use sessionId prop directly - it comes from backend
    console.log("PlaceCard - Using session_id:", sessionId ? sessionId.substring(0, 8) + "..." : "NONE");
    
    if (!sessionId) {
      console.error("No session ID available");
      alert("Unable to add to cart: No session ID. Please refresh and try again.");
      return;
    }

    if (!place.id) {
      console.error("No product ID available");
      alert("Unable to add to cart: Product ID is missing.");
      return;
    }

    setIsAdding(true);

    try {
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
              qty: quantity,
            },
          });
          
          console.log("Add to cart successful via requestTool:", result);
          
          if (onAddToCart) {
            onAddToCart(result);
          }
          
          alert(`Added ${quantity} ${place.name} to cart!`);
        } catch (toolError) {
          console.error("Error calling add_to_cart tool:", toolError);
          throw toolError;
        }
      } else {
        // Fallback: Call HTTP endpoint directly
        const openai = window.openai || {};
        let baseUrl = openai.toolOutput?.apiBaseUrl || 
                     openai.context?.apiBaseUrl ||
                     import.meta.env.VITE_BASE_URL || 
                     'https://api.omnity.shyftops.io';
        // Ensure /api/v1 is in the path
        if (!baseUrl.includes('/api/v1')) {
          baseUrl = `${baseUrl}/api/v1`;
        }
        const apiUrl = baseUrl.includes('/api/v1') 
          ? `${baseUrl}/sessions/${sessionId}/cart` 
          : `${baseUrl}/api/v1/sessions/${sessionId}/cart`;
        
        console.log("PlaceCard - API URL:", apiUrl);
        
        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            product_id: productId,
            qty: quantity,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        console.log("Add to cart successful via HTTP:", result);
        
        if (onAddToCart) {
          onAddToCart(result);
        }
        
        alert(`Added ${quantity} ${place.name} to cart!`);
      }
    } catch (error) {
      console.error("Error adding to cart:", error);
      alert(`Failed to add to cart: ${error.message || 'Unknown error'}`);
    } finally {
      setIsAdding(false);
    }
  };

  return (
    <div className="min-w-[220px] select-none max-w-[220px] w-[65vw] sm:w-[220px] self-stretch flex flex-col bg-white rounded-2xl overflow-hidden hover:shadow-lg transition-shadow duration-200">
      <div className="w-full relative">
        <img
          src={place.thumbnail}
          alt={place.name || "Product"}
          className="w-full aspect-square object-cover"
        />
      </div>
      <div className="p-4 flex flex-col flex-1">
        <div className="text-base font-semibold text-black truncate line-clamp-2 mb-2">
          {place.name}
        </div>
        {formattedPrice && (
          <div className="text-lg font-bold text-black mb-2">
            {formattedPrice}
          </div>
        )}
        {/* Star Rating */}
        {place.rating !== undefined && place.rating !== null && (
          <div className="flex items-center gap-1.5 mb-2">
            <div className="flex items-center gap-0.5">
              {Array.from({ length: 5 }).map((_, index) => {
                const roundedRating = Math.round(place.rating || 0);
                return (
                  <Star
                    key={`star-${place.id}-${index}`}
                    className={`h-3.5 w-3.5 ${
                      index < roundedRating
                        ? "fill-yellow-400 text-yellow-400"
                        : "text-gray-300"
                    }`}
                  />
                );
              })}
            </div>
            {place.rating > 0 && (
              <span className="text-xs text-black/60">
                {typeof place.rating === 'number' ? place.rating.toFixed(1) : place.rating}
              </span>
            )}
          </div>
        )}
        {place.description && (
          <div className="text-sm text-black/70 line-clamp-2 flex-auto mb-3">
            {place.description}
          </div>
        )}
        <div className="flex flex-col gap-2 mt-auto">
          <div className="flex items-center justify-center rounded-lg border border-black/10 bg-white w-full">
            <button
              type="button"
              onClick={handleDecrease}
              disabled={isAdding}
              className="flex items-center justify-center w-8 h-8 rounded-l-lg hover:bg-black/5 active:bg-black/10 transition-colors flex-shrink-0 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Decrease quantity"
            >
              <Minus className="h-3.5 w-3.5 text-black" strokeWidth={2.5} />
            </button>
            <span className="flex-1 text-center text-sm font-medium text-black min-w-[2rem]">
              {quantity}
            </span>
            <button
              type="button"
              onClick={handleIncrease}
              disabled={isAdding}
              className="flex items-center justify-center w-8 h-8 rounded-r-lg hover:bg-black/5 active:bg-black/10 transition-colors flex-shrink-0 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Increase quantity"
            >
              <Plus className="h-3.5 w-3.5 text-black" strokeWidth={2.5} />
            </button>
          </div>
          <button
            type="button"
            onClick={handleAddToCart}
            disabled={isAdding || !sessionId}
            className="w-full cursor-pointer inline-flex items-center justify-center gap-2 rounded-lg bg-[#00274A] text-white px-3 py-2 text-sm font-medium hover:bg-[#003d6b] active:bg-[#001f3a] transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ShoppingCart className="h-4 w-4 flex-shrink-0" strokeWidth={2.5} />
            <span className="truncate">{isAdding ? "Adding..." : "Add to Cart"}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
