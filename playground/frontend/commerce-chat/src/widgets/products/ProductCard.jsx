import React, { useState } from "react";
import { Minus, Plus, Star } from "lucide-react";
import { Badge } from "@openai/apps-sdk-ui/components/Badge";

export const CARD_WIDTH = 240;

// Map status colors to OpenAI SDK Badge color prop
const statusColorMap = {
  success: "success",
  info: "info",
  warning: "warning",
  danger: "danger",
};

export function ProductCard({ product, quantity, sessionId, onAddToCart, onUpdateQuantity, errorMessage, onError, cartItemId }) {
  const { category, name, statusLabel, statusColor, price, rating, reviews, description, image } = product;
  const roundedRating = Math.round(rating);
  const badgeColor = statusColorMap[statusColor] || statusColorMap.info;
  const [isAdding, setIsAdding] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  
  // Show +/- buttons if item is in cart (has cartItemId) and quantity > 0
  const isInCart = !!cartItemId && quantity > 0;

  const handleAddToCart = async (e) => {
    e.stopPropagation();
    
    if (!sessionId) {
      alert("Unable to add to cart: No session ID. Please start a new chat to create a session.");
      return;
    }

    if (!product.id) {
      alert("Unable to add to cart: Product ID is missing.");
      return;
    }

    setIsAdding(true);

    try {
      // Call the add_to_cart API - always add 1
      const productId = parseInt(product.id, 10);
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
              product_id: product.id,
              product_name: name,
              quantity: 1, // Always add 1
              cart_item_id: cartItemId,
            });
          }
          
          // Success - callback will handle notification
        } catch (toolError) {
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
        const apiUrl = `${baseUrl}/sessions/${sessionId}/cart`;
        
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
            product_id: product.id,
            product_name: name,
            quantity: 1, // Always add 1
            cart_item_id: cartItemId,
          });
        }
        
        // Success - callback will handle notification
      }
    } catch (error) {
      const errorMsg = error.message || 'Unknown error';
      if (onError) {
        onError(errorMsg);
      } else {
        alert(`Failed to add to cart: ${errorMsg}`);
      }
    } finally {
      setIsAdding(false);
    }
  };

  const handleIncrement = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (!cartItemId) {
      if (onError) {
        onError("Cart item ID is missing. Please try adding to cart again.");
      }
      return;
    }
    
    setIsUpdating(true);
    const newQty = quantity + 1;
    if (onUpdateQuantity) {
      try {
        await onUpdateQuantity(cartItemId, newQty, product.id);
        if (onError) onError(null); // Clear error on success
      } catch (error) {
        if (onError) {
          onError(error.message || "Failed to update quantity");
        }
      } finally {
        setIsUpdating(false);
      }
    }
  };

  const handleDecrement = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (!cartItemId) {
      if (onError) {
        onError("Cart item ID is missing. Please try adding to cart again.");
      }
      return;
    }
    
    setIsUpdating(true);
    const newQty = Math.max(0, quantity - 1);
    if (onUpdateQuantity) {
      try {
        await onUpdateQuantity(cartItemId, newQty, product.id);
        if (onError) onError(null); // Clear error on success
      } catch (error) {
        if (onError) {
          onError(error.message || "Failed to update quantity");
        }
      } finally {
        setIsUpdating(false);
      }
    }
  };

  return (
    <article
      className="rounded-2xl border border-black/10 bg-white shadow-lg p-4"
      style={{ width: `${CARD_WIDTH}px` }}
    >
      <div className="aspect-squre overflow-hidden w-full rounded-2xl flex items-center justify-center">
        <img src={image} alt={name} className=" w-full object-contain h-[160px]" loading="lazy" />
      </div>

      <div className="mt-4 flex items-center justify-between gap-2">
        <p className="text-black/60 text-sm">{category}</p>
        <Badge color={badgeColor} size="sm">
          {statusLabel}
        </Badge>
      </div>

      <h2 className="mt-1 text-md font-semibold text-black max-w-[200px] truncate">{name}</h2>
      <p className="text-sm text-black/70">{description}</p>
      <p className="mt-2 text-lg font-semibold text-black">{price}</p>

      <div className="mt-2 flex items-center gap-2">
        <div className="flex items-center gap-1">
          {Array.from({ length: 5 }).map((_, index) => (
            <Star
              key={`star-${product.id}-${index}`}
              className={`h-3 w-3 ${
                index < roundedRating
                  ? "fill-yellow-400 text-yellow-400"
                  : "text-gray-300"
              }`}
            />
          ))}
        </div>
        <p className="text-sm text-black/60">
          {rating.toFixed(1)} · {reviews} reviews
        </p>
      </div>

      <footer className="mt-5 border-t border-black/10 pt-4">
        {errorMessage && (
          <div className="mb-3 rounded-lg bg-red-50 border border-red-200 p-2.5">
            <p className="text-sm text-red-700 font-medium">Error</p>
            <p className="text-xs text-red-600 mt-0.5">{errorMessage}</p>
            {onError && (
              <button
                type="button"
                onClick={() => onError(null)}
                className="mt-1.5 text-xs text-red-600 hover:text-red-800 underline"
                aria-label="Dismiss error"
              >
                Dismiss
              </button>
            )}
          </div>
        )}
        {isInCart ? (
          // Show quantity counter if item is in cart
          <div className="flex items-center justify-between gap-3 w-full">
            <button
              type="button"
              onClick={handleDecrement}
              disabled={quantity <= 0 || isUpdating}
              className="flex items-center justify-center w-10 h-10 rounded-lg bg-black/5 hover:bg-black/10 active:bg-black/15 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            aria-label={`Decrease quantity for ${name}`}
            >
              <Minus className="h-4 w-4 text-black" strokeWidth={2.5} />
            </button>
            <span className="text-lg font-semibold text-black flex-1 text-center">{quantity}</span>
            <button
              type="button"
              onClick={handleIncrement}
              disabled={isUpdating}
              className="flex items-center justify-center w-10 h-10 rounded-lg bg-black/5 hover:bg-black/10 active:bg-black/15 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label={`Increase quantity for ${name}`}
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
            className="w-full rounded-lg px-4 py-2.5 text-sm font-medium transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed bg-[#00274A] text-white hover:bg-[#003d6b] active:bg-[#001f3a]"
          >
            {isAdding ? "Adding..." : "Add to cart"}
          </button>
        )}
      </footer>
    </article>
  );
}

