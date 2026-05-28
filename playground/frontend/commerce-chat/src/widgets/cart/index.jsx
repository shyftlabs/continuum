import React, { useEffect, useMemo, useState, useCallback } from "react";
import { createRoot } from "react-dom/client";
import { Minus, Plus, ShoppingCart, Trash2 } from "lucide-react";
import { Badge } from "@openai/apps-sdk-ui/components/Badge";
import { Button } from "@openai/apps-sdk-ui/components/Button";
import { EmptyMessage } from "@openai/apps-sdk-ui/components/EmptyMessage";
import { useWidgetProps } from "../use-widget-props";

const fallbackCartItems = [];

function CartItemRow({ item, onUpdateQuantity, onRemove, sessionId, onError }) {
  const priceEach = item.price || (item.price_cents ? item.price_cents / 100 : 0);
  const lineTotal = priceEach * (item.qty || 0);
  const [isUpdating, setIsUpdating] = useState(false);
  const priceFormatter = useMemo(
    () => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }),
    []
  );

  const handleDecrease = async (e) => {
    e.stopPropagation();
    e.preventDefault();
    console.log("handleDecrease clicked, current qty:", item.qty);
    
    if (item.qty <= 1) {
      console.log("Cannot decrease, qty is already 1");
      if (onError) onError("Cannot decrease quantity below 1");
      return;
    }
    
    setIsUpdating(true);
    try {
      const newQty = item.qty - 1;
      console.log("Calling onUpdateQuantity with:", { itemId: item.cart_item_id || item.product_id, newQty });
      
      if (onUpdateQuantity) {
        await onUpdateQuantity(item.cart_item_id || item.product_id, newQty);
        if (onError) onError(null); // Clear error on success
      } else {
        if (onError) onError("onUpdateQuantity is not defined!");
      }
    } catch (error) {
      if (onError) onError(`Error updating quantity: ${error.message || error}`);
    } finally {
      setIsUpdating(false);
    }
  };

  const handleIncrease = async (e) => {
    e.stopPropagation();
    e.preventDefault();
    console.log("handleIncrease clicked, current qty:", item.qty);
    
    setIsUpdating(true);
    try {
      const newQty = item.qty + 1;
      console.log("Calling onUpdateQuantity with:", { itemId: item.cart_item_id || item.product_id, newQty });
      
      if (onUpdateQuantity) {
        await onUpdateQuantity(item.cart_item_id || item.product_id, newQty);
        if (onError) onError(null); // Clear error on success
      } else {
        if (onError) onError("onUpdateQuantity is not defined!");
      }
    } catch (error) {
      if (onError) onError(`Error updating quantity: ${error.message || error}`);
    } finally {
      setIsUpdating(false);
    }
  };

  const handleRemove = async (e) => {
    e.stopPropagation();
    e.preventDefault();
    console.log("handleRemove clicked");
    
    setIsUpdating(true);
    try {
      const itemId = item.cart_item_id || item.product_id;
      console.log("Calling onRemove with:", { itemId });
      
      if (onRemove) {
        await onRemove(itemId);
        if (onError) onError(null); // Clear error on success
      } else {
        if (onError) onError("onRemove is not defined!");
      }
    } catch (error) {
      if (onError) onError(`Error removing item: ${error.message || error}`);
    } finally {
      setIsUpdating(false);
    }
  };

  return (
    <div className="flex items-center gap-4 p-4 border-b border-black/10 last:border-0 bg-white hover:bg-gray-50 transition-colors">
      {/* Image */}
      <div className="w-24 h-24 flex-shrink-0 overflow-hidden rounded-lg">
        <img 
          src={item.image_url || item.image || item.thumbnail || ""} 
          alt={item.product_name || item.name || "Product"} 
          className="w-full h-full object-contain" 
          loading="lazy" 
        />
      </div>

      {/* Product Info */}
      <div className="flex-1 min-w-0">
        <h3 className="text-base font-semibold text-black truncate mb-1">
          {item.product_name || item.name || "Item"}
        </h3>
        <div className="flex items-center gap-2 mb-2">
          {item.fulfillment_type && (
            <Badge 
              color={item.fulfillment_type === "pickup" ? "info" : "success"} 
              size="sm"
            >
              {item.fulfillment_type.replace('_', ' ').toUpperCase()}
            </Badge>
          )}
          <span className="text-sm text-black/60">
            {priceFormatter.format(priceEach)} each
          </span>
        </div>
      </div>

      {/* Quantity Controls */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleDecrease}
            disabled={item.qty <= 1 || isUpdating}
            className="flex items-center justify-center w-8 h-8 rounded-lg bg-black/5 hover:bg-black/10 active:bg-black/15 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            aria-label="Decrease quantity"
          >
            <Minus className="h-3.5 w-3.5 text-black" strokeWidth={2.5} />
          </button>
          <span className="text-base font-semibold text-black min-w-[2rem] text-center">
            {item.qty || 0}
          </span>
          <button
            type="button"
            onClick={handleIncrease}
            disabled={isUpdating}
            className="flex items-center justify-center w-8 h-8 rounded-lg bg-black/5 hover:bg-black/10 active:bg-black/15 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label="Increase quantity"
          >
            <Plus className="h-3.5 w-3.5 text-black" strokeWidth={2.5} />
          </button>
        </div>
      </div>

      {/* Price */}
      <div className="text-right min-w-[100px]">
        <div className="text-lg font-semibold text-black">
          {priceFormatter.format(lineTotal)}
        </div>
      </div>

      {/* Remove Button */}
      <button
        type="button"
        onClick={handleRemove}
        disabled={isUpdating}
        className="flex items-center justify-center w-10 h-10 rounded-lg text-red-600 hover:bg-red-50 active:bg-red-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        aria-label="Remove item"
      >
        <Trash2 className="h-4 w-4" strokeWidth={2.5} />
      </button>
    </div>
  );
}

function App() {
  const widgetProps = useWidgetProps(() => ({}));
  const [cartData, setCartData] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // Poll for data injection and sessionId
  useEffect(() => {
    const MAX_ATTEMPTS = 60;
    const DELAY = 150;
    let attempts = 0;

    function pullData() {
      const openai = window.openai || {};
      const toolOutput = openai.toolOutput;

      console.log("Cart - pullData attempt:", attempts);
      console.log("Cart - toolOutput:", toolOutput);
      console.log("Cart - window.openai.context:", openai.context);

      // PRIORITY 1: Get MCP session_id from window.openai.context (set by WidgetRenderer)
      // This is the MCP tool session_id, not the chat session_id
      // ALWAYS check context first and use it if available
      let mcpSessionIdFromContext = null;
      if (openai.context) {
        mcpSessionIdFromContext = openai.context.mcp_session_id || 
                                  openai.context.sessionId || 
                                  openai.context.session_id;
        if (mcpSessionIdFromContext) {
          console.log("Cart - ✅ Found MCP sessionId in context:", mcpSessionIdFromContext);
          setSessionId(mcpSessionIdFromContext);
        }
      }

      if (toolOutput && typeof toolOutput === 'object' && Object.keys(toolOutput).length > 0) {
        console.log("Cart - Found data in toolOutput:", toolOutput);
        setCartData(toolOutput);
        
        // Only use sessionId from toolOutput if we don't have one from context
        // (context has the MCP session_id, which is what we need)
        // Check the actual value, not the state variable (which might be stale)
        if (!mcpSessionIdFromContext) {
          // Check for MCP session_id in toolOutput first
          const mcpSessionIdFromOutput = toolOutput.mcp_session_id;
          const foundSessionId = mcpSessionIdFromOutput ||
                                 toolOutput.sessionId || 
                                 toolOutput.session_id || 
                                 toolOutput.session;
          
          if (foundSessionId) {
            console.log("Cart - ⚠️ Found sessionId in toolOutput (fallback):", foundSessionId);
            // Only use if it's the MCP one, not the chat one
            if (mcpSessionIdFromOutput || foundSessionId.length > 20) {
              // MCP session_ids are typically longer UUIDs
              setSessionId(foundSessionId);
            } else {
              console.warn("Cart - ⚠️ Ignoring short session_id from toolOutput (likely chat session_id):", foundSessionId);
            }
          }
        } else {
          console.log("Cart - ✅ Using MCP sessionId from context, ignoring toolOutput sessionId");
        }
        
        return true;
      }

      if (widgetProps && typeof widgetProps === 'object' && Object.keys(widgetProps).length > 0) {
        console.log("Cart - Found data in widgetProps:", widgetProps);
        const data = widgetProps.structuredContent || widgetProps;
        setCartData(data);
        
        // Only use sessionId from widgetProps if we don't have MCP one from context
        if (!mcpSessionIdFromContext) {
          const mcpSessionIdFromProps = data.mcp_session_id;
          const foundSessionId = mcpSessionIdFromProps ||
                                data.sessionId || 
                                data.session_id || 
                                data.session;
          
          if (foundSessionId) {
            // Prefer MCP session_id
            if (mcpSessionIdFromProps || (foundSessionId.length > 20)) {
              console.log("Cart - Found sessionId in widgetProps:", foundSessionId);
              setSessionId(foundSessionId);
            } else {
              console.warn("Cart - Ignoring short session_id from widgetProps (likely chat session_id):", foundSessionId);
            }
          }
        }
        
        return true;
      }

      // Also check window.widgetProps if available
      if (window.widgetProps && typeof window.widgetProps === 'object' && Object.keys(window.widgetProps).length > 0) {
        console.log("Cart - Found data in window.widgetProps:", window.widgetProps);
        const data = window.widgetProps.structuredContent || window.widgetProps;
        setCartData(data);
        
        // Only use sessionId from window.widgetProps if we don't have MCP one from context
        if (!mcpSessionIdFromContext) {
          const mcpSessionIdFromWindow = data.mcp_session_id;
          const foundSessionId = mcpSessionIdFromWindow ||
                                data.sessionId || 
                                data.session_id || 
                                data.session;
          
          if (foundSessionId) {
            // Prefer MCP session_id
            if (mcpSessionIdFromWindow || (foundSessionId.length > 20)) {
              console.log("Cart - Found sessionId in window.widgetProps:", foundSessionId);
              setSessionId(foundSessionId);
            } else {
              console.warn("Cart - Ignoring short session_id from window.widgetProps (likely chat session_id):", foundSessionId);
            }
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
        console.warn("Cart - No data found after", MAX_ATTEMPTS, "attempts, using fallback");
        setCartData({});
      }
    }

    hydrate();
  }, [widgetProps]);

  // Continuously check for MCP session_id from context (set by WidgetRenderer)
  // This ensures we get the MCP session_id even if it's set after initial render
  useEffect(() => {
    const checkInterval = setInterval(() => {
      if (window.openai && window.openai.context) {
        const mcpSessionId = window.openai.context.sessionId || window.openai.context.session_id;
        if (mcpSessionId && mcpSessionId !== sessionId) {
          console.log("Cart - Updated MCP sessionId from context:", mcpSessionId);
          setSessionId(mcpSessionId);
        }
      }
    }, 500); // Check every 500ms

    return () => clearInterval(checkInterval);
  }, [sessionId]);

  // Transform cart data to items format
  const items = useMemo(() => {
    if (!cartData) {
      return fallbackCartItems;
    }

    console.log("Cart - Processing cartData:", cartData);

    let rawItems = null;

    // Check for different data formats
    if (Array.isArray(cartData)) {
      rawItems = cartData;
      console.log("Cart - Found direct array:", rawItems.length);
    } else if (Array.isArray(cartData.items)) {
      rawItems = cartData.items;
      console.log("Cart - Found items array:", rawItems.length);
    } else if (Array.isArray(cartData.cart_items)) {
      rawItems = cartData.cart_items;
      console.log("Cart - Found cart_items array:", rawItems.length);
    }

    if (rawItems && Array.isArray(rawItems) && rawItems.length > 0) {
      return rawItems.map((item, index) => {
        // Extract price - try multiple field names and formats
        let price = 0;
        let price_cents = 0;
        
        // Try price in dollars first
        if (item.price != null && item.price > 0) {
          price = item.price;
          price_cents = Math.round(price * 100);
        }
        // Try price_cents
        else if (item.price_cents != null && item.price_cents > 0) {
          price_cents = item.price_cents;
          price = price_cents / 100;
        }
        // Try unit_price
        else if (item.unit_price != null && item.unit_price > 0) {
          price = item.unit_price;
          price_cents = Math.round(price * 100);
        }
        // Try unit_price_cents
        else if (item.unit_price_cents != null && item.unit_price_cents > 0) {
          price_cents = item.unit_price_cents;
          price = price_cents / 100;
        }
        // Try line_total / qty (if line_total exists)
        else if (item.line_total != null && item.line_total > 0 && (item.qty || item.quantity)) {
          const qty = item.qty || item.quantity || 1;
          price = item.line_total / qty;
          price_cents = Math.round(price * 100);
        }
        // Try line_total_cents / qty
        else if (item.line_total_cents != null && item.line_total_cents > 0 && (item.qty || item.quantity)) {
          const qty = item.qty || item.quantity || 1;
          price_cents = item.line_total_cents;
          price = price_cents / qty;
        }
        
        console.log(`Cart - Item ${index} price extraction:`, {
          item,
          extractedPrice: price,
          extractedPriceCents: price_cents,
        });
        
        return {
        cart_item_id: item.cart_item_id || item.id || `cart-item-${index}`,
        product_id: item.product_id || item.id || `product-${index}`,
        product_name: item.product_name || item.name || "Item",
        name: item.name || item.product_name || "Item",
          price: price,
          price_cents: price_cents,
        qty: item.qty || item.quantity || 1,
        image_url: item.image_url || item.image || item.thumbnail || "",
        fulfillment_type: item.fulfillment_type || item.fulfillment || null,
        };
      });
    }

    // Fallback to default items
    console.log("Cart - Using fallback items");
    return fallbackCartItems;
  }, [cartData]);

  // Calculate totals
  const { subtotal, taxes, total } = useMemo(() => {
    // Calculate subtotal from items
    const calculatedSubtotal = items.reduce((sum, item) => {
      const price = item.price || (item.price_cents ? item.price_cents / 100 : 0);
      return sum + (price * (item.qty || 0));
    }, 0);
    
    // Use subtotal from cartData if available and valid, otherwise use calculated
    // Check for both subtotal and subtotal_cents
    const sub = (cartData?.subtotal != null && cartData.subtotal > 0) 
      ? cartData.subtotal 
      : (cartData?.subtotal_cents != null && cartData.subtotal_cents > 0)
        ? cartData.subtotal_cents / 100
        : calculatedSubtotal;
    
    // Calculate taxes
    const calculatedTax = sub * 0.13; // Default 13% tax
    const tax = (cartData?.taxes != null && cartData.taxes >= 0)
      ? cartData.taxes
      : (cartData?.tax_cents != null && cartData.tax_cents >= 0)
        ? cartData.tax_cents / 100
        : calculatedTax;
    
    // Calculate total - use cartData.total if it's a valid number (not null/undefined/0 when items exist)
    // If total is 0 but we have items with prices, calculate from subtotal + tax
    const calculatedTotal = sub + tax;
    const tot = (cartData?.total != null && cartData.total > 0)
      ? cartData.total
      : (cartData?.total_cents != null && cartData.total_cents > 0)
        ? cartData.total_cents / 100
        : calculatedTotal;
    
    console.log("Cart - Totals calculation:", {
      calculatedSubtotal,
      cartDataSubtotal: cartData?.subtotal,
      cartDataSubtotalCents: cartData?.subtotal_cents,
      finalSubtotal: sub,
      calculatedTax,
      cartDataTax: cartData?.taxes,
      cartDataTaxCents: cartData?.tax_cents,
      finalTax: tax,
      calculatedTotal,
      cartDataTotal: cartData?.total,
      cartDataTotalCents: cartData?.total_cents,
      finalTotal: tot,
      itemsCount: items.length,
    });
    
    return {
      subtotal: sub,
      taxes: tax,
      total: tot,
    };
  }, [items, cartData]);

  const priceFormatter = useMemo(
    () => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }),
    []
  );

  const handleUpdateQuantity = useCallback(async (itemId, newQty) => {
    // Always check for MCP session_id from context first (most up-to-date)
    const openai = window.openai || {};
    const currentSessionId = (openai.context && (openai.context.mcp_session_id || openai.context.sessionId || openai.context.session_id)) || sessionId;
    
    console.log("handleUpdateQuantity called:", { itemId, newQty, sessionId, currentSessionId, context: openai.context });
    
    if (!currentSessionId) {
      setErrorMessage("No session ID available");
      return;
    }
    
    // Update state if we found a different session_id in context
    if (currentSessionId !== sessionId) {
      console.log("Cart - Updating sessionId from context:", currentSessionId);
      setSessionId(currentSessionId);
    }

    setErrorMessage(null); // Clear previous errors

    try {
      // Call API first and wait for success
      const openai = window.openai || {};
      let baseUrl = openai.toolOutput?.apiBaseUrl || 
                   openai.context?.apiBaseUrl ||
                   import.meta.env.VITE_BASE_URL || 
                   'https://api.omnity.shyftops.io';
      // Ensure /api/v1 is in the path
      if (!baseUrl.includes('/api/v1')) {
        baseUrl = `${baseUrl}/api/v1`;
      }
      const apiUrl = `${baseUrl}/sessions/${currentSessionId}/cart/${itemId}`;
      
      console.log("Cart - Calling API:", apiUrl, "with payload:", { qty: newQty, sessionId: currentSessionId });
      
      const response = await fetch(apiUrl, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ qty: newQty }),
      });

      console.log("API Response status:", response.status);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.message || `HTTP ${response.status}: ${response.statusText}`;
        setErrorMessage(errorMsg);
        throw new Error(errorMsg);
      }

      // API call successful - API won't return cart data, so update locally
      console.log("Cart item updated successfully via HTTP");
      
      // Clear error on success
      setErrorMessage(null);
      
      // Update local cart data (quantity, price, totals) on client side
      setCartData((prevCartData) => {
        if (!prevCartData) return prevCartData;
        
        // Find the item and update its quantity
        const updatedItems = (prevCartData.items || prevCartData.cart_items || []).map((item) => {
          const itemIdToMatch = item.cart_item_id || item.product_id || item.id;
          if (String(itemIdToMatch) === String(itemId)) {
            return {
              ...item,
              qty: newQty,
            };
          }
          return item;
        });

        // Return updated cart data
        if (Array.isArray(prevCartData)) {
          return updatedItems;
        } else {
          return {
            ...prevCartData,
            items: updatedItems,
            cart_items: updatedItems,
          };
        }
      });
    } catch (error) {
      const errorMsg = `Error updating cart item: ${error.message || 'Unknown error'}`;
      setErrorMessage(errorMsg);
      console.error(errorMsg, error);
    }
  }, [sessionId]);

  const handleRemoveItem = useCallback(async (itemId) => {
    // Always check for MCP session_id from context first (most up-to-date)
    const openai = window.openai || {};
    const currentSessionId = (openai.context && (openai.context.mcp_session_id || openai.context.sessionId || openai.context.session_id)) || sessionId;
    
    console.log("handleRemoveItem called:", { itemId, sessionId, currentSessionId, context: openai.context });
    
    if (!currentSessionId) {
      setErrorMessage("No session ID available");
      return;
    }
    
    // Update state if we found a different session_id in context
    if (currentSessionId !== sessionId) {
      console.log("Cart - Updating sessionId from context:", currentSessionId);
      setSessionId(currentSessionId);
    }

    setErrorMessage(null); // Clear previous errors

    try {
      // Call API first and wait for success
      const openai = window.openai || {};
      let baseUrl = openai.toolOutput?.apiBaseUrl || 
                   openai.context?.apiBaseUrl ||
                   import.meta.env.VITE_BASE_URL || 
                   'https://api.omnity.shyftops.io';
      // Ensure /api/v1 is in the path
      if (!baseUrl.includes('/api/v1')) {
        baseUrl = `${baseUrl}/api/v1`;
      }
      const apiUrl = `${baseUrl}/sessions/${currentSessionId}/cart/${itemId}`;
      
      console.log("Cart - Using session_id for API call:", currentSessionId);
      
      console.log("Calling API to remove:", apiUrl, "with payload:", { qty: 0 });
      
      const response = await fetch(apiUrl, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ qty: 0 }), // Setting qty to 0 removes the item
      });

      console.log("API Response status:", response.status);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.message || `HTTP ${response.status}: ${response.statusText}`;
        setErrorMessage(errorMsg);
        throw new Error(errorMsg);
      }

      // API call successful - API won't return cart data, so update locally
      console.log("Cart item removed successfully via HTTP");
      
      // Clear error on success
      setErrorMessage(null);
      
      // Update local cart data - remove the item
      setCartData((prevCartData) => {
        if (!prevCartData) return prevCartData;
        
        // Filter out the removed item
        const updatedItems = (prevCartData.items || prevCartData.cart_items || []).filter((item) => {
          const itemIdToMatch = item.cart_item_id || item.product_id || item.id;
          return String(itemIdToMatch) !== String(itemId);
        });

        // Return updated cart data
        if (Array.isArray(prevCartData)) {
          return updatedItems;
        } else {
          return {
            ...prevCartData,
            items: updatedItems,
            cart_items: updatedItems,
          };
        }
      });
    } catch (error) {
      const errorMsg = `Error removing cart item: ${error.message || 'Unknown error'}`;
      setErrorMessage(errorMsg);
      console.error(errorMsg, error);
    }
  }, [sessionId]);

  const handleProceedToPayment = useCallback(() => {
    const checkoutUrl = '#';
    window.location.href = checkoutUrl;
  }, [cartData]);

  if (!items.length) {
    return (
      <section className="mx-auto max-w-6xl space-y-5 px-4 py-8">
        <EmptyMessage>
          <EmptyMessage.Icon>
            <ShoppingCart className="h-8 w-8" />
          </EmptyMessage.Icon>
          <EmptyMessage.Title>Your cart is empty</EmptyMessage.Title>
          <EmptyMessage.Description>
            Add items to your cart to get started
          </EmptyMessage.Description>
        </EmptyMessage>
      </section>
    );
  }

  return (
    <section className="mx-auto max-w-6xl space-y-5 px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-black/60 text-sm">Shopping Cart</p>
          <h1 className="text-2xl font-semibold text-black">Your PETCO cart</h1>
          {errorMessage && (
            <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-600 text-sm font-medium">{errorMessage}</p>
            </div>
          )}
        </div>
        <div className="text-sm text-black/60">
          {items.length} {items.length === 1 ? 'item' : 'items'}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_400px] gap-6">
        {/* Cart Items - List View */}
        <div className="bg-white rounded-2xl border border-black/10 shadow-lg overflow-hidden">
          <div className="overflow-x-auto">
            <div className="min-w-full">
              {items.map((item) => (
                <CartItemRow
                  key={item.cart_item_id || item.product_id}
                  item={item}
                  onUpdateQuantity={handleUpdateQuantity}
                  onRemove={handleRemoveItem}
                  sessionId={sessionId}
                  onError={setErrorMessage}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Summary */}
        <div className="lg:sticky lg:top-4 h-fit">
          <div className="rounded-2xl border border-black/10 bg-white shadow-lg p-6 space-y-4">
            <h2 className="text-lg font-semibold text-black mb-4">Order Summary</h2>
            
            <div className="space-y-3">
              <div className="flex justify-between text-base text-black/70">
                <span>Subtotal</span>
                <span>{priceFormatter.format(subtotal)}</span>
              </div>
              <div className="flex justify-between text-base text-black/70">
                <span>Estimated taxes</span>
                <span>{priceFormatter.format(taxes)}</span>
              </div>
              <div className="flex justify-between text-xl font-bold text-black pt-3 border-t border-black/10">
                <span>Estimated total</span>
                <span>{priceFormatter.format(total)}</span>
              </div>
            </div>

            <Button
              onClick={handleProceedToPayment}
              color="primary"
              size="lg"
              block
            >
              Proceed to payment
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}

export { App };

const rootElement = document.getElementById("cart-root");
if (rootElement) {
  createRoot(rootElement).render(<App />);
} else {
  console.error("Cart - Root element 'cart-root' not found!");
}

