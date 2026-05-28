import React, { useMemo, useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { useWidgetProps } from "../use-widget-props";
import { useMaxHeight } from "../use-max-height";
import { useDisplayMode } from "../use-display-mode";

const STATUS_COLORS = {
  pending: '#f79009',
  processing: '#0ba5ec',
  shipped: '#7f56d9',
  delivered: '#12b76a',
  cancelled: '#f04438'
};

function OrderApp() {
  const maxHeight = useMaxHeight() ?? undefined;
  const displayMode = useDisplayMode();
  const widgetProps = useWidgetProps(() => ({}));
  const [orderData, setOrderData] = useState(null);

  // Poll for data injection from window.openai.toolOutput
  useEffect(() => {
    const MAX_ATTEMPTS = 60;
    const DELAY = 150;
    let attempts = 0;

    function pullData() {
      const openai = window.openai || {};
      const toolOutput = openai.toolOutput;

      if (toolOutput && typeof toolOutput === 'object' && Object.keys(toolOutput).length > 0) {
        setOrderData(toolOutput);
        return true;
      }

      if (widgetProps && typeof widgetProps === 'object' && Object.keys(widgetProps).length > 0) {
        const data = widgetProps.structuredContent || widgetProps;
        setOrderData(data);
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
        setOrderData({});
      }
    }

    hydrate();
  }, [widgetProps]);

  // Use orderData or fallback to empty object
  const orderDataFinal = orderData || widgetProps?.structuredContent || widgetProps || {};
  const orderId = orderDataFinal.id || orderDataFinal.order_id;
  const items = Array.isArray(orderDataFinal.items) ? orderDataFinal.items : [];
  const total = orderDataFinal.total || (orderDataFinal.total_cents ? orderDataFinal.total_cents / 100 : 0);
  const subtotal = orderDataFinal.subtotal || (orderDataFinal.subtotal_cents ? orderDataFinal.subtotal_cents / 100 : total);
  const taxes = total - subtotal;
  const status = (orderDataFinal.status || 'processing').toLowerCase();
  const statusColor = STATUS_COLORS[status] || STATUS_COLORS.processing;

  const priceFormatter = useMemo(
    () => new Intl.NumberFormat('en-CA', { style: 'currency', currency: 'CAD' }),
    []
  );

  if (!orderId) {
    return (
      <div className="w-full max-w-2xl mx-auto px-4 py-12">
        <div className="text-center text-black/60">No order data available.</div>
      </div>
    );
  }

  return (
    <div
      className="w-full max-w-2xl mx-auto bg-white rounded-2xl shadow-lg overflow-hidden"
      style={{ maxHeight }}
    >
      {/* Header */}
      <div className="bg-[#00274A] text-white px-6 py-5">
        <h1 className="text-xl font-semibold m-0">Your PETCO order summary</h1>
        <div className="text-sm opacity-85 mt-1">
          Tracking, delivery, and invoice details below
        </div>
      </div>

      {/* Order Body */}
      <div className="px-6 py-5">
        {/* Order Meta */}
        <div className="flex flex-wrap gap-4 mb-4 text-sm text-black/60">
          <div>Order #{orderId}</div>
          {orderDataFinal.created_at && (
            <div>{new Date(orderDataFinal.created_at).toLocaleString()}</div>
          )}
          <div
            className="px-3 py-1 rounded-full font-semibold text-xs uppercase tracking-wide"
            style={{ background: `${statusColor}20`, color: statusColor }}
          >
            {status}
          </div>
          {orderDataFinal.fulfillment_type && (
            <div>{orderDataFinal.fulfillment_type.replace('_', ' ')}</div>
          )}
        </div>

        {/* Order Items */}
        <div className="border border-black/10 rounded-xl overflow-hidden mb-5">
          {items.length > 0 ? (
            items.map((item, index) => {
              const priceEach = item.price || (item.price_cents ? item.price_cents / 100 : 0);
              const lineTotal = priceEach * (item.qty || 0);
              return (
                <div
                  key={index}
                  className="grid grid-cols-[1fr_auto] gap-3 px-5 py-4 border-b border-black/10 last:border-0"
                >
                  <div>
                    <div className="font-semibold text-sm text-black mb-1">
                      {item.product_name || 'Item'}
                    </div>
                    <div className="text-xs text-black/60">
                      Qty {item.qty || 0} · {priceFormatter.format(priceEach)}
                    </div>
                  </div>
                  <div className="text-sm text-black/60">
                    {priceFormatter.format(lineTotal)}
                  </div>
                </div>
              );
            })
          ) : (
            <div className="px-5 py-8 text-center text-black/60">
              No items in order.
            </div>
          )}
        </div>

        {/* Order Totals */}
        <div className="space-y-2 pt-5">
          <div className="flex justify-between text-sm text-black/70">
            <span>Subtotal</span>
            <span>{priceFormatter.format(subtotal)}</span>
          </div>
          <div className="flex justify-between text-sm text-black/70">
            <span>Taxes & fees</span>
            <span>{priceFormatter.format(taxes)}</span>
          </div>
          <div className="flex justify-between text-xl font-bold text-[#00274A] pt-2 border-t border-black/10">
            <span>Total paid</span>
            <span>{priceFormatter.format(total)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export { OrderApp as App };

const rootElement = document.getElementById("pizzaz-order-root");
if (rootElement) {
  createRoot(rootElement).render(<OrderApp />);
}

