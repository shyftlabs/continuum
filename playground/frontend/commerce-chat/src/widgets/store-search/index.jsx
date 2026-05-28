import React, { useEffect, useMemo, useState, useRef } from "react";
import { createRoot } from "react-dom/client";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { Badge } from "@openai/apps-sdk-ui/components/Badge";
import { EmptyMessage } from "@openai/apps-sdk-ui/components/EmptyMessage";
import { useWidgetProps } from "../use-widget-props";
import { ShoppingCart, MapPin } from "lucide-react";

// Set Mapbox access token (you may want to make this configurable)
mapboxgl.accessToken =
  "pk.eyJ1IjoiZXJpY25pbmciLCJhIjoiY21icXlubWM1MDRiczJvb2xwM2p0amNyayJ9.n-3O6JI5nOp_Lw96ZO5vJQ";

const fallbackStores = [
  {
    id: 1,
    name: "Toronto Downtown Store",
    address: {
      street: "100 King St W",
      city: "Toronto",
      state: "ON",
      zip: "M5X1A9",
      country: "Canada",
    },
    latitude: 43.6487,
    longitude: -79.3854,
  },
];

function fitMapToMarkers(map, coords) {
  if (!map || !coords.length) return;
  if (coords.length === 1) {
    map.flyTo({ center: coords[0], zoom: 12 });
    return;
  }
  const bounds = coords.reduce(
    (b, c) => b.extend(c),
    new mapboxgl.LngLatBounds(coords[0], coords[0])
  );
  map.fitBounds(bounds, { padding: 60, animate: true });
}

function App() {
  const widgetProps = useWidgetProps(() => ({}));
  const [itemsData, setItemsData] = useState(null);
  const mapRef = useRef(null);
  const mapObj = useRef(null);
  const markerObjs = useRef([]);
  const [selectedStore, setSelectedStore] = useState(null);

  // Poll for data injection
  useEffect(() => {
    const MAX_ATTEMPTS = 60;
    const DELAY = 150;
    let attempts = 0;

    function pullData() {
      const openai = window.openai || {};
      const toolOutput = openai.toolOutput;

      console.log("StoreSearch - pullData attempt:", attempts);
      console.log("StoreSearch - toolOutput:", toolOutput);

      if (toolOutput && typeof toolOutput === 'object' && Object.keys(toolOutput).length > 0) {
        console.log("StoreSearch - Found data in toolOutput:", toolOutput);
        setItemsData(toolOutput);
        return true;
      }

      if (widgetProps && typeof widgetProps === 'object' && Object.keys(widgetProps).length > 0) {
        console.log("StoreSearch - Found data in widgetProps:", widgetProps);
        setItemsData(widgetProps);
        return true;
      }

      // Also check window.widgetProps if available
      if (window.widgetProps && typeof window.widgetProps === 'object' && Object.keys(window.widgetProps).length > 0) {
        console.log("StoreSearch - Found data in window.widgetProps:", window.widgetProps);
        const data = window.widgetProps.structuredContent || window.widgetProps;
        setItemsData(data);
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
        console.warn("StoreSearch - No data found after", MAX_ATTEMPTS, "attempts, using fallback");
        setItemsData({});
      }
    }

    hydrate();
  }, [widgetProps]);

  // Transform data to stores format
  const stores = useMemo(() => {
    if (!itemsData) {
      return fallbackStores;
    }

    console.log("StoreSearch - Processing itemsData:", itemsData);

    let rawStores = null;

    // Check for different data formats
    if (Array.isArray(itemsData)) {
      rawStores = itemsData;
      console.log("StoreSearch - Found direct array:", rawStores.length);
    } else if (itemsData.structuredContent && Array.isArray(itemsData.structuredContent.items)) {
      rawStores = itemsData.structuredContent.items;
      console.log("StoreSearch - Found structuredContent.items:", rawStores.length);
    } else if (Array.isArray(itemsData.items)) {
      rawStores = itemsData.items;
      console.log("StoreSearch - Found items array:", rawStores.length);
    } else if (Array.isArray(itemsData.stores)) {
      rawStores = itemsData.stores;
      console.log("StoreSearch - Found stores array:", rawStores.length);
    }

    if (rawStores && Array.isArray(rawStores) && rawStores.length > 0) {
      return rawStores.map((item, index) => ({
        id: String(item.id || `store-${index}`),
        name: item.name || item.store_name || "Store",
        address: item.address || {
          street: item.street || "",
          city: item.city || "",
          state: item.state || item.province || "",
          zip: item.zip || item.postal_code || "",
          country: item.country || "",
        },
        latitude: item.latitude || item.lat || 0,
        longitude: item.longitude || item.lng || item.lon || 0,
        distance_meters: item.distance_meters || item.distance || null,
      }));
    }

    // Fallback to default stores
    console.log("StoreSearch - Using fallback stores");
    return fallbackStores;
  }, [itemsData]);

  // Initialize map
  useEffect(() => {
    if (mapObj.current || !mapRef.current || stores.length === 0) return;

    const coords = stores
      .filter((store) => store.latitude && store.longitude)
      .map((store) => [store.longitude, store.latitude]);

    if (coords.length === 0) return;

    const center = coords[0];

    mapObj.current = new mapboxgl.Map({
      container: mapRef.current,
      style: "mapbox://styles/mapbox/streets-v12",
      center: center,
      zoom: coords.length === 1 ? 12 : 10,
      attributionControl: false,
    });

    // Add markers
    addAllMarkers(stores);

    // Fit map to markers after initial load
    setTimeout(() => {
      fitMapToMarkers(mapObj.current, coords);
    }, 100);

    // Handle resize
    const handleResize = () => {
      if (mapObj.current) {
        mapObj.current.resize();
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (mapObj.current) {
        mapObj.current.remove();
        mapObj.current = null;
      }
    };
  }, [stores]);

  // Update markers when stores change
  useEffect(() => {
    if (!mapObj.current || stores.length === 0) return;
    addAllMarkers(stores);
    
    const coords = stores
      .filter((store) => store.latitude && store.longitude)
      .map((store) => [store.longitude, store.latitude]);
    
    if (coords.length > 0) {
      fitMapToMarkers(mapObj.current, coords);
    }
  }, [stores]);

  function addAllMarkers(storesList) {
    // Remove existing markers
    markerObjs.current.forEach((m) => m.remove());
    markerObjs.current = [];

    storesList.forEach((store) => {
      if (!store.latitude || !store.longitude) return;

      // Create custom marker element with label
      const el = document.createElement("div");
      el.className = "custom-marker";
      el.innerHTML = `
        <div style="
          background: #00274A;
          color: white;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          font-weight: 600;
          white-space: nowrap;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
          position: relative;
        ">
          ${store.name}
          <div style="
            position: absolute;
            bottom: -6px;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-top: 6px solid #00274A;
          "></div>
        </div>
      `;

      const marker = new mapboxgl.Marker({
        element: el,
        anchor: "bottom",
      })
        .setLngLat([store.longitude, store.latitude])
        .addTo(mapObj.current);

      // Add click handler
      el.style.cursor = "pointer";
      el.addEventListener("click", () => {
        setSelectedStore(store);
        mapObj.current.flyTo({
          center: [store.longitude, store.latitude],
          zoom: 14,
        });
      });

      markerObjs.current.push(marker);
    });
  }

  const formatAddress = (address) => {
    if (!address) return "";
    const parts = [
      address.street,
      address.city,
      address.state,
      address.zip,
      address.country,
    ].filter(Boolean);
    return parts.join(", ");
  };

  if (stores.length === 0) {
    return (
      <section className="mx-auto max-w-6xl space-y-5 px-4 py-8">
        <EmptyMessage>
          <EmptyMessage.Icon>
            <MapPin className="h-8 w-8" />
          </EmptyMessage.Icon>
          <EmptyMessage.Title>No stores found</EmptyMessage.Title>
          <EmptyMessage.Description>
            No store locations are available at this time.
          </EmptyMessage.Description>
        </EmptyMessage>
      </section>
    );
  }

  return (
    <section className="mx-auto max-w-6xl space-y-5 px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-black/60 text-sm">Store Locations</p>
          <h1 className="text-2xl font-semibold text-black">Find a Store</h1>
        </div>
        <div className="text-sm text-black/60">
          {stores.length} {stores.length === 1 ? "store" : "stores"}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_400px] gap-6">
        {/* Map */}
        <div className="bg-white rounded-2xl border border-black/10 shadow-lg overflow-hidden max-h-[200px]">
          <div className="relative w-full" style={{ height: "200px" }}>
            <div ref={mapRef} className="w-full h-full" />
          </div>
        </div>

        {/* Store List */}
        <div className="lg:sticky lg:top-4 h-fit">
          <div className="bg-white rounded-2xl border border-black/10 shadow-lg overflow-hidden">
            <div className="p-4 border-b border-black/10">
              <h2 className="text-lg font-semibold text-black">Stores</h2>
            </div>
            <div className="max-h-[600px] overflow-y-auto">
              {stores.map((store) => (
                <div
                  key={store.id}
                  onClick={() => {
                    setSelectedStore(store);
                    if (mapObj.current) {
                      mapObj.current.flyTo({
                        center: [store.longitude, store.latitude],
                        zoom: 14,
                      });
                    }
                  }}
                  className={`p-4 border-b border-black/10 last:border-0 cursor-pointer transition-colors ${
                    selectedStore?.id === store.id
                      ? "bg-[#00274A]/5"
                      : "bg-white hover:bg-gray-50"
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-[#00274A]/10 flex items-center justify-center">
                      <MapPin className="h-5 w-5 text-[#00274A]" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="text-base font-semibold text-black mb-1">
                        {store.name}
                      </h3>
                      <p className="text-sm text-black/60 mb-2">
                        {formatAddress(store.address)}
                      </p>
                      {store.distance_meters && (
                        <Badge color="info" size="sm">
                          {store.distance_meters < 1000
                            ? `${store.distance_meters}m away`
                            : `${(store.distance_meters / 1000).toFixed(1)}km away`}
                        </Badge>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export { App };

const rootElement = document.getElementById("store-search-root");
if (rootElement) {
  createRoot(rootElement).render(<App />);
} else {
  console.error("StoreSearch - Root element 'store-search-root' not found!");
}
