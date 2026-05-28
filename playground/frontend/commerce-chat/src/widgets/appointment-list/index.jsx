import React, { useState, useEffect, useMemo, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import { useWidgetProps } from '../use-widget-props';
import { EmptyMessage } from '@openai/apps-sdk-ui/components/EmptyMessage';
import { Badge } from '@openai/apps-sdk-ui/components/Badge';
import { Calendar, Clock, MapPin, User } from 'lucide-react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

// Set Mapbox access token
mapboxgl.accessToken = "pk.eyJ1IjoiZXJpY25pbmciLCJhIjoiY21icXlubWM1MDRiczJvb2xwM2p0amNyayJ9.n-3O6JI5nOp_Lw96ZO5vJQ";

// Appointment Card Component with Map
const AppointmentCard = ({ appointment, formatDate, formatTime, getStatusColor, getStatusLabel, getAnimalIcon }) => {
  const mapRef = useRef(null);
  const mapObj = useRef(null);
  const markerObj = useRef(null);

  // Initialize map when coordinates are available
  useEffect(() => {
    if (!appointment.store_latitude || !appointment.store_longitude) return;
    if (mapObj.current || !mapRef.current) return;

    const coords = [appointment.store_longitude, appointment.store_latitude];

    mapObj.current = new mapboxgl.Map({
      container: mapRef.current,
      style: "mapbox://styles/mapbox/streets-v12",
      center: coords,
      zoom: 14,
      attributionControl: false,
    });

    // Create custom marker element with store name (matching store list style)
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
        ${appointment.store_name || 'Store'}
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

    // Add marker with custom element
    markerObj.current = new mapboxgl.Marker({
      element: el,
      anchor: "bottom",
    })
      .setLngLat(coords)
      .addTo(mapObj.current);

    // Handle resize
    const handleResize = () => {
      if (mapObj.current) {
        mapObj.current.resize();
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (markerObj.current) {
        markerObj.current.remove();
        markerObj.current = null;
      }
      if (mapObj.current) {
        mapObj.current.remove();
        mapObj.current = null;
      }
    };
  }, [appointment.store_latitude, appointment.store_longitude, appointment.store_name]);

  return (
    <div className="bg-white rounded-2xl border border-black/10 shadow-lg overflow-hidden">
      <div className="flex flex-row">
        {/* Left: Appointment Info */}
        <div className="p-6 flex-1 min-w-0">
          {/* Status Badge */}
          <div className="flex justify-end mb-4">
            <Badge color={getStatusColor(appointment.status)} size="sm">
              {getStatusLabel(appointment.status)}
            </Badge>
          </div>

          {/* Animal Info */}
          <div className="flex items-center gap-3 mb-4 pb-4 border-b border-black/10">
            <span className="text-3xl">
              {getAnimalIcon(appointment.animal_type)}
            </span>
            <div className="flex-1">
              <div className="text-lg font-semibold text-black mb-1">
                {appointment.animal_type ? appointment.animal_type.charAt(0).toUpperCase() + appointment.animal_type.slice(1) : 'Pet'}
              </div>
              <div className="text-sm text-black/60">
                {appointment.animal_age || 'Age not specified'}
              </div>
            </div>
          </div>

          {/* Date and Time */}
          <div className="space-y-2 mb-4">
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4 text-black/60" />
              <span className="text-base font-medium text-black">
                {formatDate(appointment.appointment_datetime)}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-black/60" />
              <span className="text-base text-black/70">
                {formatTime(appointment.appointment_datetime)}
              </span>
            </div>
          </div>

          {/* Store Info */}
          {appointment.store_name && (
            <div className="flex items-center gap-2 mb-3 p-2 bg-black/5 rounded-lg">
              <MapPin className="h-4 w-4 text-black/60" />
              <span className="text-sm font-medium text-black/70">
                {appointment.store_name}
              </span>
            </div>
          )}

          {/* Owner Info */}
          {appointment.user_name && (
            <div className="flex items-center gap-2 mb-3">
              <User className="h-4 w-4 text-black/60" />
              <span className="text-sm text-black/70">
                {appointment.user_name}
              </span>
              {appointment.mobile_number && (
                <span className="text-sm text-black/60">
                  • {appointment.mobile_number}
                </span>
              )}
            </div>
          )}

          {/* Created Date */}
          {appointment.created_at && (
            <div className="mt-4 pt-4 border-t border-black/10">
              <span className="text-xs text-black/50">
                Booked on {formatDate(appointment.created_at)} at {formatTime(appointment.created_at)}
              </span>
            </div>
          )}
        </div>

        {/* Right: Map */}
        {appointment.store_latitude && appointment.store_longitude ? (
          <div className="bg-gray-100 border-l border-black/10 w-[400px] flex-shrink-0">
            <div className="relative w-full h-full" style={{ minHeight: '300px', height: '100%' }}>
              <div ref={mapRef} className="w-full h-full" style={{ minHeight: '300px' }} />
            </div>
          </div>
        ) : (
          <div className="bg-gray-100 border-l border-black/10 w-[400px] flex-shrink-0 flex items-center justify-center p-6" style={{ minHeight: '300px' }}>
            <div className="text-center">
              <MapPin className="h-8 w-8 text-black/40 mx-auto mb-2" />
              <p className="text-sm text-black/60">Location not available</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const AppointmentList = () => {
  const widgetProps = useWidgetProps(() => ({}));
  const [appointmentsData, setAppointmentsData] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Poll for data injection (same pattern as other widgets)
  useEffect(() => {
    const MAX_ATTEMPTS = 60;
    const DELAY = 150;
    let attempts = 0;

    function pullData() {
      const openai = window.openai || {};
      const toolOutput = openai.toolOutput;

      if (toolOutput && typeof toolOutput === 'object' && Object.keys(toolOutput).length > 0) {
        setAppointmentsData(toolOutput);
        
        const foundSessionId = toolOutput.sessionId || 
                               toolOutput.session_id || 
                               toolOutput.session ||
                               toolOutput.structuredContent?.sessionId ||
                               toolOutput.structuredContent?.session_id;
        
        if (foundSessionId) {
          setSessionId(foundSessionId);
        }
        
        return true;
      }

      if (widgetProps && typeof widgetProps === 'object' && Object.keys(widgetProps).length > 0) {
        setAppointmentsData(widgetProps);
        
        const foundSessionId = widgetProps.sessionId || 
                              widgetProps.session_id || 
                              widgetProps.session;
        
        if (foundSessionId) {
          setSessionId(foundSessionId);
        }
        
        return true;
      }

      if (window.widgetProps && typeof window.widgetProps === 'object' && Object.keys(window.widgetProps).length > 0) {
        const data = window.widgetProps.structuredContent || window.widgetProps;
        setAppointmentsData(data);
        
        const foundSessionId = data.sessionId || 
                              data.session_id || 
                              data.session;
        
        if (foundSessionId) {
          setSessionId(foundSessionId);
        }
        
        return true;
      }
      
      return false;
    }

    function hydrate() {
      if (pullData()) {
        setLoading(false);
        return;
      }
      if (attempts++ < MAX_ATTEMPTS) {
        setTimeout(hydrate, DELAY);
      } else {
        setAppointmentsData({});
        setLoading(false);
      }
    }

    hydrate();
  }, [widgetProps]);

  // Extract appointments from the polled data
  const appointments = useMemo(() => {
    if (!appointmentsData) return [];
    
    // Handle direct array
    if (Array.isArray(appointmentsData)) {
      return appointmentsData;
    }
    
    // Handle items array (from list_appointments_by_session)
    if (appointmentsData.items && Array.isArray(appointmentsData.items)) {
      return appointmentsData.items;
    }
    
    // Handle structuredContent
    const content = appointmentsData.structuredContent || appointmentsData;
    
    // Handle single appointment (from get_appointment)
    if (content.id && content.appointment_datetime) {
      return [content];
    }
    
    // Handle items in structuredContent
    if (content.items && Array.isArray(content.items)) {
      return content.items;
    }
    
    // Handle single appointment in structuredContent
    if (content.appointment_datetime) {
      return [content];
    }
    
    // Handle appointment object directly
    if (content.appointment && content.appointment.id) {
      return [content.appointment];
    }
    
    return [];
  }, [appointmentsData]);

  const formatDate = (isoString) => {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', { 
      weekday: 'long',
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      timeZone: 'America/New_York'
    });
  };

  const formatTime = (isoString) => {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { 
      hour: 'numeric', 
      minute: '2-digit',
      hour12: true,
      timeZone: 'America/New_York'
    });
  };

  const getStatusColor = (status) => {
    const statusLower = (status || '').toLowerCase();
    if (statusLower === 'confirmed' || statusLower === 'completed') {
      return 'success';
    } else if (statusLower === 'booked' || statusLower === 'pending') {
      return 'warning';
    } else if (statusLower === 'cancelled' || statusLower === 'no_show') {
      return 'error';
    }
    return 'info';
  };

  const getStatusLabel = (status) => {
    if (!status) return 'Unknown';
    return status.charAt(0).toUpperCase() + status.slice(1).toLowerCase().replace('_', ' ');
  };

  const getAnimalIcon = (animalType) => {
    const type = (animalType || '').toLowerCase();
    if (type === 'dog') return '🐕';
    if (type === 'cat') return '🐈';
    if (type === 'bird') return '🐦';
    if (type === 'rabbit') return '🐰';
    return '🐾';
  };

  if (loading) {
    return (
      <section className="mx-auto max-w-6xl space-y-5 px-4 py-8">
        <div className="py-12 text-center text-black/60">Loading appointments...</div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="mx-auto max-w-6xl space-y-5 px-4 py-8">
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-600 text-sm font-medium">{error}</p>
        </div>
      </section>
    );
  }

  if (appointments.length === 0) {
    return (
      <section className="mx-auto max-w-6xl space-y-5 px-4 py-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-black/60 text-sm">Vet Appointments</p>
            <h1 className="text-2xl font-semibold text-black">My Appointments</h1>
          </div>
        </div>
        <EmptyMessage>
          <EmptyMessage.Icon>
            <Calendar className="h-8 w-8" />
          </EmptyMessage.Icon>
          <EmptyMessage.Title>No Appointments Found</EmptyMessage.Title>
          <EmptyMessage.Description>
            You don't have any appointments scheduled.
          </EmptyMessage.Description>
        </EmptyMessage>
      </section>
    );
  }

  return (
    <section className="mx-auto max-w-6xl space-y-5 px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-black/60 text-sm">Vet Appointments</p>
          <h1 className="text-2xl font-semibold text-black">
            {appointments.length === 1 ? 'Appointment Details' : `My Appointments (${appointments.length})`}
          </h1>
        </div>
      </div>

      <div className="space-y-4">
        {appointments.map((appointment, index) => (
          <AppointmentCard 
            key={appointment.id || index} 
            appointment={appointment}
            formatDate={formatDate}
            formatTime={formatTime}
            getStatusColor={getStatusColor}
            getStatusLabel={getStatusLabel}
            getAnimalIcon={getAnimalIcon}
          />
        ))}
      </div>
    </section>
  );
};

// Initialize the widget
const rootElement = document.getElementById('appointment-list-root');
if (rootElement) {
  const root = createRoot(rootElement);
  root.render(<AppointmentList />);
}

export default AppointmentList;
export { AppointmentList as App };
