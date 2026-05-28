import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { useWidgetProps } from '../use-widget-props';
import { Button } from '@openai/apps-sdk-ui/components/Button';
import { EmptyMessage } from '@openai/apps-sdk-ui/components/EmptyMessage';
import { DatePicker } from '@openai/apps-sdk-ui/components/DatePicker';
import { CheckCircle2, ArrowLeft, MapPin, Calendar } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { DateTime } from 'luxon';

// Age group options
const AGE_GROUPS = [
  { label: 'Puppy/Kitten', value: '0-1 year' },
  { label: 'Young', value: '1-3 years' },
  { label: 'Adult', value: '3-7 years' },
  { label: 'Senior', value: '7+ years' }
];

// Animal types
const ANIMAL_TYPES = [
  { label: 'Dog', value: 'dog', icon: '🐕' },
  { label: 'Cat', value: 'cat', icon: '🐈' }
];

const AppointmentBooking = () => {
  const widgetProps = useWidgetProps(() => ({}));
  const [step, setStep] = useState(1);
  const [sessionId, setSessionId] = useState(null);
  const [apiBaseUrl, setApiBaseUrl] = useState(null); // API base URL from props
  const [selectedAnimalType, setSelectedAnimalType] = useState(null);
  const [selectedAge, setSelectedAge] = useState(null);
  const [stores, setStores] = useState([]);
  const [selectedStore, setSelectedStore] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [availableSlots, setAvailableSlots] = useState([]);
  const [selectedSlot, setSelectedSlot] = useState(null);
  const [userName, setUserName] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [bookingSuccess, setBookingSuccess] = useState(false);

  // Poll for data injection and sessionId (same pattern as other widgets)
  useEffect(() => {
    const MAX_ATTEMPTS = 60;
    const DELAY = 150;
    let attempts = 0;

    function pullData() {
      const openai = window.openai || {};
      const toolOutput = openai.toolOutput;

      // PRIORITY 1: Get MCP session_id from window.openai.context (set by WidgetRenderer)
      if (openai.context) {
        const mcpSessionId = openai.context.mcp_session_id || 
                            openai.context.sessionId || 
                            openai.context.session_id;
        if (mcpSessionId) {
          console.log('book-appointment - Using MCP session_id from context:', mcpSessionId.substring(0, 8) + '...');
          setSessionId(mcpSessionId);
        }
      }

      if (toolOutput && typeof toolOutput === 'object' && Object.keys(toolOutput).length > 0) {
        // Try to get sessionId from toolOutput
        const foundSessionId = toolOutput.mcp_session_id ||
                               toolOutput.sessionId || 
                               toolOutput.session_id || 
                               toolOutput.session ||
                               toolOutput.structuredContent?.mcp_session_id ||
                               toolOutput.structuredContent?.sessionId ||
                               toolOutput.structuredContent?.session_id;
        
        if (foundSessionId) {
          console.log('book-appointment - Found session_id:', foundSessionId.substring(0, 8) + '...');
          setSessionId(foundSessionId);
        }
        
        // Get API base URL from props (backend injects this)
        const foundApiBaseUrl = toolOutput.apiBaseUrl || 
                                toolOutput.api_base_url ||
                                toolOutput.structuredContent?.apiBaseUrl ||
                                toolOutput.structuredContent?.api_base_url;
        if (foundApiBaseUrl) {
          console.log('book-appointment - Found apiBaseUrl:', foundApiBaseUrl);
          setApiBaseUrl(foundApiBaseUrl);
        }
        
        // Extract appointment data
        const appointmentData = toolOutput.structuredContent || toolOutput;
        if (appointmentData) {
          if (appointmentData.animal_type) {
            setSelectedAnimalType(appointmentData.animal_type);
          }
          if (appointmentData.animal_age) {
            setSelectedAge(appointmentData.animal_age);
          }
          if (appointmentData.user_name) {
            setUserName(appointmentData.user_name);
          }
          if (appointmentData.mobile_number) {
            setPhoneNumber(appointmentData.mobile_number);
          }
          if (appointmentData.store_id) {
            // Store ID will be handled in a separate effect after stores are loaded
            setSelectedStore({ id: appointmentData.store_id });
          }
          if (appointmentData.appointment_datetime) {
            const appointmentDate = DateTime.fromISO(appointmentData.appointment_datetime);
            if (appointmentDate.isValid) {
              setSelectedDate(appointmentDate);
              setSelectedSlot(appointmentData.appointment_datetime);
            }
          }
        }
        
        return true;
      }

      if (widgetProps && typeof widgetProps === 'object' && Object.keys(widgetProps).length > 0) {
        const foundSessionId = widgetProps.mcp_session_id ||
                              widgetProps.sessionId || 
                              widgetProps.session_id || 
                              widgetProps.session;
        
        if (foundSessionId) {
          setSessionId(foundSessionId);
        }
        
        // Get API base URL
        const foundApiBaseUrl = widgetProps.apiBaseUrl || widgetProps.api_base_url;
        if (foundApiBaseUrl) {
          setApiBaseUrl(foundApiBaseUrl);
        }
        
        // Extract appointment data from widgetProps
        if (widgetProps.animal_type) {
          setSelectedAnimalType(widgetProps.animal_type);
        }
        if (widgetProps.animal_age) {
          setSelectedAge(widgetProps.animal_age);
        }
        if (widgetProps.user_name) {
          setUserName(widgetProps.user_name);
        }
        if (widgetProps.mobile_number) {
          setPhoneNumber(widgetProps.mobile_number);
        }
        if (widgetProps.store_id) {
          setSelectedStore({ id: widgetProps.store_id });
        }
        if (widgetProps.appointment_datetime) {
          const appointmentDate = DateTime.fromISO(widgetProps.appointment_datetime);
          if (appointmentDate.isValid) {
            setSelectedDate(appointmentDate);
            setSelectedSlot(widgetProps.appointment_datetime);
          }
        }
        
        return true;
      }

      if (window.widgetProps && typeof window.widgetProps === 'object' && Object.keys(window.widgetProps).length > 0) {
        const data = window.widgetProps.structuredContent || window.widgetProps;
        
        const foundSessionId = data.mcp_session_id ||
                              data.sessionId || 
                              data.session_id || 
                              data.session;
        
        if (foundSessionId) {
          setSessionId(foundSessionId);
        }
        
        // Get API base URL
        const foundApiBaseUrl = data.apiBaseUrl || data.api_base_url;
        if (foundApiBaseUrl) {
          setApiBaseUrl(foundApiBaseUrl);
        }
        
        // Extract appointment data from window.widgetProps
        if (data.animal_type) {
          setSelectedAnimalType(data.animal_type);
        }
        if (data.animal_age) {
          setSelectedAge(data.animal_age);
        }
        if (data.user_name) {
          setUserName(data.user_name);
        }
        if (data.mobile_number) {
          setPhoneNumber(data.mobile_number);
        }
        if (data.store_id) {
          setSelectedStore({ id: data.store_id });
        }
        if (data.appointment_datetime) {
          const appointmentDate = DateTime.fromISO(data.appointment_datetime);
          if (appointmentDate.isValid) {
            setSelectedDate(appointmentDate);
            setSelectedSlot(data.appointment_datetime);
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
      }
    }

    hydrate();
  }, [widgetProps]);

  // Load stores when step 3 is reached or when store_id is provided
  useEffect(() => {
    if ((step === 3 || selectedStore?.id) && stores.length === 0) {
      loadStores();
    }
  }, [step, selectedStore]);

  // Set selectedStore when stores are loaded and we have a store_id
  useEffect(() => {
    if (stores.length > 0 && selectedStore?.id && !selectedStore?.name) {
      const foundStore = stores.find(store => store.id === selectedStore.id);
      if (foundStore) {
        setSelectedStore(foundStore);
      }
    }
  }, [stores, selectedStore]);

  // Load available slots when store and date are provided
  useEffect(() => {
    if (selectedStore?.id && selectedDate) {
      const dateString = selectedDate.toISODate();
      // Always load slots when we have store and date, especially if we have a selectedSlot
      // This ensures slots are available for matching
      loadAvailableSlots(selectedStore.id, dateString);
    }
  }, [selectedStore, selectedDate]);

  // Match selectedSlot with available slots when slots are loaded
  useEffect(() => {
    if (selectedSlot && availableSlots.length > 0) {
      // Check if selectedSlot exactly matches any available slot
      const exactMatch = availableSlots.find(slot => slot === selectedSlot);
      if (exactMatch) {
        // Already matched, no action needed
        return;
      }
      
      // Try to find a match by comparing the actual datetime values
      // This handles timezone and format differences
      try {
        const selectedDateTime = DateTime.fromISO(selectedSlot);
        if (!selectedDateTime.isValid) {
          return;
        }
        
        // Normalize to UTC for comparison to handle timezone differences
        const selectedUTC = selectedDateTime.toUTC();
        
        const matchingSlot = availableSlots.find(slot => {
          const slotDateTime = DateTime.fromISO(slot);
          if (!slotDateTime.isValid) {
            return false;
          }
          // Compare if they represent the same moment in time
          // Normalize both to UTC for accurate comparison
          const slotUTC = slotDateTime.toUTC();
          return selectedUTC.toMillis() === slotUTC.toMillis();
        });
        
        if (matchingSlot) {
          // Update selectedSlot to match the exact format from available slots
          setSelectedSlot(matchingSlot);
        }
      } catch (error) {
        console.error('Error matching slot:', error);
      }
    }
  }, [availableSlots, selectedSlot]);

  // Helper function to determine target step based on available data
  const getTargetStep = () => {
    // If all data is available, go to final step
    if (selectedAnimalType && selectedAge && selectedStore?.name && selectedDate && selectedSlot && userName && phoneNumber) {
      return 5;
    }
    // If we have everything except user info, go to step 5
    if (selectedAnimalType && selectedAge && selectedStore?.name && selectedDate && selectedSlot) {
      return 5;
    }
    // If we have store and date but no slot, go to step 4
    if (selectedAnimalType && selectedAge && selectedStore?.name && selectedDate) {
      return 4;
    }
    // If we have store but no date, go to step 4
    if (selectedAnimalType && selectedAge && selectedStore?.name) {
      return 4;
    }
    // If we have animal info but no store, go to step 3
    if (selectedAnimalType && selectedAge) {
      return 3;
    }
    // If we have animal type but no age, go to step 2
    if (selectedAnimalType) {
      return 2;
    }
    // Otherwise, start at step 1
    return 1;
  };

  // Track if we've done the initial auto-advance
  const [hasAutoAdvanced, setHasAutoAdvanced] = useState(false);

  // Advance to appropriate step based on available data (only once on initial data load)
  useEffect(() => {
    if (!hasAutoAdvanced) {
      const targetStep = getTargetStep();
      // Only auto-advance if we have at least some data (not all empty)
      if (targetStep > 1 || selectedAnimalType || selectedAge || selectedStore || selectedDate || selectedSlot || userName || phoneNumber) {
        setStep(targetStep);
        setHasAutoAdvanced(true);
      }
    }
  }, [selectedAnimalType, selectedAge, selectedStore, selectedDate, selectedSlot, userName, phoneNumber, hasAutoAdvanced]);

  // Handle manual step navigation
  const handleStepClick = (targetStep) => {
    setStep(targetStep);
  };

  const getApiBaseUrl = () => {
    // Priority: props apiBaseUrl > env var > default
    return apiBaseUrl || import.meta.env.VITE_BASE_URL || 'https://api.omnity.shyftops.io/api/v1';
  };

  const loadStores = async () => {
    setLoading(true);
    setError(null);
    try {
      const apiBaseUrl = getApiBaseUrl();
      const response = await fetch(`${apiBaseUrl}/stores?page=1&per_page=100`);
    
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    
      const data = await response.json();
    
      if (data?.items) {
        setStores(data.items);
      } else if (data?.data?.items) {
        setStores(data.data.items);
      }
    } catch (err) {
      setError('Failed to load stores. Please try again.');
      console.error('Error loading stores:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadAvailableSlots = async (storeId, date) => {
    setLoading(true);
    setError(null);
    try {
      // Get today's date in YYYY-MM-DD format
      const today = date || new Date().toISOString().split('T')[0];
      const apiBaseUrl = getApiBaseUrl();
    
      const response = await fetch(`${apiBaseUrl}/appointments/store/${storeId}/available-slots?date=${today}`);
    
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    
      const data = await response.json();
    
      // API returns { store_id, date, slots: [{ datetime, available }] }
      if (data?.slots) {
        // Extract datetime strings from slot objects
        const slotDatetimes = data.slots
          .filter(slot => slot.available)
          .map(slot => slot.datetime);
        setAvailableSlots(slotDatetimes);
      } else if (data?.data?.slots) {
        const slotDatetimes = data.data.slots
          .filter(slot => slot.available)
          .map(slot => slot.datetime);
        setAvailableSlots(slotDatetimes);
      }
    } catch (err) {
      setError('Failed to load available slots. Please try again.');
      console.error('Error loading slots:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleStoreSelect = (store) => {
    setSelectedStore(store);
    setStep(4);
  };

  const handleDateChange = async (date) => {
    if (!date) return;
    setSelectedDate(date);
    // Automatically load slots when date is selected
    if (selectedStore) {
      const dateString = date.toISODate();
      await loadAvailableSlots(selectedStore.id, dateString);
    }
  };

  const handleSlotSelect = (slot) => {
    setSelectedSlot(slot);
    setStep(5); // Move to user info step
  };

  const handleBookAppointment = async () => {
    if (!sessionId || !selectedStore || !selectedSlot || !userName || !phoneNumber) {
      setError('Please fill in all fields. Session ID may not be available yet.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const apiBaseUrl = getApiBaseUrl();
      const response = await fetch(`${apiBaseUrl}/appointments`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          session_id: sessionId,
          store_id: selectedStore.id,
          animal_type: selectedAnimalType,
          animal_age: selectedAge,
          user_name: userName,
          mobile_number: phoneNumber,
          appointment_datetime: selectedSlot
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      // API returns { appointment: {...}, message: "..." }
      if (data?.appointment?.id || data?.data?.appointment?.id || data?.id || data?.data?.id) {
        setBookingSuccess(true);
      } else {
        throw new Error('Booking failed - no appointment ID returned');
      }
    } catch (err) {
      setError(err.message || 'Failed to book appointment. Please try again.');
      console.error('Error booking appointment:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { 
      hour: 'numeric', 
      minute: '2-digit',
      hour12: true,
      timeZone: 'UTC'
    });
  };

  const formatDate = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', { 
      weekday: 'long',
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      timeZone: 'UTC'
    });
  };

  if (bookingSuccess) {
    return (
      <section className="mx-auto max-w-6xl space-y-5 px-4 py-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-black/60 text-sm">Vet Appointments</p>
            <h1 className="text-2xl font-semibold text-black">Appointment Confirmed</h1>
          </div>
        </div>
        <div className="bg-white rounded-2xl border border-black/10 shadow-lg p-8 text-center">
          <div className="flex justify-center mb-4">
            <div className="w-16 h-16 rounded-full bg-green-500 flex items-center justify-center">
              <CheckCircle2 className="h-8 w-8 text-white" />
            </div>
          </div>
          <h2 className="text-2xl font-semibold text-black mb-4">Appointment Booked!</h2>
          <p className="text-base text-black/70 mb-2">
            Your appointment has been successfully booked for {formatDate(selectedSlot)} at {formatTime(selectedSlot)}.
          </p>
          <p className="text-base text-black/70">
            Store: {selectedStore?.name}
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="mx-auto max-w-6xl space-y-5 px-4 py-8 overflow-visible">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-black/60 text-sm">Vet Appointments</p>
          <h1 className="text-2xl font-semibold text-black">Book Appointment</h1>
        </div>
        <div className="text-sm text-black/60">
          Step {step} of 5
        </div>
      </div>

      <div className="bg-white rounded-2xl border border-black/10 shadow-lg p-6" style={{ overflow: 'visible', position: 'relative' }}>
        {/* Step Indicator - Clickable */}
        <div className="flex justify-center gap-2 mb-6">
          {[1, 2, 3, 4, 5].map((s) => {
            const isCompleted = (s === 1 && selectedAnimalType) ||
                               (s === 2 && selectedAnimalType && selectedAge) ||
                               (s === 3 && selectedAnimalType && selectedAge && selectedStore?.name) ||
                               (s === 4 && selectedAnimalType && selectedAge && selectedStore?.name && selectedDate && selectedSlot) ||
                               (s === 5 && selectedAnimalType && selectedAge && selectedStore?.name && selectedDate && selectedSlot && userName && phoneNumber);
            
            return (
              <button
                key={s}
                onClick={() => handleStepClick(s)}
                className={`h-2 rounded-full transition-all cursor-pointer hover:opacity-80 ${
                  s === step
                    ? 'w-6 bg-blue-500'
                    : isCompleted
                    ? 'w-2 bg-green-500'
                    : 'w-2 bg-gray-300'
                }`}
                title={`Step ${s}: ${s === 1 ? 'Animal Type' : s === 2 ? 'Age' : s === 3 ? 'Store' : s === 4 ? 'Date & Time' : 'User Info'}`}
              />
            );
          })}
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-600 text-sm font-medium">{error}</p>
          </div>
        )}

        {/* Step 1: Animal Type Selection */}
        <AnimatePresence mode="wait">
          {step === 1 && (
            <motion.div
              key="step1"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="min-h-[300px]"
            >
              <h2 className="text-xl font-semibold text-black mb-6 text-center">Select Animal Type</h2>
              <div className="grid grid-cols-2 gap-4 mb-6">
                {ANIMAL_TYPES.map((animal, index) => (
                  <motion.button
                    key={animal.value}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                    onClick={() => {
                      setSelectedAnimalType(animal.value);
                      setStep(2);
                    }}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className={`p-8 border-2 rounded-xl transition-all text-center ${
                      selectedAnimalType === animal.value
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-black/10 bg-white hover:border-black/20'
                    }`}
                  >
                    <motion.div
                      animate={selectedAnimalType === animal.value ? { scale: [1, 1.1, 1] } : {}}
                      transition={{ duration: 0.3 }}
                      className="text-5xl mb-3"
                    >
                      {animal.icon}
                    </motion.div>
                    <div className="text-lg font-semibold text-black">{animal.label}</div>
                  </motion.button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Step 2: Age Selection */}
        <AnimatePresence mode="wait">
          {step === 2 && (
            <motion.div
              key="step2"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="min-h-[300px]"
            >
              <h2 className="text-xl font-semibold text-black mb-6 text-center">Select Age Group</h2>
              <div className="grid grid-cols-2 gap-4 mb-6">
                {AGE_GROUPS.map((age, index) => (
                  <motion.button
                    key={age.value}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                    onClick={() => {
                      setSelectedAge(age.value);
                      setStep(3);
                    }}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className={`p-8 border-2 rounded-xl transition-all text-center ${
                      selectedAge === age.value
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-black/10 bg-white hover:border-black/20'
                    }`}
                  >
                    <div className="text-lg font-semibold text-black mb-1">{age.label}</div>
                    <div className="text-sm text-gray-600">{age.value}</div>
                  </motion.button>
                ))}
              </div>
              <Button
                onClick={() => setStep(1)}
                variant="outline"
                size="md"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Step 3: Store Selection */}
        <AnimatePresence mode="wait">
          {step === 3 && (
            <motion.div
              key="step3"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="min-h-[300px]"
            >
              <h2 className="text-xl font-semibold text-black mb-6 text-center">Select Store</h2>
              {loading && stores.length === 0 ? (
                <div className="py-12 text-center text-gray-600">Loading stores...</div>
              ) : stores.length === 0 ? (
                <EmptyMessage>
                  <EmptyMessage.Icon>
                    <MapPin className="h-8 w-8" />
                  </EmptyMessage.Icon>
                  <EmptyMessage.Title>No stores available</EmptyMessage.Title>
                  <EmptyMessage.Description>
                    Please try again later
                  </EmptyMessage.Description>
                </EmptyMessage>
              ) : (
                <>
                  <div className="space-y-3 mb-6">
                    {stores.map((store, index) => (
                      <motion.button
                        key={store.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.3, delay: index * 0.05 }}
                        onClick={() => handleStoreSelect(store)}
                        whileHover={{ scale: 1.01, x: 4 }}
                        whileTap={{ scale: 0.99 }}
                        className={`w-full p-4 border-2 rounded-lg text-left transition-all ${
                          selectedStore?.id === store.id
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-black/10 bg-white hover:border-black/20'
                        }`}
                      >
                        <div className="text-base font-semibold text-black mb-1">{store.name}</div>
                        {store.address && (
                          <div className="text-sm text-gray-600">
                            {store.address.street}, {store.address.city}, {store.address.state} {store.address.zip}
                          </div>
                        )}
                      </motion.button>
                    ))}
                  </div>

                  <Button
                    onClick={() => setStep(2)}
                    variant="outline"
                    size="md"
                  >
                    <ArrowLeft className="h-4 w-4 mr-2" />
                    Back
                  </Button>
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Step 4: Date and Time Slot Selection */}
        <AnimatePresence mode="wait">
          {step === 4 && (
            <motion.div
              key="step4"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="min-h-[300px]"
              style={{ overflow: 'visible', position: 'relative' }}
            >
              <h2 className="text-xl font-semibold text-black mb-6 text-center">Select Date & Time</h2>
              {selectedStore && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className="p-3 bg-black/5 rounded-lg mb-4 text-center"
                >
                  <p className="text-sm text-black/70"><strong>Store:</strong> {selectedStore.name}</p>
                </motion.div>
              )}
              
              {/* Compact Date Picker */}
              <div className="mb-6" style={{ position: 'relative', zIndex: 50 }}>
                <DatePicker
                  id="appointment-date-picker"
                  value={selectedDate}
                  onChange={handleDateChange}
                  min={DateTime.now().startOf('day')}
                  placeholder="Select appointment date"
                  triggerDateFormat="MMM d, yyyy"
                  block={true}
                  size="md"
                  triggerShowIcon={true}
                />
              </div>

              {/* Time Slots */}
              {selectedDate && (
                <div className="mb-6" style={{ position: 'relative'}}>
                  <h3 className="text-base font-semibold text-black mb-3">Available Time Slots</h3>
                  {loading ? (
                    <div className="py-8 text-center text-gray-600">Loading available slots...</div>
                  ) : availableSlots.length > 0 ? (
                    <div className="grid grid-cols-4 gap-2">
                      {availableSlots.map((slot, index) => (
                        <motion.button
                          key={slot}
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.2, delay: index * 0.03 }}
                          onClick={() => handleSlotSelect(slot)}
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          className={`p-3 border-2 rounded-lg text-sm transition-all ${
                            selectedSlot === slot
                              ? 'border-blue-500 bg-blue-50 font-semibold'
                              : 'border-black/10 bg-white hover:border-black/20'
                          }`}
                        >
                          {formatTime(slot)}
                        </motion.button>
                      ))}
                    </div>
                  ) : (
                    <EmptyMessage>
                      <EmptyMessage.Icon>
                        <MapPin className="h-8 w-8" />
                      </EmptyMessage.Icon>
                      <EmptyMessage.Title>No available slots</EmptyMessage.Title>
                      <EmptyMessage.Description>
                        No available slots for this date. Please try another date.
                      </EmptyMessage.Description>
                    </EmptyMessage>
                  )}
                </div>
              )}

              <Button
                onClick={() => setStep(3)}
                variant="outline"
                size="md"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Step 5: User Information */}
        <AnimatePresence mode="wait">
          {step === 5 && (
            <motion.div
              key="step5"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="min-h-[300px]"
            >
              <h2 className="text-xl font-semibold text-black mb-6 text-center">Your Information</h2>
              <div className="space-y-4 mb-6">
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.1 }}
                >
                  <label className="block text-sm font-semibold text-black mb-2">Full Name</label>
                  <input
                    type="text"
                    value={userName}
                    onChange={(e) => setUserName(e.target.value)}
                    placeholder="John Doe"
                    className="w-full p-3 border-2 border-black/10 rounded-lg text-base focus:outline-none focus:border-blue-500"
                  />
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.2 }}
                >
                  <label className="block text-sm font-semibold text-black mb-2">Phone Number</label>
                  <input
                    type="tel"
                    value={phoneNumber}
                    onChange={(e) => setPhoneNumber(e.target.value)}
                    placeholder="+1234567890"
                    className="w-full p-3 border-2 border-black/10 rounded-lg text-base focus:outline-none focus:border-blue-500"
                  />
                </motion.div>

                {selectedStore && selectedSlot && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.3, delay: 0.3 }}
                    className="p-4 bg-black/5 rounded-lg mt-4"
                  >
                    <h3 className="text-base font-semibold text-black mb-3">Appointment Summary</h3>
                    <div className="space-y-2 text-sm text-black/70">
                      <p><strong>Store:</strong> {selectedStore.name}</p>
                      <p><strong>Date:</strong> {formatDate(selectedSlot)}</p>
                      <p><strong>Time:</strong> {formatTime(selectedSlot)}</p>
                      <p><strong>Animal:</strong> {selectedAnimalType} ({selectedAge})</p>
                    </div>
                  </motion.div>
                )}

                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.4 }}
                >
                  <Button
                    onClick={handleBookAppointment}
                    disabled={loading || !userName || !phoneNumber}
                    color="primary"
                    size="lg"
                    block
                    className="mt-4"
                  >
                    {loading ? 'Booking...' : 'Book Appointment'}
                  </Button>
                </motion.div>
              </div>

              <Button
                onClick={() => setStep(4)}
                variant="outline"
                size="md"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
};


// Initialize the widget
const rootElement = document.getElementById('book-appointment-root');
if (rootElement) {
  const root = createRoot(rootElement);
  root.render(<AppointmentBooking />);
}

export default AppointmentBooking;
export { AppointmentBooking as App };