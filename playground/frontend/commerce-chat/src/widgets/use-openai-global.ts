import { useSyncExternalStore } from "react";
import {
  SET_GLOBALS_EVENT_TYPE,
  SetGlobalsEvent,
  type OpenAiGlobals,
} from "./types";

// Initialize window.openai with default values if it doesn't exist (for dev mode)
if (typeof window !== "undefined" && !window.openai) {
  window.openai = {
    theme: "light",
    userAgent: {
      device: { type: "unknown" },
      capabilities: { hover: true, touch: false },
    },
    locale: "en",
    maxHeight: 600,
    displayMode: "inline",
    safeArea: {
      insets: { top: 0, bottom: 0, left: 0, right: 0 },
    },
    toolInput: {},
    toolOutput: null,
    toolResponseMetadata: null,
    widgetState: null,
    setWidgetState: async () => {},
    callTool: async () => ({ result: "" }),
    sendFollowUpMessage: async () => {},
    openExternal: () => {},
    requestDisplayMode: async () => ({ mode: "inline" }),
  } as any;
}

export function useOpenAiGlobal<K extends keyof OpenAiGlobals>(
  key: K
): OpenAiGlobals[K] | null {
  return useSyncExternalStore(
    (onChange) => {
      if (typeof window === "undefined") {
        return () => {};
      }

      const handleSetGlobal = (event: SetGlobalsEvent) => {
        const value = event.detail.globals[key];
        if (value === undefined) {
          return;
        }

        onChange();
      };

      window.addEventListener(SET_GLOBALS_EVENT_TYPE, handleSetGlobal, {
        passive: true,
      });

      return () => {
        window.removeEventListener(SET_GLOBALS_EVENT_TYPE, handleSetGlobal);
      };
    },
    () => window.openai?.[key] ?? null,
    () => window.openai?.[key] ?? null
  );
}
