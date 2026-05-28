import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 8090,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  // Pre-bundle dependencies used by widgets to prevent runtime reloads
  optimizeDeps: {
    include: [
      // OpenAI SDK UI components used by widgets
      '@openai/apps-sdk-ui/components/DatePicker',
      '@openai/apps-sdk-ui/components/Badge',
      '@openai/apps-sdk-ui/components/Button',
      '@openai/apps-sdk-ui/components/EmptyMessage',
      // Animation & carousel libraries
      'framer-motion',
      'embla-carousel-react',
      // Date/time library
      'luxon',
      // Map library
      'mapbox-gl',
      // Icons
      'lucide-react',
    ],
  },
})
