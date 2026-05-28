# Chat UI

A React application built with Vite, TypeScript, Tailwind CSS, and shadcn/ui for the Petco agent chat interface.

## Setup

This project uses:
- **React 19** with TypeScript
- **Vite** as the build tool
- **Tailwind CSS v4** for styling
- **shadcn/ui** for component library

## Configuration

Create a `.env` file in the root directory:

```env
VITE_API_URL=http://localhost:8088
```

This should point to your Petco agent API endpoint.

## Development

To start the development server:

```bash
npm run dev
```

The app will be available at `http://localhost:8090`

## Building

To build for production:

```bash
npm run build
```

## Features

- **Chat Interface**: Full chat UI with message history
- **Widget Rendering**: Automatically renders widgets from MCP tool responses
- **Session Management**: Frontend chat session management (separate from MCP session)
- **New Chat**: Create new chat sessions with a single click

## How It Works

1. User sends a message through the chat interface
2. Frontend calls the Petco agent API with `chat_session_id`
3. Agent processes the message and calls MCP tools
4. MCP tools return widgets via `run_artifacts` with:
   - `meta['openai/outputTemplate']`: Widget template URL (e.g., `ui://widget/cart.html`)
   - `structured_content`: Data to inject into the widget
5. Frontend extracts widget name from template URL
6. Frontend loads the corresponding widget from `src/widgets/`
7. Widget receives data via `window.openai.toolOutput`
8. Widget renders inline in the chat message

## Widget System

Widgets are located in `src/widgets/` and follow this pattern:
- Each widget has an `index.jsx` file
- Widgets look for a root element by ID (e.g., `cart-root`)
- Widgets read data from `window.openai.toolOutput`
- Widgets are automatically rendered when MCP tools return widget metadata

## Project Structure

```
chat_ui/
├── src/
│   ├── components/
│   │   ├── chat/          # Chat UI components
│   │   │   ├── ChatContainer.tsx
│   │   │   ├── ChatMessage.tsx
│   │   │   ├── ChatInput.tsx
│   │   │   └── WidgetRenderer.tsx
│   │   └── ui/            # shadcn/ui components
│   ├── widgets/           # Widget components (from MCP)
│   ├── types/             # TypeScript types
│   ├── lib/               # Utility functions
│   ├── hooks/             # Custom React hooks
│   ├── App.tsx            # Main app component
│   └── main.tsx           # Entry point
├── components.json        # shadcn/ui configuration
└── vite.config.ts         # Vite configuration (port: 8090)
```

## Adding shadcn/ui Components

To add shadcn/ui components, you can use the CLI:

```bash
npx shadcn@latest add [component-name]
```

For example:
```bash
npx shadcn@latest add button
npx shadcn@latest add card
npx shadcn@latest add input
```
