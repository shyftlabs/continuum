# Assortment Frontend

A modern Next.js application with a powerful design system built on top of:

- **Next.js 16** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - Beautiful, accessible components
- **Radix UI** - Unstyled, accessible component primitives
- **Framer Motion** - Production-ready animations
- **React Hook Form** - Performant forms with easy validation
- **Zod** - TypeScript-first schema validation
- **Lucide React** - Beautiful icon library
- **next-themes** - Dark mode support
- **Recharts** - Composable charting library
- **Sonner** - Toast notifications
- **Vaul** - Drawer component
- **cmdk** - Command menu component

## Getting Started

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

### Build for Production

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Root layout with theme provider
│   ├── page.tsx           # Home page
│   └── globals.css        # Global styles with shadcn variables
├── components/            # React components
│   ├── ui/               # shadcn/ui components
│   └── theme-provider.tsx # Theme provider wrapper
├── lib/                  # Utility functions
│   └── utils.ts          # cn() utility for className merging
├── hooks/                # Custom React hooks
├── components.json        # shadcn/ui configuration
└── tailwind.config.ts    # Tailwind CSS configuration
```

## Adding shadcn/ui Components

To add new shadcn/ui components:

```bash
npx shadcn@latest add [component-name]
```

For example:
```bash
npx shadcn@latest add input
npx shadcn@latest add dialog
npx shadcn@latest add dropdown-menu
```

## Features

- ✅ **Dark Mode** - System-aware theme switching
- ✅ **TypeScript** - Full type safety
- ✅ **Responsive Design** - Mobile-first approach
- ✅ **Accessible** - WCAG compliant components
- ✅ **Modern Stack** - Latest versions of all libraries
- ✅ **Fast Development** - Hot module replacement
- ✅ **Production Ready** - Optimized builds

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [shadcn/ui Documentation](https://ui.shadcn.com)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Radix UI Documentation](https://www.radix-ui.com)
