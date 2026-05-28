/// <reference types="vite/client" />

declare module '*.jsx' {
  import type { ComponentType } from 'react';
  export const App: ComponentType<any>;
  export const OrderApp: ComponentType<any>;
  const content: ComponentType<any>;
  export default content;
}
