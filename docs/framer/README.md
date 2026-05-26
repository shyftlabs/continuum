# Framer code component

`ContinuumDocs.tsx` is the entire docs page packaged as a Framer code component.

## How to use it in Framer

1. Open your Framer project → **Assets → Code → +** (new code file).
2. Paste the contents of `ContinuumDocs.tsx`. Framer will compile it.
3. Drag the component onto a page. The sidebar shows 5 controls:
   - **Logo** — image picker (use the hex S `logo.jpeg`).
   - **Inter font** — Inter Variable woff2.
   - **Geist font** — Geist Variable woff2.
   - **JetBrains Mono font** — woff2.
   - **Space Mono font** — woff2 (regular + bold combined, or just bold).
4. Set width/height to fill the canvas (or `100%`).

When any of those props are empty, the component still renders — it'll just
fall back to system fonts and a missing-image alt for the logo.

## Where to get font files

Download woff2 files from the official sources and upload to Framer's
Asset library:

- Inter — https://rsms.me/inter/inter.zip → `Inter-roman.var.woff2`
- Geist — https://github.com/vercel/geist-font → `GeistVariableVF.woff2`
- JetBrains Mono — https://www.jetbrains.com/lp/mono/ → `JetBrainsMono[wght].woff2`
- Space Mono — https://fonts.google.com/specimen/Space+Mono → `SpaceMono-Bold.woff2`

## Regenerating after edits

After any change to `docs/index.html`:

```bash
python3 docs/framer/build_framer.py
```

This re-extracts the `<style>`, `<body>`, and `<script>` blocks, replaces the
local logo path with a runtime placeholder, and writes a fresh
`ContinuumDocs.tsx`.

## What the generator does

1. Pulls everything inside `<style>...</style>` → embedded CSS.
2. Pulls everything inside `<body>...</body>` minus the inline `<script>` → page markup.
3. Pulls the inline `<script>` → bound onto `window` inside `useEffect` so the
   `onclick="showSection(...)"` attributes in the markup still resolve.
4. Rewrites `src="assets/logo.jpeg"` → `src="__LOGO__"` placeholder; the React
   component substitutes the `logo` prop at render time.
5. Adds runtime `@font-face` declarations sourced from the four font props.
