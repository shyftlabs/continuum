"""
Convert docs/index.html into a self-contained Framer code component.

Output: docs/framer/ContinuumDocs.tsx

The TSX takes 5 Framer props (image + 4 font files). Logo references and the
Google Fonts <link> are stripped from the body/head; the component injects
@font-face declarations from the prop URLs at runtime, then renders the
existing HTML body via dangerouslySetInnerHTML and re-binds the inline JS
(showSection / scrollToAnchor / copyCode) onto window so the onclick=...
attributes in the markup still resolve.

Re-run after editing index.html:

    python3 docs/framer/build_framer.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = ROOT / "index.html"
OUT_PATH = Path(__file__).resolve().parent / "ContinuumDocs.tsx"

html = HTML_PATH.read_text(encoding="utf-8")

# 1. Extract <style>...</style> from <head>
style_m = re.search(r"<style>(.*?)</style>", html, re.DOTALL)
assert style_m, "no <style> block found"
CSS = style_m.group(1).strip()


# 1a. In the Framer build, @media queries fire against the browser
#     viewport — but the component is rendered inside a frame that can
#     be much smaller than the viewport. @container queries depend on
#     runtime + ancestor wiring and don't reliably fire in Framer.
#
#     Transform @media (max-width: N) { rules } into class-prefixed
#     rules. The React component uses ResizeObserver to toggle
#     .cn-tablet / .cn-mobile / .cn-small on the root element based on
#     actual measured width. Works in any runtime.
def _transform_media_to_classes(css: str) -> str:
    BP = {1024: "cn-tablet", 768: "cn-mobile", 480: "cn-small"}

    def parse_rules(body: str):
        i, n = 0, len(body)
        while i < n:
            while i < n and body[i].isspace():
                i += 1
            if i >= n:
                break
            open_pos = body.find("{", i)
            if open_pos == -1:
                break
            selector = body[i:open_pos].strip()
            depth, j = 1, open_pos + 1
            while j < n and depth > 0:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                j += 1
            props = body[open_pos + 1 : j - 1].strip()
            if selector:
                yield selector, props
            i = j

    def transform(m):
        size = int(m.group(1))
        body = m.group(2)
        cls = BP.get(size, f"cn-bp{size}")
        out = []
        for selector, props in parse_rules(body):
            # Drop /* … */ comments embedded in selectors and squash whitespace
            selector = re.sub(r"/\*.*?\*/", "", selector, flags=re.DOTALL)
            selector = re.sub(r"\s+", " ", selector).strip()
            if not selector:
                continue
            parts = []
            for s in selector.split(","):
                s = s.strip()
                if not s:
                    continue
                if s == ":root":
                    # :root inside a container override → put the
                    # variable definition on the class-bearing root
                    # element so it cascades to descendants.
                    parts.append(f".{cls}")
                else:
                    parts.append(f".{cls} {s}")
            # Also squash whitespace in props for tidier output
            props_one_line = re.sub(r"\s+", " ", props).strip()
            out.append(", ".join(parts) + " { " + props_one_line + " }")
        return "\n      " + "\n      ".join(out)

    pattern = re.compile(
        r"@media\s*\(\s*max-width:\s*(\d+)px\s*\)\s*\{((?:[^{}]+|\{[^{}]*\})*)\}",
        re.DOTALL,
    )
    return pattern.sub(transform, css)


CSS = _transform_media_to_classes(CSS)

# 2. Extract <body>...</body>
body_m = re.search(r"<body>(.*?)</body>", html, re.DOTALL)
assert body_m, "no <body> block found"
BODY_RAW = body_m.group(1).strip()

# 3. Pull the inline <script>...</script> out of the body
script_m = re.search(r"<script>(.*?)</script>", BODY_RAW, re.DOTALL)
JS = (script_m.group(1).strip() if script_m else "").strip()
BODY = re.sub(r"<script>.*?</script>", "", BODY_RAW, flags=re.DOTALL).strip()

# 4. Insert runtime placeholders for the user-configurable bits.
#    The React component substitutes these via String.replaceAll at render.
BODY = BODY.replace('src="assets/logo.jpeg"', 'src="__LOGO__"')
BODY = BODY.replace(
    '<span class="wordmark">Continuum</span>',
    '<span class="wordmark">__WORDMARK__</span>',
    1,
)
BODY = BODY.replace(
    '<span class="byline">by <em>ShyftLabs</em></span>',
    '<span class="byline">__BYLINE__</span>',
    1,
)
BODY = BODY.replace(
    '<span class="nav-badge">v0.2.0</span>',
    '<span class="nav-badge">__VERSION__</span>',
    1,
)

# 5. Build the TSX. Use String.raw with our own delimiter so the raw template
#    literal in JS code can contain backticks ("`") and ${ } unescaped.
DELIM = "===END==="
assert DELIM not in CSS and DELIM not in BODY and DELIM not in JS, (
    "delimiter collision — pick a different DELIM"
)


def raw(s: str) -> str:
    # JS template literals don't have a String.raw-with-custom-delimiter form
    # the way Python does, so we manually escape ` and ${.
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


tsx = f"""// AUTO-GENERATED by docs/framer/build_framer.py — do not edit.
// Re-run the generator after any change to docs/index.html.

import * as React from "react"
import {{ addPropertyControls, ControlType }} from "framer"

const CSS = `
{raw(CSS)}
`

const BODY = `
{raw(BODY)}
`

const JS = `
{raw(JS)}
`

interface Props {{
    logo?: string
    wordmark?: string
    byline?: string
    version?: string
    fontInter?: string
    fontGeist?: string
    fontJetBrains?: string
    fontSpaceMono?: string
    width?: number | string
    height?: number | string
}}

function buildFontCss(p: Props) {{
    const face = (family: string, url?: string) =>
        url
            ? `@font-face {{ font-family: '${{family}}'; src: url('${{url}}'); font-weight: 400 800; font-display: swap; font-style: normal; }}`
            : ""
    return [
        face("Inter", p.fontInter),
        face("Geist", p.fontGeist),
        face("JetBrains Mono", p.fontJetBrains),
        face("Space Mono", p.fontSpaceMono),
    ].join("\\n")
}}

export default function ContinuumDocs(props: Props) {{
    const {{
        logo = "",
        wordmark = "",
        byline = "",
        version = "v0.2.0",
        width = "100%",
        height,
    }} = props
    const rootRef = React.useRef<HTMLDivElement>(null)

    // Empty brand props collapse to nothing — when all three are empty
    // we drop the entire <a class="nav-logo"> wrapper so no gap/margin
    // is left behind in the topnav layout.
    const html = React.useMemo(() => {{
        let h = BODY
        const hasBrand = !!(logo || wordmark || byline)

        if (!hasBrand) {{
            // Remove the whole nav-logo anchor (img + wordmark + byline + wrapper)
            h = h.replace(/<a class="nav-logo"[\\s\\S]*?<\\/a>/m, "")
        }} else {{
            if (logo) {{
                h = h.replaceAll("__LOGO__", logo)
            }} else {{
                h = h.replace(/<img class="mark"[^>]*\\/>/g, "")
            }}
            if (wordmark) {{
                h = h.replaceAll("__WORDMARK__", wordmark)
            }} else {{
                h = h.replace(/<span class="wordmark">__WORDMARK__<\\/span>/g, "")
            }}
            if (byline) {{
                h = h.replaceAll("__BYLINE__", byline)
            }} else {{
                h = h.replace(/<span class="byline">__BYLINE__<\\/span>/g, "")
            }}
            // Drop the inner .text wrapper if both wordmark and byline are empty
            if (!wordmark && !byline) {{
                h = h.replace(/<span class="text">[\\s\\S]*?<\\/span>/m, "")
            }}
        }}

        h = h.replaceAll("__VERSION__", version || "")
        return h
    }}, [logo, wordmark, byline, version])

    // Watch the component's measured width and toggle breakpoint classes
    // on the root. This drives all responsive styles — we don't rely on
    // @media or @container queries because Framer's runtime doesn't
    // reliably propagate either down to the component frame size.
    React.useEffect(() => {{
        if (!rootRef.current) return
        const el = rootRef.current
        const apply = (w: number) => {{
            el.classList.toggle("cn-tablet", w <= 1024)
            el.classList.toggle("cn-mobile", w <= 768)
            el.classList.toggle("cn-small", w <= 480)
        }}
        apply(el.getBoundingClientRect().width)
        const ro = new ResizeObserver((entries) => {{
            for (const e of entries) apply(e.contentRect.width)
        }})
        ro.observe(el)
        return () => ro.disconnect()
    }}, [])

    React.useEffect(() => {{
        // Re-bind the inline JS so onclick="..." attributes in the HTML
        // resolve to globally-visible functions on window.
        //
        // IMPORTANT: the wrapper string below is evaluated as plain
        // JavaScript at runtime via new Function(). It MUST NOT contain
        // any TypeScript syntax (e.g. `as any`) — the TS compiler can't
        // strip casts from inside a template literal, and the resulting
        // raw text would be a SyntaxError to the JS parser.
        try {{
            const w = window as unknown as Record<string, unknown>
            const wrapper = `
                (function (w) {{
                    ${{JS}}
                    if (typeof showSection === 'function') w.showSection = showSection;
                    if (typeof scrollToAnchor === 'function') w.scrollToAnchor = scrollToAnchor;
                    if (typeof copyCode === 'function') w.copyCode = copyCode;
                    if (typeof toggleSidebar === 'function') w.toggleSidebar = toggleSidebar;
                    if (typeof closeSidebarIfMobile === 'function') w.closeSidebarIfMobile = closeSidebarIfMobile;
                    if (typeof isMobile === 'function') w.isMobile = isMobile;
                }})(window);
            `
            void w // silence unused-warning in case the linter checks
            // eslint-disable-next-line no-new-func
            new Function(wrapper)()
        }} catch (e) {{
            // eslint-disable-next-line no-console
            console.error("ContinuumDocs: failed to bind inline JS", e)
        }}
    }}, [])

    // NOTE: no `overflow: auto` on the root — the inline JS uses
    // window.scrollY for scrollspy, which only works when the page
    // (not a nested div) is the scroller. Let the component grow to
    // its natural height; Framer's page handles scrolling.
    return (
        <div
            ref={{rootRef}}
            style={{{{
                width,
                ...(height !== undefined ? {{ height }} : null),
                background: "#fff",
                color: "#0a0a0a",
            }}}}
        >
            <style dangerouslySetInnerHTML={{{{ __html: buildFontCss(props) + "\\n" + CSS }}}} />
            <div dangerouslySetInnerHTML={{{{ __html: html }}}} />
        </div>
    )
}}

addPropertyControls(ContinuumDocs, {{
    logo: {{
        type: ControlType.File,
        title: "Logo",
        allowedFileTypes: ["png", "jpg", "jpeg", "svg", "webp", "gif"],
        description: "Leave empty to hide.",
    }},
    wordmark: {{
        type: ControlType.String,
        title: "Wordmark",
        defaultValue: "",
        placeholder: "e.g. Continuum",
        description: "Leave empty to hide.",
    }},
    byline: {{
        type: ControlType.String,
        title: "Byline",
        defaultValue: "",
        placeholder: "e.g. by <em>ShyftLabs</em>",
        description: "HTML allowed. Leave empty to hide.",
    }},
    fontInter: {{
        type: ControlType.File,
        title: "Inter font",
        allowedFileTypes: ["woff2", "woff", "ttf", "otf"],
    }},
    fontGeist: {{
        type: ControlType.File,
        title: "Geist font",
        allowedFileTypes: ["woff2", "woff", "ttf", "otf"],
    }},
    fontJetBrains: {{
        type: ControlType.File,
        title: "JetBrains Mono font",
        allowedFileTypes: ["woff2", "woff", "ttf", "otf"],
    }},
    fontSpaceMono: {{
        type: ControlType.File,
        title: "Space Mono font",
        allowedFileTypes: ["woff2", "woff", "ttf", "otf"],
    }},
}})
"""

OUT_PATH.write_text(tsx, encoding="utf-8")
size_kb = OUT_PATH.stat().st_size / 1024
print(f"wrote {OUT_PATH} ({size_kb:.1f} KB)")
