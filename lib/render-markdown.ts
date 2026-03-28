import { marked, Renderer } from 'marked';
import katex from 'katex';

/**
 * Convert heading text to a URL-friendly slug for anchor IDs.
 * Keeps CJK characters, ASCII word chars, and hyphens.
 */
export function slugify(text: string): string {
    return text
        .replace(/<[^>]+>/g, '')          // strip HTML tags (e.g. <span> from KaTeX)
        .trim()
        .toLowerCase()
        .replace(/\s+/g, '-')
        .replace(/[^\u4e00-\u9fa5\w-]/g, ''); // keep CJK, alphanumeric, hyphen
}

/**
 * Render Markdown string to HTML with KaTeX math support and heading ID anchors.
 *
 * Supports:
 *  - Display math:  $$ ... $$
 *  - Inline math:   $ ... $  (single-line or multi-line)
 *
 * Math expressions are rendered FIRST (so marked doesn't mangle them),
 * then the rest is passed through marked for GFM rendering.
 * Each heading receives an `id` attribute for TOC anchor navigation.
 */
export function renderMarkdown(source: string): string {
    // 1. Protect & render display math  $$ ... $$
    let html = source.replace(
        /\$\$([\s\S]+?)\$\$/g,
        (_, tex) => {
            try {
                return katex.renderToString(tex.trim(), { displayMode: true, throwOnError: false, strict: false });
            } catch {
                return `<code>${tex.trim()}</code>`;
            }
        }
    );

    // 2. Protect & render inline/display math  $ ... $  (but NOT $$). Matches across newlines.
    html = html.replace(
        /(?<!\$)\$(?!\$)([\s\S]+?)(?<!\$)\$(?!\$)/g,
        (_, tex) => {
            try {
                const isDisplay = tex.includes('\n');
                return katex.renderToString(tex.trim(), { displayMode: isDisplay, throwOnError: false, strict: false });
            } catch {
                return `<code>${tex.trim()}</code>`;
            }
        }
    );

    // 3. Custom renderer: inject id anchors into headings for TOC navigation
    const renderer = new Renderer();
    renderer.heading = function ({ text, depth }: { text: string; depth: number }) {
        const id = slugify(text);
        return `<h${depth} id="${id}">${text}</h${depth}>\n`;
    };

    // 4. Run marked with custom renderer
    marked.setOptions({ gfm: true, breaks: false });
    html = marked(html, { renderer }) as string;

    return html;
}
