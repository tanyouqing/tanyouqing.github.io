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
 * Strategy:
 *  1. Extract all math blocks ($$ ... $$ and $ ... $) and replace with unique
 *     placeholders. This prevents marked from mangling KaTeX-rendered HTML
 *     (e.g., GFM strikethrough treating `~` characters as `~~` syntax).
 *  2. Render the placeholder text with marked (GFM + custom heading renderer).
 *  3. Substitute placeholders back with KaTeX-rendered HTML.
 *
 * Supports:
 *  - Display math:  $$ ... $$
 *  - Inline math:   $ ... $
 */
export function renderMarkdown(source: string): string {
    const mathBlocks: Array<{ placeholder: string; html: string }> = [];
    let counter = 0;

    // Helper: render a TeX string and store it, return a placeholder
    const storeMath = (tex: string, displayMode: boolean): string => {
        const placeholder = `\x02MATH_${counter++}\x03`;
        let mathHtml: string;
        try {
            mathHtml = katex.renderToString(tex.trim(), {
                displayMode,
                throwOnError: false,
                strict: false,
            });
        } catch {
            mathHtml = `<code>${tex.trim()}</code>`;
        }
        mathBlocks.push({ placeholder, html: mathHtml });
        return placeholder;
    };

    // 1. Extract display math  $$ ... $$  (must come before inline $ match)
    let processed = source.replace(
        /\$\$([\s\S]+?)\$\$/g,
        (_, tex) => storeMath(tex, true)
    );

    // 2. Extract inline math  $ ... $  (but NOT $$)
    processed = processed.replace(
        /(?<!\$)\$(?!\$)([\s\S]+?)(?<!\$)\$(?!\$)/g,
        (_, tex) => {
            const isDisplay = tex.includes('\n');
            return storeMath(tex, isDisplay);
        }
    );

    // 3. Custom renderer: inject id anchors into headings for TOC navigation
    const renderer = new Renderer();
    renderer.heading = function ({ text, depth }: { text: string; depth: number }) {
        const id = slugify(text);
        return `<h${depth} id="${id}">${text}</h${depth}>\n`;
    };

    // 4. Run marked on the placeholder text (marked never sees KaTeX HTML)
    marked.setOptions({ gfm: true, breaks: false });
    let html = marked(processed, { renderer }) as string;

    // 5. Restore KaTeX HTML by replacing placeholders
    for (const { placeholder, html: mathHtml } of mathBlocks) {
        // Escape the placeholder for use in a RegExp (it only contains word chars and digits)
        html = html.split(placeholder).join(mathHtml);
    }

    return html;
}
