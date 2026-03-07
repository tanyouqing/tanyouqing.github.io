import { marked } from 'marked';
import katex from 'katex';

/**
 * Render Markdown string to HTML with KaTeX math support.
 *
 * Supports:
 *  - Display math:  $$ ... $$
 *  - Inline math:   $ ... $
 *
 * Math expressions are rendered FIRST (so marked doesn't mangle them),
 * then the rest is passed through marked for GFM rendering.
 */
export function renderMarkdown(source: string): string {
    // 1. Protect & render display math  $$ ... $$
    let html = source.replace(
        /\$\$([\s\S]+?)\$\$/g,
        (_, tex) => {
            try {
                return katex.renderToString(tex.trim(), { displayMode: true, throwOnError: false });
            } catch {
                return `<code>${tex.trim()}</code>`;
            }
        }
    );

    // 2. Protect & render inline math  $ ... $  (but NOT $$)
    html = html.replace(
        /(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g,
        (_, tex) => {
            try {
                return katex.renderToString(tex.trim(), { displayMode: false, throwOnError: false });
            } catch {
                return `<code>${tex.trim()}</code>`;
            }
        }
    );

    // 3. Run marked for the rest (headings, lists, code blocks, etc.)
    marked.setOptions({ gfm: true, breaks: false });
    html = marked(html) as string;

    return html;
}
