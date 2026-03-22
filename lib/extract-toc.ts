import { slugify } from './render-markdown';

export interface TocItem {
    id: string;
    text: string;
    level: number; // 1 | 2 | 3
}

/**
 * Extract h1-h3 headings from raw markdown for building a Table of Contents.
 * Skips headings that appear inside fenced code blocks.
 */
export function extractToc(markdown: string): TocItem[] {
    const items: TocItem[] = [];
    const lines = markdown.split('\n');
    let inCodeBlock = false;

    for (const line of lines) {
        // Toggle fenced code block tracking
        if (/^```/.test(line.trim())) {
            inCodeBlock = !inCodeBlock;
            continue;
        }
        if (inCodeBlock) continue;

        const match = line.match(/^(#{1,3})\s+(.+)/);
        if (match) {
            const level = match[1].length as 1 | 2 | 3;
            const raw = match[2].trim();
            // Strip common inline markdown so the text reads cleanly
            const text = raw
                .replace(/\*\*(.+?)\*\*/g, '$1')
                .replace(/\*(.+?)\*/g, '$1')
                .replace(/`(.+?)`/g, '$1')
                .replace(/\[(.+?)\]\(.+?\)/g, '$1');
            items.push({ id: slugify(text), text, level });
        }
    }

    return items;
}
