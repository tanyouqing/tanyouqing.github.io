import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

export interface ContentMeta {
    slug: string;
    title: string;
    date: string;
    description: string;
    tags: string[];
    category?: string;
    venue?: string;       // for research: conference/journal
    authors?: string[];   // for research
    pdf?: string;
    code?: string;
    demo?: string;
    github?: string;
    tech?: string[];      // for projects
    image?: string;
    bibtex?: string;
    readingTime?: number;
}

export type ContentType = 'articles' | 'projects' | 'research';

const contentDir = (type: ContentType) =>
    path.join(process.cwd(), 'content', type);

export function getAllContent(type: ContentType): ContentMeta[] {
    const dir = contentDir(type);
    if (!fs.existsSync(dir)) return [];

    const files = fs.readdirSync(dir).filter(f => f.endsWith('.md') || f.endsWith('.mdx'));

    return files.map(filename => {
        const slug = filename.replace(/\.(md|mdx)$/, '');
        const fullPath = path.join(dir, filename);
        const raw = fs.readFileSync(fullPath, 'utf-8');
        const { data, content } = matter(raw);

        // Estimate reading time
        const words = content.split(/\s+/).length;
        const readingTime = Math.ceil(words / 200);

        return {
            slug,
            title: data.title ?? slug,
            date: data.date ?? '',
            description: data.description ?? '',
            tags: data.tags ?? [],
            category: data.category,
            venue: data.venue,
            authors: data.authors,
            pdf: data.pdf,
            code: data.code,
            demo: data.demo,
            github: data.github,
            tech: data.tech,
            image: data.image,
            bibtex: data.bibtex,
            readingTime,
        } as ContentMeta;
    }).sort((a, b) => (a.date < b.date ? 1 : -1));
}

export function getContentBySlug(type: ContentType, slug: string): { meta: ContentMeta; content: string } | null {
    const dir = contentDir(type);
    const filePath = path.join(dir, `${slug}.md`);
    const mdxPath = path.join(dir, `${slug}.mdx`);
    const fp = fs.existsSync(filePath) ? filePath : fs.existsSync(mdxPath) ? mdxPath : null;

    if (!fp) return null;
    const raw = fs.readFileSync(fp, 'utf-8');
    const { data, content } = matter(raw);
    const words = content.split(/\s+/).length;

    return {
        meta: {
            slug,
            title: data.title ?? slug,
            date: data.date ?? '',
            description: data.description ?? '',
            tags: data.tags ?? [],
            category: data.category,
            venue: data.venue,
            authors: data.authors,
            pdf: data.pdf,
            code: data.code,
            demo: data.demo,
            github: data.github,
            tech: data.tech,
            image: data.image,
            bibtex: data.bibtex,
            readingTime: Math.ceil(words / 200),
        },
        content,
    };
}

export function getAllTags(type: ContentType): string[] {
    const items = getAllContent(type);
    const tags = new Set<string>();
    items.forEach(item => item.tags.forEach(t => tags.add(t)));
    return Array.from(tags).sort();
}
