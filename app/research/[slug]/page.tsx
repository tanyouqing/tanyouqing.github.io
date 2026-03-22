import { getContentBySlug, getAllContent } from '@/lib/mdx';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, FileText, Code2, Calendar } from 'lucide-react';
import { renderMarkdown } from '@/lib/render-markdown';
import { extractToc } from '@/lib/extract-toc';
import { TableOfContents } from '@/components/TableOfContents';

export async function generateStaticParams() {
    const items = getAllContent('research');
    // Next.js dev server has a bug with Unicode params in static export mode. 
    // We encode during development to fix local 500s, but use native strings for production github pages build.
    return items.map(item => ({ 
        slug: process.env.NODE_ENV === 'development' ? encodeURIComponent(item.slug) : item.slug 
    }));
}

export default function ResearchDetailPage({ params }: { params: { slug: string } }) {
    const result = getContentBySlug('research', params.slug);
    if (!result) notFound();
    const { meta, content } = result;

    const html = renderMarkdown(content);
    const toc = extractToc(content);

    return (
        <div className="max-w-6xl mx-auto px-6 py-12">
            {/* Back link */}
            <Link
                href="/research"
                className="inline-flex items-center gap-2 text-sm text-[var(--muted)] hover:text-orange-400 dark:hover:text-[#d4b978] transition-colors mb-8"
            >
                <ArrowLeft size={14} /> 返回研究列表
            </Link>

            <div className="flex gap-12">
                {/* ── Main content ── */}
                <article className="min-w-0 flex-1">
                    {/* Header */}
                    <header className="space-y-4 mb-8 pb-8 border-b border-[var(--border)]">
                        <div className="flex flex-wrap gap-1.5">
                            {meta.tags.map(tag => (
                                <span key={tag} className="tag-badge" style={{ pointerEvents: 'none' }}>{tag}</span>
                            ))}
                        </div>

                        <h1 className="text-2xl md:text-3xl font-bold text-[var(--fg)] leading-snug">{meta.title}</h1>

                        {/* Authors */}
                        {meta.authors && meta.authors.length > 0 && (
                            <p className="text-[var(--muted)]">
                                {meta.authors.map((author, idx) => (
                                    <span key={idx}>
                                        {author === 'Cao Jiahao' || author === '曹嘉豪'
                                            ? <strong className="text-[var(--fg)] font-semibold">{author}</strong>
                                            : author}
                                        {idx < meta.authors!.length - 1 ? ', ' : ''}
                                    </span>
                                ))}
                            </p>
                        )}

                        {/* Venue + Date */}
                        {meta.venue && (
                            <p>
                                <span className="italic text-orange-400 dark:text-[#d4b978]">{meta.venue}</span>
                                <span className="text-[var(--muted)] font-mono text-sm ml-2">{meta.date}</span>
                            </p>
                        )}

                        {/* Links */}
                        <div className="flex gap-3 flex-wrap">
                            {meta.pdf && (
                                <a href={meta.pdf} target="_blank" rel="noopener noreferrer"
                                    className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium
                                    bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 transition-all">
                                    <FileText size={14} /> PDF
                                </a>
                            )}
                            {meta.code && (
                                <a href={meta.code} target="_blank" rel="noopener noreferrer"
                                    className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium
                                    border border-[var(--border)] bg-[var(--card)] text-[var(--fg)]
                                    hover:border-cyan-500/30 dark:hover:border-[#c9a55a]/25 transition-all">
                                    <Code2 size={14} /> Code
                                </a>
                            )}
                        </div>
                    </header>

                    {/* Content */}
                    {content.trim() && (
                        <div className="mdx-content" dangerouslySetInnerHTML={{ __html: html }} />
                    )}
                </article>

                {/* ── TOC Sidebar (hidden on small screens) ── */}
                {toc.length > 0 && (
                    <aside className="hidden xl:block">
                        <TableOfContents items={toc} />
                    </aside>
                )}
            </div>
        </div>
    );
}
