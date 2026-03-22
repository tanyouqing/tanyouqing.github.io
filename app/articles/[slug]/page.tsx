import { getContentBySlug, getAllContent } from '@/lib/mdx';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, Clock } from 'lucide-react';
import { renderMarkdown } from '@/lib/render-markdown';
import { extractToc } from '@/lib/extract-toc';
import { TableOfContents } from '@/components/TableOfContents';

export async function generateStaticParams() {
    const items = getAllContent('articles');
    // Next.js dev server has a bug with Unicode params in static export mode. 
    // We encode during development to fix local 500s, but use native strings for production github pages build.
    return items.map(item => ({ 
        slug: process.env.NODE_ENV === 'development' ? encodeURIComponent(item.slug) : item.slug 
    }));
}

export default function ArticleDetailPage({ params }: { params: { slug: string } }) {
    const result = getContentBySlug('articles', params.slug);
    if (!result) notFound();
    const { meta, content } = result;

    const html = renderMarkdown(content);
    const toc = extractToc(content);

    return (
        <div className="max-w-6xl mx-auto px-6 py-12">
            {/* Back link */}
            <Link
                href="/articles"
                className="inline-flex items-center gap-2 text-sm text-[var(--muted)] hover:text-cyan-500 dark:hover:text-[#c9a55a] transition-colors mb-8"
            >
                <ArrowLeft size={14} /> 返回文章列表
            </Link>

            <div className="flex gap-12">
                {/* ── Main content ── */}
                <article className="min-w-0 flex-1">
                    <header className="space-y-4 mb-10">
                        <div className="flex flex-wrap gap-1.5">
                            {meta.tags.map(tag => (
                                <span key={tag} className="tag-badge">{tag}</span>
                            ))}
                        </div>
                        <h1 className="text-3xl md:text-4xl font-bold text-[var(--fg)] leading-tight">
                            {meta.title}
                        </h1>
                        <div className="flex items-center gap-4 text-sm text-[var(--muted)]">
                            <span className="flex items-center gap-1 font-mono">
                                <Clock size={13} />{meta.date}
                            </span>
                            {meta.readingTime && <span>约 {meta.readingTime} 分钟阅读</span>}
                        </div>
                        {meta.description && (
                            <p className="text-[var(--muted)] leading-relaxed border-l-2 border-cyan-500/50 dark:border-[#c9a55a]/40 pl-4">
                                {meta.description}
                            </p>
                        )}
                    </header>

                    {/* Rendered Markdown */}
                    <div
                        className="mdx-content"
                        dangerouslySetInnerHTML={{ __html: html }}
                    />
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
