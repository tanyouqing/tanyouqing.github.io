import { getContentBySlug, getAllContent } from '@/lib/mdx';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, FileText, Code2, Calendar } from 'lucide-react';
import { renderMarkdown } from '@/lib/render-markdown';

export async function generateStaticParams() {
    const items = getAllContent('research');
    return items.map(item => ({ slug: item.slug }));
}

export default function ResearchDetailPage({ params }: { params: { slug: string } }) {
    const result = getContentBySlug('research', params.slug);
    if (!result) notFound();
    const { meta, content } = result;

    const html = renderMarkdown(content);

    return (
        <article className="max-w-3xl mx-auto px-6 py-12">
            <Link
                href="/research"
                className="inline-flex items-center gap-2 text-sm text-[var(--muted)] hover:text-orange-400 transition-colors mb-8"
            >
                <ArrowLeft size={14} /> 返回研究列表
            </Link>

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
                        <span className="italic text-orange-400">{meta.venue}</span>
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
                hover:border-cyan-500/30 transition-all">
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
    );
}
