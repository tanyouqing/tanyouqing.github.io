import { getContentBySlug, getAllContent } from '@/lib/mdx';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, Github, ExternalLink, Calendar } from 'lucide-react';
import { marked } from 'marked';

export async function generateStaticParams() {
    const items = getAllContent('projects');
    return items.map(item => ({ slug: item.slug }));
}

const techColors: Record<string, string> = {
    React: 'bg-sky-400/10 text-sky-400 border-sky-400/20',
    Next: 'bg-slate-500/10 text-slate-400 border-slate-400/20',
    TypeScript: 'bg-blue-400/10 text-blue-400 border-blue-400/20',
    Python: 'bg-yellow-400/10 text-yellow-500 border-yellow-400/20',
    Go: 'bg-cyan-400/10 text-cyan-400 border-cyan-400/20',
    Rust: 'bg-orange-400/10 text-orange-400 border-orange-400/20',
    Docker: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    default: 'bg-slate-400/10 text-slate-400 border-slate-400/20',
};

export default function ProjectDetailPage({ params }: { params: { slug: string } }) {
    const result = getContentBySlug('projects', params.slug);
    if (!result) notFound();
    const { meta, content } = result;

    marked.setOptions({ gfm: true, breaks: false });
    const html = marked(content) as string;

    return (
        <article className="max-w-3xl mx-auto px-6 py-12">
            <Link
                href="/projects"
                className="inline-flex items-center gap-2 text-sm text-[var(--muted)] hover:text-cyan-500 transition-colors mb-8"
            >
                <ArrowLeft size={14} /> 返回项目列表
            </Link>

            {/* Hero */}
            <div className="rounded-2xl bg-gradient-to-br from-cyan-500/15 via-transparent to-orange-400/10
        border border-[var(--border)] p-8 mb-8 flex flex-col items-center gap-3 text-center">
                <span className="text-6xl">{meta.image ?? '🛠️'}</span>
                <h1 className="text-3xl font-bold text-[var(--fg)]">{meta.title}</h1>
                {meta.description && (
                    <p className="text-[var(--muted)] max-w-xl leading-relaxed">{meta.description}</p>
                )}

                {/* Links */}
                <div className="flex gap-3 mt-2">
                    {meta.github && (
                        <a href={meta.github} target="_blank" rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium
                border border-[var(--border)] bg-[var(--card)] text-[var(--fg)]
                hover:border-cyan-500/40 transition-all">
                            <Github size={14} /> GitHub
                        </a>
                    )}
                    {meta.demo && (
                        <a href={meta.demo} target="_blank" rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium
                bg-cyan-500 text-white hover:bg-cyan-400 transition-all shadow-lg shadow-cyan-500/20">
                            <ExternalLink size={14} /> Demo
                        </a>
                    )}
                </div>
            </div>

            {/* Meta */}
            <div className="flex flex-wrap gap-4 mb-8 pb-6 border-b border-[var(--border)]">
                <div className="flex items-center gap-1.5 text-sm text-[var(--muted)]">
                    <Calendar size={13} />
                    <span className="font-mono">{meta.date}</span>
                </div>
                {meta.tech && meta.tech.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                        {meta.tech.map(t => {
                            const cls = techColors[t] ?? techColors.default;
                            return (
                                <span key={t} className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono border ${cls}`}>
                                    {t}
                                </span>
                            );
                        })}
                    </div>
                )}
                <div className="flex flex-wrap gap-1.5">
                    {meta.tags.map(tag => (
                        <span key={tag} className="tag-badge" style={{ pointerEvents: 'none' }}>{tag}</span>
                    ))}
                </div>
            </div>

            {/* Content */}
            {content.trim() && (
                <div className="mdx-content" dangerouslySetInnerHTML={{ __html: html }} />
            )}
        </article>
    );
}
