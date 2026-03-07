'use client';

import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import { TagFilter } from '@/components/TagFilter';
import { ContentMeta } from '@/lib/mdx';
import { FileText, Code2, ChevronDown, Copy, Check, Search, ExternalLink } from 'lucide-react';

interface ResearchClientProps {
    papers: ContentMeta[];
    tags: string[];
}

function BibTeXPanel({ bibtex }: { bibtex: string }) {
    const [open, setOpen] = useState(false);
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        await navigator.clipboard.writeText(bibtex);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="mt-2">
            <button
                onClick={() => setOpen(!open)}
                className="flex items-center gap-1.5 text-xs text-[var(--muted)] hover:text-[var(--fg)] transition-colors font-mono"
            >
                <ChevronDown size={12} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
                BibTeX
            </button>
            <AnimatePresence>
                {open && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.25 }}
                        className="overflow-hidden"
                    >
                        <div className="mt-2 relative rounded-xl bg-slate-900 dark:bg-black/40 border border-[var(--border)] p-4">
                            <button
                                onClick={handleCopy}
                                className="absolute top-3 right-3 p-1.5 rounded-lg bg-white/5 hover:bg-white/10
                  text-slate-400 hover:text-white transition-all"
                                aria-label="Copy BibTeX"
                            >
                                {copied ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
                            </button>
                            <pre className="text-xs font-mono text-slate-300 overflow-x-auto whitespace-pre-wrap">
                                {bibtex}
                            </pre>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

export function ResearchClient({ papers, tags }: ResearchClientProps) {
    const [selectedTag, setSelectedTag] = useState<string | null>(null);
    const [query, setQuery] = useState('');

    const filtered = useMemo(() => {
        return papers.filter(p => {
            const matchTag = !selectedTag || p.tags.includes(selectedTag) || p.venue === selectedTag;
            const matchQuery = !query || p.title.toLowerCase().includes(query.toLowerCase());
            return matchTag && matchQuery;
        });
    }, [papers, selectedTag, query]);

    return (
        <div className="space-y-8">
            <div className="space-y-4">
                <div className="relative">
                    <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                    <input
                        type="text"
                        placeholder="搜索论文..."
                        value={query}
                        onChange={e => setQuery(e.target.value)}
                        className="w-full pl-10 pr-4 py-2.5 rounded-xl
              bg-[var(--card)] border border-[var(--border)]
              text-[var(--fg)] placeholder:text-[var(--muted)]
              focus:outline-none focus:border-orange-400/50 text-sm transition-colors"
                    />
                </div>
                <TagFilter tags={tags} selected={selectedTag} onSelect={setSelectedTag} />
            </div>

            {/* Paper List */}
            <div className="space-y-1">
                {filtered.length === 0 ? (
                    <p className="py-12 text-center text-[var(--muted)]">暂无匹配的论文。</p>
                ) : (
                    filtered.map((paper, i) => (
                        <motion.div
                            key={paper.slug}
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.05 }}
                            className="py-5 border-b border-[var(--border)] last:border-0 space-y-2"
                        >
                            {/* Title */}
                            <Link href={`/research/${paper.slug}`}>
                                <h2 className="text-base font-semibold text-[var(--fg)] leading-snug hover:text-orange-400 transition-colors cursor-pointer">
                                    {paper.title}
                                </h2>
                            </Link>

                            {/* Authors */}
                            {paper.authors && paper.authors.length > 0 && (
                                <p className="text-sm text-[var(--muted)]">
                                    {paper.authors.map((author, idx) => (
                                        <span key={idx}>
                                            {author === 'Cao Jiahao' || author === '曹嘉豪'
                                                ? <strong className="text-[var(--fg)] font-semibold">{author}</strong>
                                                : author}
                                            {idx < paper.authors!.length - 1 ? ', ' : ''}
                                        </span>
                                    ))}
                                </p>
                            )}

                            {/* Venue */}
                            {paper.venue && (
                                <p className="text-sm">
                                    <span className="italic text-orange-400/80">{paper.venue}</span>
                                    <span className="text-[var(--muted)] ml-2 font-mono text-xs">{paper.date}</span>
                                </p>
                            )}

                            {/* Links */}
                            <div className="flex items-center gap-3 flex-wrap">
                                {paper.pdf && (
                                    <a href={paper.pdf} target="_blank" rel="noopener noreferrer"
                                        className="inline-flex items-center gap-1.5 text-xs font-medium text-red-400 hover:text-red-300 transition-colors">
                                        <FileText size={13} /> PDF
                                    </a>
                                )}
                                {paper.code && (
                                    <a href={paper.code} target="_blank" rel="noopener noreferrer"
                                        className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-slate-200 transition-colors">
                                        <Code2 size={13} /> Code
                                    </a>
                                )}
                                <div className="flex gap-1.5">
                                    {paper.tags.slice(0, 3).map(tag => (
                                        <span key={tag} className="tag-badge" style={{ pointerEvents: 'none' }}>{tag}</span>
                                    ))}
                                </div>
                            </div>

                            {/* BibTeX */}
                            {paper.bibtex && <BibTeXPanel bibtex={paper.bibtex} />}
                        </motion.div>
                    ))
                )}
            </div>
        </div>
    );
}
