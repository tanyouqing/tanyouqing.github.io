'use client';

import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { TagFilter } from '@/components/TagFilter';
import { ContentMeta } from '@/lib/mdx';
import { Clock, Search, ArrowRight } from 'lucide-react';

interface ArticlesClientProps {
    articles: ContentMeta[];
    tags: string[];
}

export function ArticlesClient({ articles, tags }: ArticlesClientProps) {
    const [selectedTag, setSelectedTag] = useState<string | null>(null);
    const [query, setQuery] = useState('');

    const filtered = useMemo(() => {
        return articles.filter(a => {
            const matchTag = !selectedTag || a.tags.includes(selectedTag);
            const matchQuery = !query || a.title.toLowerCase().includes(query.toLowerCase()) || a.description.toLowerCase().includes(query.toLowerCase());
            return matchTag && matchQuery;
        });
    }, [articles, selectedTag, query]);

    return (
        <div className="space-y-8">
            {/* Search + Tags */}
            <div className="space-y-4">
                <div className="relative">
                    <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                    <input
                        type="text"
                        placeholder="搜索文章..."
                        value={query}
                        onChange={e => setQuery(e.target.value)}
                        className="w-full pl-10 pr-4 py-2.5 rounded-xl
              bg-[var(--card)] border border-[var(--border)]
              text-[var(--fg)] placeholder:text-[var(--muted)]
              focus:outline-none focus:border-cyan-500/50 dark:focus:border-[#c9a55a]/40 text-sm transition-colors"
                    />
                </div>
                <TagFilter tags={tags} selected={selectedTag} onSelect={setSelectedTag} />
            </div>

            {/* Article List */}
            <div className="divide-y divide-[var(--border)]">
                {filtered.length === 0 ? (
                    <p className="py-12 text-center text-[var(--muted)]">暂无匹配的文章。</p>
                ) : (
                    filtered.map((article, i) => (
                        <motion.article
                            key={article.slug}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.06 }}
                            className="py-6 group"
                        >
                            <Link href={`/articles/${encodeURIComponent(article.slug)}`} className="block space-y-2">
                                <div className="flex items-start justify-between gap-4">
                                    <h2 className="text-lg font-semibold text-[var(--fg)] group-hover:text-cyan-500 dark:group-hover:text-[#c9a55a] transition-colors leading-snug">
                                        {article.title}
                                    </h2>
                                    <ArrowRight size={16} className="flex-shrink-0 mt-1 text-[var(--muted)] group-hover:text-cyan-500 dark:group-hover:text-[#c9a55a] group-hover:translate-x-1 transition-all" />
                                </div>
                                {article.description && (
                                    <p className="text-sm text-[var(--muted)] leading-relaxed line-clamp-2">
                                        {article.description}
                                    </p>
                                )}
                                <div className="flex items-center gap-4 pt-1">
                                    <span className="text-xs text-[var(--muted)] font-mono flex items-center gap-1">
                                        <Clock size={12} />
                                        {article.date}
                                    </span>
                                    {article.readingTime && (
                                        <span className="text-xs text-[var(--muted)]">约 {article.readingTime} 分钟阅读</span>
                                    )}
                                    <div className="flex gap-1.5 flex-wrap">
                                        {article.tags.slice(0, 3).map(tag => (
                                            <span key={tag} className="tag-badge" style={{ pointerEvents: 'none' }}>{tag}</span>
                                        ))}
                                    </div>
                                </div>
                            </Link>
                        </motion.article>
                    ))
                )}
            </div>
        </div>
    );
}
