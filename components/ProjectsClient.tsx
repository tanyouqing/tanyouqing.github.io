'use client';

import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { TagFilter } from '@/components/TagFilter';
import { ContentMeta } from '@/lib/mdx';
import { Github, ExternalLink, Search } from 'lucide-react';

interface ProjectsClientProps {
    projects: ContentMeta[];
    tags: string[];
}

const techColors: Record<string, string> = {
    React: 'bg-sky-400/10 text-sky-400 border-sky-400/20',
    Next: 'bg-slate-500/10 text-slate-400 border-slate-400/20',
    TypeScript: 'bg-blue-400/10 text-blue-400 border-blue-400/20',
    Python: 'bg-yellow-400/10 text-yellow-500 border-yellow-400/20',
    Go: 'bg-cyan-400/10 text-cyan-400 border-cyan-400/20',
    Rust: 'bg-orange-400/10 text-orange-400 border-orange-400/20',
    Docker: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    Kubernetes: 'bg-blue-600/10 text-blue-400 border-blue-600/20',
    default: 'bg-slate-400/10 text-slate-400 border-slate-400/20',
};

function TechBadge({ tech }: { tech: string }) {
    const cls = techColors[tech] ?? techColors.default;
    return (
        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono border ${cls}`}>
            {tech}
        </span>
    );
}

export function ProjectsClient({ projects, tags }: ProjectsClientProps) {
    const router = useRouter();
    const [selectedTag, setSelectedTag] = useState<string | null>(null);
    const [query, setQuery] = useState('');

    const filtered = useMemo(() => {
        return projects.filter(p => {
            const matchTag = !selectedTag || p.tags.includes(selectedTag);
            const matchQuery = !query || p.title.toLowerCase().includes(query.toLowerCase()) || p.description.toLowerCase().includes(query.toLowerCase());
            return matchTag && matchQuery;
        });
    }, [projects, selectedTag, query]);

    return (
        <div className="space-y-8">
            {/* Controls */}
            <div className="space-y-4">
                <div className="relative">
                    <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                    <input
                        type="text"
                        placeholder="搜索项目..."
                        value={query}
                        onChange={e => setQuery(e.target.value)}
                        className="w-full pl-10 pr-4 py-2.5 rounded-xl
              bg-[var(--card)] border border-[var(--border)]
              text-[var(--fg)] placeholder:text-[var(--muted)]
              focus:outline-none focus:border-cyan-500/50 text-sm transition-colors"
                    />
                </div>
                <TagFilter tags={tags} selected={selectedTag} onSelect={setSelectedTag} />
            </div>

            {/* Grid */}
            {filtered.length === 0 ? (
                <p className="py-12 text-center text-[var(--muted)]">暂无匹配的项目。</p>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                    {filtered.map((project, i) => (
                        <motion.div
                            key={project.slug}
                            initial={{ opacity: 0, scale: 0.96 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: i * 0.07 }}
                            className="group flex flex-col rounded-2xl border border-[var(--border)] bg-[var(--card)]
                overflow-hidden card-hover cursor-pointer"
                            onClick={() => router.push(`/projects/${project.slug}`)}
                        >
                            {/* Gradient Header */}
                            <div className="h-32 bg-gradient-to-br from-cyan-500/20 via-transparent to-orange-400/10
                flex items-center justify-center relative overflow-hidden">
                                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500
                  bg-gradient-to-br from-cyan-500/10 to-orange-400/10" />
                                <span className="text-5xl">
                                    {project.image ?? '🛠️'}
                                </span>
                            </div>

                            {/* Card Body */}
                            <div className="p-5 flex flex-col gap-3 flex-1">
                                <div className="flex items-start justify-between gap-2">
                                    <h2 className="font-semibold text-[var(--fg)] group-hover:text-cyan-500 transition-colors leading-snug">
                                        {project.title}
                                    </h2>
                                    <div className="flex gap-2 flex-shrink-0">
                                        {project.github && (
                                            <a href={project.github} target="_blank" rel="noopener noreferrer"
                                                onClick={e => e.stopPropagation()}
                                                className="text-[var(--muted)] hover:text-[var(--fg)] transition-colors" aria-label="GitHub">
                                                <Github size={15} />
                                            </a>
                                        )}
                                        {project.demo && (
                                            <a href={project.demo} target="_blank" rel="noopener noreferrer"
                                                onClick={e => e.stopPropagation()}
                                                className="text-[var(--muted)] hover:text-cyan-500 transition-colors" aria-label="Demo">
                                                <ExternalLink size={15} />
                                            </a>
                                        )}
                                    </div>
                                </div>

                                <p className="text-sm text-[var(--muted)] leading-relaxed line-clamp-3 flex-1">
                                    {project.description || '暂无项目简介。'}
                                </p>

                                {/* Tech Stack */}
                                {project.tech && project.tech.length > 0 && (
                                    <div className="flex flex-wrap gap-1.5">
                                        {project.tech.slice(0, 4).map(t => <TechBadge key={t} tech={t} />)}
                                    </div>
                                )}

                                <div className="flex flex-wrap gap-1.5 pt-1 border-t border-[var(--border)]">
                                    {project.tags.slice(0, 3).map(tag => (
                                        <span key={tag} className="tag-badge" style={{ pointerEvents: 'none' }}>{tag}</span>
                                    ))}
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            )}
        </div>
    );
}
