'use client';

import { motion } from 'framer-motion';
import { Clock as ClockIcon, Award, FileText, Zap } from 'lucide-react';

interface NewsItem {
    date: string;
    type: 'paper' | 'award' | 'post' | 'event';
    content: string;
    link?: string;
}

const newsData: NewsItem[] = [
    {
        date: '2026.04',
        type: 'event',
        content: 'Joined ByteDance Openviking team as an intern, exploring Agent Memory mechanisms',
    },
    {
        date: '2026.03',
        type: 'event',
        content: 'Thesis project: backdoor attack system based on text-to-image foundation models',
    },
    {
        date: '2025.03',
        type: 'event',
        content: 'Internship at Chengdu Shangcheng Big Data, focusing on big-data algorithms and recommender systems',
    },
];

const typeConfig = {
    paper: { icon: FileText, color: 'text-cyan-500', bg: 'bg-cyan-500/10 border-cyan-500/20' },
    award: { icon: Award, color: 'text-orange-400', bg: 'bg-orange-400/10 border-orange-400/20' },
    post: { icon: ClockIcon, color: 'text-violet-400', bg: 'bg-violet-400/10 border-violet-400/20' },
    event: { icon: Zap, color: 'text-emerald-400', bg: 'bg-emerald-400/10 border-emerald-400/20' },
};

export function NewsTimeline() {
    return (
        <div className="space-y-4">
            {newsData.map((item, i) => {
                const { icon: Icon, color, bg } = typeConfig[item.type];
                return (
                    <motion.div
                        key={i}
                        initial={{ opacity: 0, x: -20 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: i * 0.1, duration: 0.4 }}
                        className="flex gap-4 items-start group"
                    >
                        {/* Icon */}
                        <div className={`flex-shrink-0 w-8 h-8 rounded-lg border flex items-center justify-center mt-0.5 ${bg}`}>
                            <Icon size={14} className={color} />
                        </div>
                        {/* Content */}
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                                <span className="text-xs font-mono text-[var(--muted)]">{item.date}</span>
                            </div>
                            <p className="text-sm text-[var(--fg)] leading-relaxed mt-0.5">
                                {item.link ? (
                                    <a href={item.link} className="hover:text-cyan-500 dark:hover:text-[#c9a55a] transition-colors">
                                        {item.content}
                                    </a>
                                ) : (
                                    item.content
                                )}
                            </p>
                        </div>
                    </motion.div>
                );
            })}
        </div>
    );
}
