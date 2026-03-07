import { getAllContent, getAllTags } from '@/lib/mdx';
import { ArticlesClient } from '@/components/ArticlesClient';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Articles | Cao Jiahao',
    description: '曹嘉豪的技术文章与学习笔记',
};

export default function ArticlesPage() {
    const articles = getAllContent('articles');
    const tags = getAllTags('articles');

    return (
        <div className="max-w-3xl mx-auto px-6 py-12">
            {/* Header */}
            <div className="mb-10 space-y-2">
                <p className="text-xs font-mono tracking-widest uppercase text-violet-400">Writing</p>
                <h1 className="text-4xl font-bold text-[var(--fg)]">文章</h1>
                <p className="text-[var(--muted)]">
                    记录技术思考、学习笔记与个人观点。共 <span className="text-[var(--fg)] font-medium">{articles.length}</span> 篇文章。
                </p>
            </div>

            <ArticlesClient articles={articles} tags={tags} />
        </div>
    );
}
