import { getAllContent, getAllTags } from '@/lib/mdx';
import { ArticlesClient } from '@/components/ArticlesClient';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Articles | Cao Jiahao',
    description: 'Technical articles and study notes by Jiahao Cao',
};

export default function ArticlesPage() {
    const articles = getAllContent('articles');
    const tags = getAllTags('articles');

    return (
        <div className="max-w-3xl mx-auto px-6 py-12">
            {/* Header */}
            <div className="mb-10 space-y-2">
                <p className="text-xs font-mono tracking-widest uppercase text-violet-400 dark:text-[#d4b978]">Writing</p>
                <h1 className="text-4xl font-bold text-[var(--fg)]">Articles</h1>
                <p className="text-[var(--muted)]">
                    Technical notes, study logs, and personal reflections. <span className="text-[var(--fg)] font-medium">{articles.length}</span> articles in total.
                </p>
            </div>

            <ArticlesClient articles={articles} tags={tags} />
        </div>
    );
}
