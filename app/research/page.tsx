import { getAllContent, getAllTags } from '@/lib/mdx';
import { ResearchClient } from '@/components/ResearchClient';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Research | Cao Jiahao',
    description: 'Academic research by Jiahao Cao',
};

export default function ResearchPage() {
    const papers = getAllContent('research');
    const tags = getAllTags('research');

    return (
        <div className="max-w-4xl mx-auto px-6 py-12">
            <div className="mb-10 space-y-2">
                <p className="text-xs font-mono tracking-widest uppercase text-orange-400 dark:text-[#d4b978]">Academic</p>
                <h1 className="text-4xl font-bold text-[var(--fg)]">Research</h1>
                <p className="text-[var(--muted)]">
                    Papers and research work. <span className="text-[var(--fg)] font-medium">{papers.length}</span> items in total.
                </p>
            </div>
            <ResearchClient papers={papers} tags={tags} />
        </div>
    );
}
