import { getAllContent, getAllTags } from '@/lib/mdx';
import { ResearchClient } from '@/components/ResearchClient';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Research | Cao Jiahao',
    description: '曹嘉豪的学术研究成果',
};

export default function ResearchPage() {
    const papers = getAllContent('research');
    const tags = getAllTags('research');

    return (
        <div className="max-w-4xl mx-auto px-6 py-12">
            <div className="mb-10 space-y-2">
                <p className="text-xs font-mono tracking-widest uppercase text-orange-400">Academic</p>
                <h1 className="text-4xl font-bold text-[var(--fg)]">研究</h1>
                <p className="text-[var(--muted)]">
                    学术论文与研究工作。共 <span className="text-[var(--fg)] font-medium">{papers.length}</span> 篇/项。
                </p>
            </div>
            <ResearchClient papers={papers} tags={tags} />
        </div>
    );
}
