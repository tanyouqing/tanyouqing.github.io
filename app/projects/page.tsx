import { getAllContent, getAllTags } from '@/lib/mdx';
import { ProjectsClient } from '@/components/ProjectsClient';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Projects | Cao Jiahao',
    description: '曹嘉豪的开发项目展示',
};

export default function ProjectsPage() {
    const projects = getAllContent('projects');
    const tags = getAllTags('projects');

    return (
        <div className="max-w-6xl mx-auto px-6 py-12">
            <div className="mb-10 space-y-2">
                <p className="text-xs font-mono tracking-widest uppercase text-cyan-400 dark:text-[#c9a55a]">Portfolio</p>
                <h1 className="text-4xl font-bold text-[var(--fg)]">项目</h1>
                <p className="text-[var(--muted)]">
                    动手实践与探索。共 <span className="text-[var(--fg)] font-medium">{projects.length}</span> 个项目。
                </p>
            </div>
            <ProjectsClient projects={projects} tags={tags} />
        </div>
    );
}
