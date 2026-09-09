'use client';

import { motion } from 'framer-motion';
import { Github, Mail, Linkedin, ExternalLink, MapPin, GraduationCap, Briefcase } from 'lucide-react';
import Image from 'next/image';

const education = [
    {
        degree: 'M.Phil. in Artificial Intelligence',
        school: 'The Chinese University of Hong Kong, Shenzhen',
        period: '2026.09 — 2028.06',
        description: 'Currently an M.Phil. student advised by Prof. Menglin Liu, with a research focus on agents and reinforcement learning.',
        icon: '🎓',
    },
    {
        degree: 'B.Eng. in Software Engineering',
        school: 'Xi\'an Jiaotong University',
        period: '2022.09 — 2026.06',
        description: 'Major in Software Engineering. Capstone project on backdoor attacks against text-to-image foundation models.',
        icon: '📚',
    },
];

const experience = [
    {
        title: 'Undergraduate Capstone Project',
        org: 'Xi\'an Jiaotong University',
        period: '2025.09 — 2026.06',
        description: 'Studied diffusion models and backdoor attack principles, and explored potential defenses.',
        icon: '🔬',
    },
    {
        title: 'Big Data Project Internship',
        org: 'Chengdu Shangcheng Big Data Co., Ltd.',
        period: '2025.02 — 2025.03',
        description: 'Designed and implemented the "Pinhaofan" restaurant recommender as a project manager, gaining hands-on experience with recommender systems and big-data development.',
        icon: '🛠️',
    },
    {
        title: 'NUS SoC Summer Workshop',
        org: 'National University of Singapore',
        period: '2024.06 — 2024.07',
        description: 'Studied classical machine learning and deep learning models, foundational finance concepts, and AI applications in finance; completed a group stock analysis and prediction project using Vue and Flask.',
        icon: '🛠️',
    }
];

const skills = {
    'Languages': ['Python', 'Java', 'Go', 'C/C++'],
    'Frameworks': ['PyTorch', 'Langgraph', 'Next.js', 'Spring Boot', 'Flask'],
    'Tools': ['Git', 'Linux', 'MySQL'],
    'Research': ['Agents', 'Backdoor Attacks', 'Reinforcement Learning'],
};

const socialLinks = [
    { icon: Github, label: 'GitHub', href: 'https://github.com/tanyouqing', color: 'hover:text-slate-200' },
    { icon: Mail, label: 'Email', href: 'mailto:caojiahao@stu.xjtu.edu.cn', color: 'hover:text-orange-400' },
    { icon: Linkedin, label: 'LinkedIn', href: 'https://linkedin.com', color: 'hover:text-blue-400' },
    { icon: ExternalLink, label: 'Scholar', href: 'https://scholar.google.com', color: 'hover:text-cyan-400' },
];

function TimelineItem({ item, i }: {
    item: { degree?: string; title?: string; school?: string; org?: string; period: string; description: string; icon: string };
    i: number;
}) {
    return (
        <motion.div
            initial={{ opacity: 0, x: -16 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.12, duration: 0.5 }}
            className="relative pl-10"
        >
            {/* Dot */}
            <div className="absolute left-0 top-1 w-7 h-7 rounded-full
        bg-gradient-to-br from-cyan-500/20 to-orange-400/20 dark:from-[#c9a55a]/15 dark:to-[#8a7340]/10
        border border-cyan-500/30 dark:border-[#c9a55a]/25 flex items-center justify-center text-sm">
                {item.icon}
            </div>
            {/* Vertical line */}
            <div className="absolute left-3.5 top-8 bottom-0 w-px bg-gradient-to-b from-cyan-500/30 dark:from-[#c9a55a]/25 to-transparent" />

            <div className="pb-8 space-y-1">
                <h3 className="font-semibold text-[var(--fg)]">{item.degree ?? item.title}</h3>
                <p className="text-sm text-cyan-500 dark:text-[#c9a55a] font-medium">{item.school ?? item.org}</p>
                <p className="text-xs font-mono text-[var(--muted)]">{item.period}</p>
                <p className="text-sm text-[var(--muted)] leading-relaxed mt-1">{item.description}</p>
            </div>
        </motion.div>
    );
}

export default function AboutPage() {
    return (
        <div className="max-w-5xl mx-auto px-6 py-12">
            <div className="mb-10 space-y-2">
                <p className="text-xs font-mono tracking-widest uppercase text-cyan-500 dark:text-[#c9a55a]">About</p>
                <h1 className="text-4xl font-bold text-[var(--fg)]">About Me</h1>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
                {/* ── Left Column: Avatar + Social ── */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="lg:col-span-1 flex flex-col items-center gap-6"
                >
                    {/* Avatar */}
                    <div className="relative group">
                        <div className="w-44 h-44 rounded-full overflow-hidden ring-2 ring-cyan-500/30 dark:ring-[#c9a55a]/25 ring-offset-2 ring-offset-[var(--bg)]
              shadow-2xl shadow-cyan-500/20 dark:shadow-[#c9a55a]/10 relative">
                            {/* 如果你想换回 emoji，可以取消下面注释，并把 Image 标签删掉 */}
                            {/* <div className="w-full h-full bg-gradient-to-br from-cyan-500 via-slate-600 to-orange-400 flex items-center justify-center text-7xl select-none">🎓</div> */}

                            <Image
                                src="/avatar.jpg"
                                alt="Avatar"
                                fill
                                className="object-cover"
                            />
                        </div>
                        <div className="absolute inset-0 rounded-full bg-gradient-to-br from-cyan-500/20 to-orange-400/20
              dark:from-[#c9a55a]/15 dark:to-[#8a7340]/10
              opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>

                    {/* Name & Position */}
                    <div className="text-center space-y-1">
                        <h2 className="text-xl font-bold text-[var(--fg)]">
                            <span className="gradient-text">Cao</span> Jiahao · 曹家豪
                        </h2>
                        <p className="text-sm text-[var(--muted)] flex items-center justify-center gap-1">
                            <GraduationCap size={13} className="text-cyan-500 dark:text-[#c9a55a]" />
                            Artificial Intelligence M.Phil.
                        </p>
                        <p className="text-sm text-[var(--muted)] flex items-center justify-center gap-1">
                            <MapPin size={13} className="text-orange-400 dark:text-[#d4b978]" />
                            The Chinese University of Hong Kong (Shenzhen)
                        </p>
                    </div>

                    {/* Social Links */}
                    <div className="flex gap-3">
                        {socialLinks.map(({ icon: Icon, label, href, color }) => (
                            <a key={label} href={href} target={href.startsWith('mailto') ? undefined : '_blank'}
                                rel="noopener noreferrer"
                                className={`w-9 h-9 flex items-center justify-center rounded-full
                  border border-[var(--border)] bg-[var(--card)]
                  text-[var(--muted)] ${color} transition-all hover:border-cyan-500/40 dark:hover:border-[#c9a55a]/30 hover:scale-110`}
                                aria-label={label}>
                                <Icon size={15} />
                            </a>
                        ))}
                    </div>

                    {/* Skills */}
                    <div className="w-full space-y-4 pt-4 border-t border-[var(--border)]">
                        {Object.entries(skills).map(([category, items]) => (
                            <div key={category} className="space-y-2">
                                <h4 className="text-xs font-mono uppercase tracking-wider text-[var(--muted)]">{category}</h4>
                                <div className="flex flex-wrap gap-1.5">
                                    {items.map(skill => (
                                        <span key={skill} className="tag-badge text-xs">{skill}</span>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </motion.div>

                {/* ── Right Column: Bio + Timeline ── */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.15 }}
                    className="lg:col-span-2 space-y-10"
                >
                    {/* Bio */}
                    <div className="space-y-3">
                        <h3 className="text-xl font-semibold text-[var(--fg)]">Introduction</h3>
                        <div className="space-y-3 text-[var(--muted)] leading-8">
                            <p>
                                Hi! I&apos;m <strong className="text-[var(--fg)]">Jiahao Cao</strong>, from{' '}
                                <strong className="text-[var(--fg)]">The Chinese University of Hong Kong, Shenzhen</strong>, majoring in{' '}
                                <strong className="text-[var(--fg)]">Artificial Intelligence</strong>. I&apos;m currently an M.Phil. student
                                advised by{' '}
                                <a href="https://menglinmileyliu.github.io/index.html" target="_blank" rel="noopener noreferrer"
                                    className="text-[var(--fg)] font-semibold underline decoration-cyan-500/50 dark:decoration-[#c9a55a]/50 underline-offset-2 hover:text-cyan-500 dark:hover:text-[#c9a55a] transition-colors">
                                    Prof. Menglin Liu
                                </a>,
                                and I&apos;m passionate about agents, large-model applications, and cutting-edge development practices.
                            </p>
                            <p>
                                I believe code is more than a tool — it is an extension of thought, and a language for building
                                better digital worlds. In study and practice, I like to understand first principles and pursue
                                clean, efficient solutions.
                            </p>
                            <p>
                                This homepage is where I record and share — technical notes, reflections, projects, and research.
                                Feel free to reach out!
                            </p>
                        </div>
                    </div>

                    {/* Education Timeline */}
                    <div className="space-y-4">
                        <h3 className="text-xl font-semibold text-[var(--fg)] flex items-center gap-2">
                            <GraduationCap size={18} className="text-cyan-500 dark:text-[#c9a55a]" />
                            Education
                        </h3>
                        <div>
                            {education.map((item, i) => (
                                <TimelineItem key={i} item={item} i={i} />
                            ))}
                        </div>
                    </div>

                    {/* Experience Timeline */}
                    <div className="space-y-4">
                        <h3 className="text-xl font-semibold text-[var(--fg)] flex items-center gap-2">
                            <Briefcase size={18} className="text-orange-400 dark:text-[#d4b978]" />
                            Experience
                        </h3>
                        <div>
                            {experience.map((item, i) => (
                                <TimelineItem key={i} item={item} i={i} />
                            ))}
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
