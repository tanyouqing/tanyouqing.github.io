'use client';

import { motion } from 'framer-motion';
import { Github, Mail, Linkedin, ExternalLink, MapPin, GraduationCap, Briefcase } from 'lucide-react';
import Image from 'next/image';

const education = [
    {
        degree: '人工智能硕士',
        school: '香港中文大学（深圳）-（预计）',
        period: '2026.09 — 预计 2028.06',
        description: '即将入读，主要研究方向为智能体与强化学习。',
        icon: '🎓',
    },
    {
        degree: '软件工程学士',
        school: '西安交通大学',
        period: '2022.09 — 2026.06',
        description: '主修软件工程，毕业设计为扩散模型的后门攻击。',
        icon: '📚',
    },
];

const experience = [
    {
        title: '本科毕业设计',
        org: '西安交通大学',
        period: '2025.09 — 2026.06',
        description: '完成毕业设计项目，学习扩散模型与后门攻击原理，探索防御方法。',
        icon: '🔬',
    },
    {
        title: '大数据项目实习',
        org: '成都上程大数据有限公司',
        period: '2025.02 — 2025.03',
        description: '作为项目经理设计并实现“品好饭”餐厅推荐系统，积累了推荐系统与大数据开发实践经验。',
        icon: '🛠️',
    },
    {
        title: 'NUS Soc Summer Workshop',
        org: '新加坡国立大学',
        period: '2024.06 — 2024.07',
        description: '学习机器学习和深度学习的经典模型、基础的金融概念和AI在金融领域的应用，并与小组合作，完成了基于Vue+Flask的股票分析和预测项目实战',
        icon: '🛠️',
    }
];

const skills = {
    '编程语言': ['Python', 'Java', 'Go', 'C/C++'],
    '框架与库': ['PyTorch', 'Langgraph', 'Next.js', 'Spring Boot', 'Flask'],
    '工具与平台': ['Git', 'Linux', 'MySQL'],
    '研究方向': ['智能体', '后门攻击', '强化学习'],
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
                <h1 className="text-4xl font-bold text-[var(--fg)]">关于我</h1>
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
                                alt="个人头像"
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
                            软件工程 硕士研究生（准）
                        </p>
                        <p className="text-sm text-[var(--muted)] flex items-center justify-center gap-1">
                            <MapPin size={13} className="text-orange-400 dark:text-[#d4b978]" />
                            西安交通大学
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
                        <h3 className="text-xl font-semibold text-[var(--fg)]">个人简介</h3>
                        <div className="space-y-3 text-[var(--muted)] leading-8">
                            <p>
                                你好！我是<strong className="text-[var(--fg)]">曹家豪</strong>，来自
                                <strong className="text-[var(--fg)]">西安交通大学</strong>，主修
                                <strong className="text-[var(--fg)]">软件工程</strong>专业。目前即将开始硕士阶段的学习旅程，
                                对软件系统设计、人工智能应用与前沿开发技术充满热情。
                            </p>
                            <p>
                                我相信代码不仅仅是工具，更是思维的延伸，是构建美好数字世界的语言。
                                在学习与实践中，我喜欢深入理解事物的底层原理，追求优雅、高效的解决方案。
                            </p>
                            <p>
                                这个主页是我记录与分享的地方——包括技术文章、项目作品以及学术探索。
                                欢迎交流！
                            </p>
                        </div>
                    </div>

                    {/* Education Timeline */}
                    <div className="space-y-4">
                        <h3 className="text-xl font-semibold text-[var(--fg)] flex items-center gap-2">
                            <GraduationCap size={18} className="text-cyan-500 dark:text-[#c9a55a]" />
                            教育经历
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
                            实践经历
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
