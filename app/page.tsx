'use client';

import { motion } from 'framer-motion';
import { Clock } from '@/components/Clock';
import { TypewriterQuote } from '@/components/TypewriterQuote';
import { NewsTimeline } from '@/components/NewsTimeline';
import { ArrowRight, MapPin, BookOpen, Code2 } from 'lucide-react';
import Link from 'next/link';

const QUOTE = '世间的面，不是吃一碗，少一碗；而是见一面，多一面';

export default function HomePage() {
    return (
        <>
            {/* ───────────── HERO SECTION ───────────── */}
            <section className="relative min-h-[90vh] flex flex-col items-center justify-center overflow-hidden">
                {/* Aurora background */}
                <div className="aurora-bg" />

                {/* Clock — top right */}
                <div className="absolute top-4 right-6 z-10">
                    <Clock />
                </div>

                {/* Central content */}
                <div className="relative z-10 max-w-3xl mx-auto px-6 text-center flex flex-col items-center gap-8">
                    {/* Quote */}
                    <motion.div
                        initial={{ opacity: 0, y: 24 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, ease: 'easeOut' }}
                        className="space-y-3"
                    >
                        <p className="text-xs font-mono tracking-[0.3em] uppercase text-[var(--muted)]">
                            Quote of Life
                        </p>
                        <blockquote
                            className="font-quote text-2xl md:text-3xl font-medium leading-loose md:whitespace-nowrap
                text-slate-700 dark:text-[#e8e4de]"
                        >
                            <TypewriterQuote text={QUOTE} speed={70} />
                        </blockquote>
                    </motion.div>

                    {/* Divider */}
                    <motion.div
                        initial={{ scaleX: 0 }}
                        animate={{ scaleX: 1 }}
                        transition={{ delay: 0.5, duration: 0.6 }}
                        className="w-24 h-px bg-gradient-to-r from-cyan-500 to-orange-400 dark:from-[#c9a55a] dark:to-[#e2c67e] rounded-full"
                    />

                    {/* Name & Title */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.7, duration: 0.6 }}
                        className="space-y-2"
                    >
                        <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
                            <span className="gradient-text">Cao</span>
                            <span className="text-[var(--fg)]"> Jiahao</span>
                        </h1>
                        <p className="text-base text-[var(--muted)] flex items-center justify-center gap-1.5">
                            <MapPin size={14} className="text-cyan-500 dark:text-[#c9a55a]" />
                            西安交通大学 · 软件工程
                        </p>
                    </motion.div>

                    {/* CTA Buttons */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 1.1, duration: 0.5 }}
                        className="flex flex-wrap gap-3 items-center justify-center"
                    >
                        <Link href="/about"
                            className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm
                bg-gradient-to-r from-cyan-500 to-cyan-600 dark:from-[#c9a55a] dark:to-[#b8944f] text-white
                hover:from-cyan-400 hover:to-cyan-500 dark:hover:from-[#d4b978] dark:hover:to-[#c9a55a] transition-all shadow-lg shadow-cyan-500/25 dark:shadow-[#c9a55a]/20">
                            了解我 <ArrowRight size={14} />
                        </Link>
                        <Link href="/research"
                            className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm
                border border-[var(--border)] text-[var(--fg)]
                hover:bg-[var(--card)] transition-all">
                            查看研究 <BookOpen size={14} />
                        </Link>
                    </motion.div>
                </div>

                {/* Scroll indicator */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1.8, duration: 0.5 }}
                    className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1"
                >
                    <div className="w-px h-12 bg-gradient-to-b from-transparent to-cyan-500/50 dark:to-[#c9a55a]/40" />
                    <div className="w-1 h-1 rounded-full bg-cyan-500 dark:bg-[#c9a55a] animate-bounce" />
                </motion.div>
            </section>

            {/* ───────────── INTRO + NEWS SECTION ───────────── */}
            <section className="max-w-6xl mx-auto px-6 py-20">
                <div className="grid grid-cols-1 md:grid-cols-5 gap-12">
                    {/* Left: Personal Intro */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.6 }}
                        className="md:col-span-3 space-y-6"
                    >
                        <div className="space-y-2">
                            <p className="text-xs font-mono tracking-widest uppercase text-cyan-500 dark:text-[#c9a55a]">About Me</p>
                            <h2 className="text-3xl font-bold text-[var(--fg)]">个人简介</h2>
                        </div>

                        <p className="text-[var(--muted)] leading-8 text-base">
                            你好！我是曹家豪，来自<span className="text-[var(--fg)] font-medium">西安交通大学</span>，
                            主修<span className="text-[var(--fg)] font-medium">软件工程</span>专业。
                            即将踏上硕士研究生的学习旅程，对前沿人工智能技术和软件系统设计充满热情。
                        </p>
                        <p className="text-[var(--muted)] leading-8 text-base">
                            我热爱以代码构建优雅的解决方案，对强化学习、人工智能应用及软件工程方法论有浓厚兴趣。
                            这里记录我的学习思考、项目探索和研究成果。
                        </p>

                        {/* Tags */}
                        <div className="flex flex-wrap gap-2 pt-2">
                            {['软件工程', '后门攻击', '人工智能', 'Full Stack', 'Research', '西安交通大学'].map(tag => (
                                <span key={tag} className="tag-badge">{tag}</span>
                            ))}
                        </div>

                        {/* Quick Links */}
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
                            {[
                                { icon: BookOpen, label: '查看文章', href: '/articles', color: 'text-violet-400 dark:text-[#d4b978]' },
                                { icon: Code2, label: '浏览项目', href: '/projects', color: 'text-cyan-400 dark:text-[#c9a55a]' },
                                { icon: ArrowRight, label: '学术研究', href: '/research', color: 'text-orange-400 dark:text-[#e2c67e]' },
                            ].map(({ icon: Icon, label, href, color }) => (
                                <Link key={href} href={href}
                                    className="flex items-center gap-2.5 px-4 py-3 rounded-xl
                    border border-[var(--border)] bg-[var(--card)]
                    hover:border-cyan-500/40 dark:hover:border-[#c9a55a]/30 transition-all group card-hover">
                                    <Icon size={16} className={color} />
                                    <span className="text-sm font-medium text-[var(--fg)]">{label}</span>
                                    <ArrowRight size={12} className="ml-auto text-[var(--muted)] group-hover:text-cyan-500 dark:group-hover:text-[#c9a55a] group-hover:translate-x-1 transition-all" />
                                </Link>
                            ))}
                        </div>
                    </motion.div>

                    {/* Right: News Timeline */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.6, delay: 0.2 }}
                        className="md:col-span-2 space-y-5"
                    >
                        <div className="space-y-2">
                            <p className="text-xs font-mono tracking-widest uppercase text-orange-400 dark:text-[#d4b978]">Latest</p>
                            <h2 className="text-3xl font-bold text-[var(--fg)]">动态</h2>
                        </div>
                        <NewsTimeline />
                    </motion.div>
                </div>
            </section>
        </>
    );
}
