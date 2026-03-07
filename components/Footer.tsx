import Link from 'next/link';
import { Github, Mail, ExternalLink } from 'lucide-react';

export function Footer() {
    return (
        <footer className="border-t border-[var(--border)] py-10 mt-20">
            <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
                <p className="text-sm text-[var(--muted)]">
                    © 2025 <span className="gradient-text font-semibold">Cao Jiahao</span>. All rights reserved.
                </p>
                <div className="flex items-center gap-4">
                    <Link href="https://github.com" target="_blank"
                        className="text-[var(--muted)] hover:text-cyan-500 transition-colors" aria-label="GitHub">
                        <Github size={18} />
                    </Link>
                    <Link href="mailto:caojiahao@stu.xjtu.edu.cn"
                        className="text-[var(--muted)] hover:text-orange-400 transition-colors" aria-label="Email">
                        <Mail size={18} />
                    </Link>
                    <Link href="https://scholar.google.com" target="_blank"
                        className="text-[var(--muted)] hover:text-cyan-500 transition-colors flex items-center gap-1 text-xs">
                        <ExternalLink size={14} />
                        Scholar
                    </Link>
                </div>
            </div>
        </footer>
    );
}
