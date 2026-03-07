'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import { Menu, X } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';

const navLinks = [
    { href: '/', label: 'Home' },
    { href: '/articles', label: 'Articles' },
    { href: '/projects', label: 'Projects' },
    { href: '/research', label: 'Research' },
    { href: '/about', label: 'About' },
];

export function Navbar() {
    const pathname = usePathname();
    const [menuOpen, setMenuOpen] = useState(false);

    return (
        <header className="fixed top-0 left-0 right-0 z-50">
            <nav className="mx-auto max-w-6xl px-6 py-4">
                <div className="flex items-center justify-between
          bg-white/70 dark:bg-[#1a1917]/80 backdrop-blur-xl
          border border-white/30 dark:border-[#2a2825]/80
          rounded-2xl px-5 py-3 shadow-lg shadow-black/5">
                    {/* Logo */}
                    <Link href="/" className="font-bold text-lg tracking-tight">
                        <span className="gradient-text">Cao</span>
                        <span className="text-[var(--fg)]"> Jiahao</span>
                    </Link>

                    {/* Desktop Links */}
                    <ul className="hidden md:flex items-center gap-1">
                        {navLinks.map(({ href, label }) => {
                            const isActive = pathname === href;
                            return (
                                <li key={href}>
                                    <Link
                                        href={href}
                                        className={`px-4 py-1.5 rounded-xl text-sm font-medium transition-all duration-200
                      ${isActive
                                                ? 'bg-gradient-to-r from-cyan-500/15 to-orange-400/10 text-cyan-500 dark:from-[#c9a55a]/12 dark:to-[#c9a55a]/6 dark:text-[#d4b978]'
                                                : 'text-slate-500 dark:text-[#8a857d] hover:text-slate-800 dark:hover:text-[#e8e4de] hover:bg-slate-100/60 dark:hover:bg-[#2a2825]/60'
                                            }`}
                                    >
                                        {label}
                                    </Link>
                                </li>
                            );
                        })}
                    </ul>

                    <div className="flex items-center gap-2">
                        <ThemeToggle />
                        {/* Mobile Menu Button */}
                        <button
                            className="md:hidden w-9 h-9 flex items-center justify-center rounded-full
                bg-white/10 hover:bg-white/20 dark:bg-white/5
                border border-white/20 dark:border-[#2a2825]
                text-slate-600 dark:text-[#8a857d] transition-all"
                            onClick={() => setMenuOpen(!menuOpen)}
                            aria-label="Toggle menu"
                        >
                            {menuOpen ? <X size={16} /> : <Menu size={16} />}
                        </button>
                    </div>
                </div>

                {/* Mobile Dropdown */}
                {menuOpen && (
                    <div className="md:hidden mt-2 bg-white/90 dark:bg-[#1a1917]/95 backdrop-blur-xl
            border border-white/30 dark:border-[#2a2825]/80
            rounded-2xl p-3 shadow-xl">
                        <ul className="flex flex-col gap-1">
                            {navLinks.map(({ href, label }) => {
                                const isActive = pathname === href;
                                return (
                                    <li key={href}>
                                        <Link
                                            href={href}
                                            onClick={() => setMenuOpen(false)}
                                            className={`block px-4 py-2.5 rounded-xl text-sm font-medium transition-all
                        ${isActive
                                                    ? 'bg-gradient-to-r from-cyan-500/15 to-orange-400/10 text-cyan-500'
                                                    : 'text-slate-600 dark:text-[#8a857d] hover:bg-slate-100/60 dark:hover:bg-[#2a2825]/60'
                                                }`}
                                        >
                                            {label}
                                        </Link>
                                    </li>
                                );
                            })}
                        </ul>
                    </div>
                )}
            </nav>
        </header>
    );
}
