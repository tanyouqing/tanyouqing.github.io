'use client';

import { useTheme } from 'next-themes';
import { Sun, Moon } from 'lucide-react';
import { useEffect, useState } from 'react';

export function ThemeToggle() {
    const { theme, setTheme } = useTheme();
    const [mounted, setMounted] = useState(false);

    useEffect(() => setMounted(true), []);
    if (!mounted) return <div className="w-9 h-9" />;

    return (
        <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="w-9 h-9 flex items-center justify-center rounded-full
        bg-white/10 hover:bg-white/20 dark:bg-white/5 dark:hover:bg-white/10
        border border-white/20 dark:border-[#2a2825]
        text-slate-600 dark:text-[#8a857d]
        transition-all duration-200 backdrop-blur-sm"
            aria-label="Toggle theme"
        >
            {theme === 'dark' ? (
                <Sun size={16} className="text-[#d4b978]" />
            ) : (
                <Moon size={16} className="text-cyan-500" />
            )}
        </button>
    );
}
