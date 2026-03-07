'use client';

import { useEffect, useState } from 'react';

export function Clock() {
    const [time, setTime] = useState<Date | null>(null);

    useEffect(() => {
        setTime(new Date());
        const id = setInterval(() => setTime(new Date()), 1000);
        return () => clearInterval(id);
    }, []);

    if (!time) return null;

    const dateStr = time.toLocaleDateString('zh-CN', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        weekday: 'long',
    });

    const timeStr = time.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
    });

    return (
        <div className="flex flex-col items-end gap-0.5 font-mono select-none">
            <span className="text-2xl font-semibold tracking-widest gradient-text tabular-nums">
                {timeStr}
            </span>
            <span className="text-xs text-[var(--muted)] tracking-wider">
                {dateStr}
            </span>
        </div>
    );
}
