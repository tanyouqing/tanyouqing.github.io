'use client';

import { useEffect, useState } from 'react';

interface TypewriterQuoteProps {
    text: string;
    speed?: number;
    className?: string;
}

export function TypewriterQuote({ text, speed = 80, className = '' }: TypewriterQuoteProps) {
    const [displayed, setDisplayed] = useState('');
    const [cursorVisible, setCursorVisible] = useState(true);
    const [done, setDone] = useState(false);

    useEffect(() => {
        let i = 0;
        const timer = setInterval(() => {
            if (i < text.length) {
                setDisplayed(text.slice(0, i + 1));
                i++;
            } else {
                setDone(true);
                clearInterval(timer);
            }
        }, speed);
        return () => clearInterval(timer);
    }, [text, speed]);

    useEffect(() => {
        if (done) {
            const blink = setInterval(() => setCursorVisible(v => !v), 530);
            return () => clearInterval(blink);
        }
    }, [done]);

    return (
        <span className={className}>
            {displayed}
            <span
                className={`inline-block h-[1.1em] align-middle ml-0.5 bg-current
          transition-all duration-150 ${done ? 'opacity-0 w-0' : `w-0.5 ${cursorVisible ? 'opacity-100' : 'opacity-0'}`}`}
            />
        </span>
    );
}
