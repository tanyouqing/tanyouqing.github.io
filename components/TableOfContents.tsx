'use client';

import { useEffect, useRef, useState } from 'react';
import type { TocItem } from '@/lib/extract-toc';

interface TableOfContentsProps {
    items: TocItem[];
}

export function TableOfContents({ items }: TableOfContentsProps) {
    const [activeId, setActiveId] = useState<string>('');
    const observerRef = useRef<IntersectionObserver | null>(null);

    useEffect(() => {
        if (items.length === 0) return;

        // Keep track of all currently-visible heading ids
        const visibleIds = new Set<string>();

        const pickActive = () => {
            // Among all visible headings, prefer the one that appears earliest in the TOC list
            for (const item of items) {
                if (visibleIds.has(item.id)) {
                    setActiveId(item.id);
                    return;
                }
            }
        };

        observerRef.current = new IntersectionObserver(
            (entries) => {
                for (const entry of entries) {
                    if (entry.isIntersecting) {
                        visibleIds.add(entry.target.id);
                    } else {
                        visibleIds.delete(entry.target.id);
                    }
                }
                pickActive();
            },
            {
                // Fire when a heading enters or exits the upper 40% of the viewport
                rootMargin: '-80px 0px -55% 0px',
                threshold: 0,
            }
        );

        const observer = observerRef.current;
        for (const item of items) {
            const el = document.getElementById(item.id);
            if (el) observer.observe(el);
        }

        return () => observer.disconnect();
    }, [items]);

    if (items.length === 0) return null;

    const handleClick = (e: React.MouseEvent<HTMLAnchorElement>, id: string) => {
        e.preventDefault();
        const el = document.getElementById(id);
        if (el) {
            // Smooth scroll with a small top offset for the sticky navbar
            const offset = 88;
            const top = el.getBoundingClientRect().top + window.scrollY - offset;
            window.scrollTo({ top, behavior: 'smooth' });
            setActiveId(id);
            window.history.replaceState(null, '', `#${id}`);
        }
    };

    return (
        <nav aria-label="目录" className="toc-sidebar">
            <p className="toc-title">目录</p>
            <ul className="toc-list">
                {items.map((item) => {
                    const isActive = activeId === item.id;
                    // Indent h2 by 12px, h3 by 24px
                    const indent = (item.level - 1) * 12;
                    return (
                        <li key={item.id} style={{ paddingLeft: `${indent}px` }}>
                            <a
                                href={`#${item.id}`}
                                onClick={(e) => handleClick(e, item.id)}
                                className={`toc-link${isActive ? ' toc-link-active' : ''}`}
                                title={item.text}
                            >
                                {item.text}
                            </a>
                        </li>
                    );
                })}
            </ul>
        </nav>
    );
}
