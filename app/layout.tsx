import type { Metadata } from 'next';
import './globals.css';
import { Providers } from './providers';
import { Navbar } from '@/components/Navbar';
import { Footer } from '@/components/Footer';

export const metadata: Metadata = {
    title: 'Cao Jiahao | Personal Academic Homepage',
    description: '西安交通大学软件工程专业学生，即将攻读硕士研究生。Personal academic homepage of Cao Jiahao.',
    keywords: ['Cao Jiahao', 'XJTU', '西安交通大学', 'Software Engineering', 'Academic'],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="zh-CN" suppressHydrationWarning>
            <head>
                <script
                    dangerouslySetInnerHTML={{
                        __html: `
                            try {
                                if (!localStorage.getItem('theme')) {
                                    var hour = new Date().getHours();
                                    var theme = (hour >= 8 && hour < 21) ? 'light' : 'dark';
                                    document.documentElement.classList.add(theme);
                                    localStorage.setItem('theme', theme);
                                }
                            } catch (e) {}
                        `,
                    }}
                />
                <link rel="preconnect" href="https://fonts.googleapis.com" />
                <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
                <link
                    href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Serif+SC:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap"
                    rel="stylesheet"
                />
            </head>
            <body>
                <Providers>
                    <Navbar />
                    <main className="min-h-screen pt-24">
                        {children}
                    </main>
                    <Footer />
                </Providers>
            </body>
        </html>
    );
}
