/** @type {import('tailwindcss').Config} */
module.exports = {
    darkMode: 'class',
    content: [
        './app/**/*.{js,ts,jsx,tsx,mdx}',
        './components/**/*.{js,ts,jsx,tsx,mdx}',
        './content/**/*.{js,ts,jsx,tsx,mdx}',
    ],
    theme: {
        extend: {
            colors: {
                cyan: {
                    400: '#22d3ee',
                    500: '#06b6d4',
                    600: '#0891b2',
                },
                orange: {
                    400: '#fb923c',
                    500: '#f97316',
                },
            },
            fontFamily: {
                sans: ['Inter', 'ui-sans-serif', 'system-ui'],
                serif: ['"Noto Serif SC"', 'ui-serif', 'Georgia'],
                mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular'],
            },
            animation: {
                aurora: 'aurora 15s ease infinite',
                'fade-in': 'fadeIn 0.8s ease forwards',
                'slide-up': 'slideUp 0.6s ease forwards',
            },
            keyframes: {
                aurora: {
                    '0%, 100%': { backgroundPosition: '0% 50%' },
                    '50%': { backgroundPosition: '100% 50%' },
                },
                fadeIn: {
                    from: { opacity: '0' },
                    to: { opacity: '1' },
                },
                slideUp: {
                    from: { opacity: '0', transform: 'translateY(20px)' },
                    to: { opacity: '1', transform: 'translateY(0)' },
                },
            },
            backgroundSize: {
                '400': '400% 400%',
            },
        },
    },
    plugins: [],
};
