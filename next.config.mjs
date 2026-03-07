import createMDX from '@next/mdx';
import remarkGfm from 'remark-gfm';

const withMDX = createMDX({
    options: {
        remarkPlugins: [remarkGfm],
        rehypePlugins: [],
    },
});

/** @type {import('next').NextConfig} */
const nextConfig = {
    pageExtensions: ['js', 'jsx', 'mdx', 'ts', 'tsx'],

    // ── GitHub Pages 静态导出配置 ──
    output: 'export',          // 生成纯静态 HTML/CSS/JS
    trailingSlash: true,       // 确保每个路径都有 /，GitHub Pages 需要
    images: {
        unoptimized: true,       // 静态导出不支持 Next.js 图片优化
    },

    // ⚠️ 如果你的仓库名不是 <username>.github.io，需要填写仓库名
    // 例如仓库名叫 "my-homepage"，则：
    // basePath: '/my-homepage',
    // assetPrefix: '/my-homepage/',
};

export default withMDX(nextConfig);
