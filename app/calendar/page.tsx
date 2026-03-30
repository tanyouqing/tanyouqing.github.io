import { getAllContent } from '@/lib/mdx';
import { Calendar as CalendarIcon, CheckCircle2, CircleDashed, Clock } from 'lucide-react';

export const metadata = {
    title: 'Working Calendar | Cao Jiahao',
    description: 'Track my schedule and working calendar.',
};

export default function CalendarPage() {
    // 1. Fetch all contents to aggregate upload history
    const allItems = [
        ...getAllContent('articles').map(i => ({ ...i, type: 'Article' })),
        ...getAllContent('projects').map(i => ({ ...i, type: 'Project' })),
        ...getAllContent('research').map(i => ({ ...i, type: 'Research' })),
    ];

    // 2. Generate 3 exact months of data (Two months ago, Last month, This month)
    const today = new Date();
    const currentYear = today.getFullYear();
    const currentMonth = today.getMonth();
    // Use string comparison for isFuture to avoid time-of-day and timezone issues
    const todayStr = `${currentYear}-${String(currentMonth + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;

    const monthsData: {
        title: string;
        cells: { empty: boolean; dateStr?: string; items?: any[]; isFuture?: boolean }[];
    }[] = [];

    for (let i = 2; i >= 0; i--) {
        let y = currentYear;
        let m = currentMonth - i;
        if (m < 0) {
            m += 12;
            y -= 1;
        }

        const firstDay = new Date(y, m, 1);
        const lastDay = new Date(y, m + 1, 0);
        const startDayOfWeek = firstDay.getDay(); // 0 = Sunday, 6 = Saturday
        const daysInMonth = lastDay.getDate();

        const cells: any[] = [];
        // Pad the start of the month with empty cells so the 1st falls on the correct weekday
        for (let pad = 0; pad < startDayOfWeek; pad++) {
            cells.push({ empty: true });
        }

        // Generate actual days — compare date strings so today itself is NOT isFuture
        for (let d = 1; d <= daysInMonth; d++) {
            const dateStr = `${y}-${String(m + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
            const itemsForDay = allItems.filter(item => item.date === dateStr);
            cells.push({
                empty: false,
                dateStr,
                items: itemsForDay,
                isFuture: dateStr > todayStr   // pure string compare: '2026-03-31' > '2026-03-30'
            });
        }

        monthsData.push({
            title: `${y}年 ${m + 1}月`,
            cells
        });
    }

    // Determine the max items per day to scale the brightness (optional, but a nice touch)
    const maxItems = Math.max(...monthsData.flatMap(m => m.cells.filter(c => !c.empty).map(c => c.items?.length || 0)), 1);

    return (
        <div className="max-w-6xl mx-auto px-6 py-12">
            <header className="mb-12">
                <h1 className="text-3xl md:text-4xl font-bold text-[var(--fg)] inline-flex items-center gap-3">
                    <CalendarIcon size={32} className="text-cyan-500 dark:text-[#c9a55a]" />
                    工作日历
                </h1>
                <p className="text-[var(--muted)] mt-4 max-w-2xl leading-relaxed">
                    用于追踪个人的计划与上传记录，展示近期的各项工作及日程。
                </p>
            </header>

            <div className="flex flex-col gap-12">
                {/* ── Section 1: Recent Schedule ── */}
                <section>
                    <h2 className="text-xl font-semibold text-[var(--fg)] mb-6 flex items-center gap-2">
                        <Clock size={20} className="text-orange-400 dark:text-[#8a7340]" />
                        近期日程
                    </h2>

                    <div className="grid md:grid-cols-2 gap-6">
                        {/* Completed Tasks */}
                        <div className="rounded-2xl border border-[var(--border)] bg-[var(--card)] p-6 shadow-sm">
                            <h3 className="text-[var(--fg)] font-medium text-lg flex items-center gap-2 mb-4 border-b border-[var(--border)] pb-3">
                                <CheckCircle2 size={18} className="text-green-500" />
                                近期完成任务
                            </h3>
                            <ul className="space-y-4">
                                <li className="flex items-start gap-3 text-[var(--muted)]">
                                    <div className="mt-1 w-1.5 h-1.5 rounded-full bg-green-500 shrink-0" />
                                    <div>
                                        <p className="text-[var(--fg)] font-medium text-sm">部署小龙虾</p>
                                        <p className="text-xs opacity-75 mt-0.5">租云服务器并部署OpenClaw+OpenViking服务</p>
                                    </div>
                                </li>
                                <li className="flex items-start gap-3 text-[var(--muted)]">
                                    <div className="mt-1 w-1.5 h-1.5 rounded-full bg-green-500 shrink-0" />
                                    <div>
                                        <p className="text-[var(--fg)] font-medium text-sm">强化学习的数学原理学习和笔记记录</p>
                                        <p className="text-xs opacity-75 mt-0.5">学习赵世钰老师《强化学习的数学原理》课程</p>
                                    </div>
                                </li>
                            </ul>
                        </div>

                        {/* Planned Tasks */}
                        <div className="rounded-2xl border border-[var(--border)] bg-[var(--card)] p-6 shadow-sm">
                            <h3 className="text-[var(--fg)] font-medium text-lg flex items-center gap-2 mb-4 border-b border-[var(--border)] pb-3">
                                <CircleDashed size={18} className="text-orange-400 dark:text-[#c9a55a]" />
                                计划完成任务
                            </h3>
                            <ul className="space-y-4">

                                <li className="flex items-start gap-3 text-[var(--muted)]">
                                    <div className="mt-1 w-1.5 h-1.5 rounded-full bg-orange-400 dark:bg-[#c9a55a] shrink-0" />
                                    <div>
                                        <p className="text-[var(--fg)] font-medium text-sm">雅思备考</p>
                                    </div>
                                </li>
                                <li className="flex items-start gap-3 text-[var(--muted)]">
                                    <div className="mt-1 w-1.5 h-1.5 rounded-full bg-orange-400 dark:bg-[#c9a55a] shrink-0" />
                                    <div>
                                        <p className="text-[var(--fg)] font-medium text-sm">Openviking项目学习</p>
                                        <p className="text-xs opacity-75 mt-0.5">学习Openviking记忆系统实现并探索改进方向</p>
                                    </div>
                                </li>
                                <li className="flex items-start gap-3 text-[var(--muted)]">
                                    <div className="mt-1 w-1.5 h-1.5 rounded-full bg-orange-400 dark:bg-[#c9a55a] shrink-0" />
                                    <div>
                                        <p className="text-[var(--fg)] font-medium text-sm">完善毕设实验</p>
                                        <p className="text-xs opacity-75 mt-0.5">融合CLIP和U-Net的后门攻击实验，并探索防御策略</p>
                                    </div>
                                </li>
                            </ul>
                        </div>
                    </div>
                </section>

                {/* ── Section 2: Upload History (GitHub Contribution Graph) ── */}
                <section>
                    <h2 className="text-xl font-semibold text-[var(--fg)] mb-6 flex items-center gap-2">
                        <CheckCircle2 size={20} className="text-cyan-500 dark:text-[#d4b978]" />
                        近三个月上传记录
                    </h2>

                    <div className="rounded-2xl border border-[var(--border)] bg-[var(--card)] p-6 md:p-8 shadow-sm flex flex-col">
                        <div className="flex justify-end gap-2 mb-8 text-xs text-[var(--muted)] items-center">
                            <span>少</span>
                            {/* Deeper empty gray for light mode */}
                            <div className="w-4 h-4 rounded-[4px] bg-slate-200 dark:bg-[#2a2825]" />
                            <div className="w-4 h-4 rounded-[4px] bg-emerald-400 dark:bg-emerald-800" />
                            <div className="w-4 h-4 rounded-[4px] bg-emerald-500 dark:bg-emerald-600" />
                            <span>多</span>
                        </div>

                        {/* 3 Months layout */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 w-full">
                            {monthsData.map((month, mIdx) => (
                                <div key={mIdx} className="w-full flex flex-col items-center">
                                    <h3 className="text-[var(--fg)] font-medium mb-4 tracking-wider">{month.title}</h3>

                                    {/* Weekday headers aligned on top */}
                                    <div className="grid grid-cols-7 gap-1.5 md:gap-2 mb-2 w-full max-w-[280px]">
                                        {['日', '一', '二', '三', '四', '五', '六'].map(dayName => (
                                            <div key={dayName} className="text-center text-[0.7rem] text-[var(--muted)]">{dayName}</div>
                                        ))}
                                    </div>

                                    {/* Month cells grid */}
                                    <div className="grid grid-cols-7 gap-1.5 md:gap-2 w-full max-w-[280px]">
                                        {month.cells.map((cell, cIdx) => {
                                            if (cell.empty) {
                                                return <div key={cIdx} className="w-full aspect-square rounded-[4px] opacity-0" />;
                                            }

                                            // Exclude future days from highlighting
                                            let bgClass = "bg-slate-200 dark:bg-[#2a2825]";
                                            const count = cell.items?.length || 0;

                                            if (!cell.isFuture && count > 0) {
                                                if (count === 1) bgClass = "bg-emerald-400 dark:bg-emerald-800 shadow-sm";
                                                if (count >= 2) bgClass = "bg-emerald-500 dark:bg-emerald-600 shadow-[0_0_10px_rgba(16,185,129,0.4)]";
                                            }

                                            return (
                                                <div
                                                    key={cIdx}
                                                    className={`group relative w-full aspect-square rounded-[4px] cursor-help transition-colors hover:ring-2 hover:ring-emerald-400/50 hover:ring-offset-1 dark:hover:ring-offset-[#1a1917] ${bgClass}`}
                                                >
                                                    {/* Custom Tooltip */}
                                                    <div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-3 w-max max-w-[220px] z-50
                                                        opacity-0 group-hover:opacity-100 transition-opacity duration-200
                                                        bg-slate-800 dark:bg-white text-white dark:text-slate-900 
                                                        text-xs rounded-lg px-3 py-2 shadow-xl">
                                                        <div className="font-semibold mb-1">{cell.dateStr}</div>
                                                        {count > 0 ? (
                                                            <ul className="list-disc pl-3 text-[0.7rem] break-words whitespace-normal leading-relaxed opacity-90">
                                                                {cell.items!.map((item, idx) => (
                                                                    <li key={idx}>[{item.type}] {item.title}</li>
                                                                ))}
                                                            </ul>
                                                        ) : (
                                                            <p className="opacity-75">{cell.isFuture ? "未到来的日期" : "没有任何更新"}</p>
                                                        )}
                                                        {/* Tooltip Arrow */}
                                                        <div className="absolute top-full left-1/2 -translate-x-1/2 border-[5px] border-transparent border-t-slate-800 dark:border-t-white" />
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>
            </div>
        </div>
    );
}
