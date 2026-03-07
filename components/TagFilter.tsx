'use client';

interface TagFilterProps {
    tags: string[];
    selected: string | null;
    onSelect: (tag: string | null) => void;
}

export function TagFilter({ tags, selected, onSelect }: TagFilterProps) {
    return (
        <div className="flex flex-wrap gap-2">
            <button
                onClick={() => onSelect(null)}
                className={`tag-badge ${selected === null ? 'active' : ''}`}
            >
                All
            </button>
            {tags.map(tag => (
                <button
                    key={tag}
                    onClick={() => onSelect(selected === tag ? null : tag)}
                    className={`tag-badge ${selected === tag ? 'active' : ''}`}
                >
                    {tag}
                </button>
            ))}
        </div>
    );
}
