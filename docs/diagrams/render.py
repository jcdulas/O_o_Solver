"""Render the documentation's editable diagram specifications as standalone SVGs."""
from pathlib import Path
from html import escape
import argparse
import json
import re
import textwrap

COLORS = {
    'blue': ('#edf3ff', '#4366b3'),
    'teal': ('#e9f6f2', '#237d69'),
    'purple': ('#f3edfc', '#7951a8'),
    'amber': ('#fff5df', '#a56b18'),
    'gray': ('#f0f3f7', '#64748b'),
}


def render(spec):
    width, height = spec['size']
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '<title id="title">' + escape(spec['title']) + '</title>',
        '<desc id="description">' + escape(spec['description']) + '</desc>',
        '<rect width="100%" height="100%" rx="18" fill="#f8fafc"/>',
        '<g font-family="sans-serif" fill="#1c2e48">',
        f'<text x="36" y="40" font-size="13" font-weight="700" letter-spacing="1.5" fill="#52647e">{escape(spec["category"].upper())}</text>',
        f'<text x="36" y="77" font-size="28" font-weight="700">{escape(spec["title"])}</text>',
    ]
    for group in spec.get('groups', []):
        x, y, w, h = group['box']
        _, stroke = COLORS[group.get('color', 'gray')]
        pieces.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="#ffffff" stroke="{stroke}" stroke-width="1.5" stroke-dasharray="7 5"/>')
        pieces.append(f'<text x="{x+20}" y="{y+30}" font-size="17" font-weight="700" fill="{stroke}">{escape(group["title"])}</text>')
        if group.get('subtitle'):
            pieces.append(f'<text x="{x+20}" y="{y+54}" font-size="15" fill="#52647e">{escape(group["subtitle"])}</text>')
    for arrow in spec.get('arrows', []):
        dashed = ' stroke-dasharray="7 5"' if arrow.get('dashed') else ''
        pieces.append(f'<path d="{escape(arrow["path"])}" fill="none" stroke="#50627b" stroke-width="2.4" stroke-linejoin="round"{dashed}/>')
        x = y = 0.0
        previous = (x, y)
        for command, values in re.findall(r'([MHV])([^MHV]+)', arrow['path']):
            numbers = [float(value) for value in values.split()]
            previous = (x, y)
            if command == 'M':
                x, y = numbers
            elif command == 'H':
                x = numbers[0]
            else:
                y = numbers[0]
        dx, dy = x-previous[0], y-previous[1]
        length = (dx*dx+dy*dy)**0.5
        ux, uy = dx/length, dy/length
        points = [(x, y), (x-12*ux+5*uy, y-12*uy-5*ux), (x-12*ux-5*uy, y-12*uy+5*ux)]
        pieces.append('<polygon points="'+' '.join(f'{a:g},{b:g}' for a, b in points)+'" fill="#50627b"/>')
        if arrow.get('label'):
            x, y = arrow['label_at']
            label = arrow['label']
            w = len(label) * 8 + 16
            pieces.append(f'<rect x="{x-w/2}" y="{y-17}" width="{w}" height="24" rx="5" fill="#f8fafc"/>')
            pieces.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="15" fill="#43546d">{escape(label)}</text>')
    for node in spec['nodes']:
        x, y, w, h = node['box']
        fill, stroke = COLORS[node.get('color', 'blue')]
        pieces.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        pieces.append(f'<rect x="{x+1}" y="{y+14}" width="4" height="{h-28}" rx="2" fill="{stroke}"/>')
        baseline = y + 25
        if node.get('tag'):
            pieces.append(f'<text x="{x+20}" y="{baseline}" font-size="12" font-weight="700" letter-spacing="1.2" fill="{stroke}">{escape(node["tag"].upper())}</text>')
            baseline += 28
        pieces.append(f'<text x="{x+20}" y="{baseline}" font-size="{node.get("title_size", 21)}" font-weight="700">{escape(node["title"])}</text>')
        baseline += 29
        lines = []
        for line in node.get('body', []):
            lines.extend(textwrap.wrap(line, max(14, int((w-40)/10)), break_long_words=False))
        if baseline + max(0, len(lines)-1)*24 > y+h-12:
            raise ValueError(f'{spec["id"]}: text exceeds node {node["title"]}')
        for line in lines:
            pieces.append(f'<text x="{x+20}" y="{baseline}" font-size="20" fill="#43546d">{escape(line)}</text>')
            baseline += 24
    for note in spec.get('notes', []):
        x, y = note['at']
        for line in note['text']:
            pieces.append(f'<text x="{x}" y="{y}" font-size="16" fill="#43546d">{escape(line)}</text>')
            y += 24
    pieces.extend(['</g>', '</svg>', ''])
    return '\n'.join(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify the committed SVGs without changing them')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    figures = json.loads((directory/'figures.json').read_text(encoding='utf-8'))
    output = directory.parent/'images'
    output.mkdir(exist_ok=True)
    for spec in figures:
        if spec['id'].endswith('-layers'):
            modules = [node['title'] for node in spec['nodes']]
            for module, dependency in zip(modules, modules[1:]):
                build = (directory.parents[1]/'src'/module/'CMakeLists.txt').read_text(encoding='utf-8')
                links = ' '.join(re.findall(r'target_link_libraries\((.*?)\)', build, re.S))
                if not re.search(r'(?:oor_|oos::)'+re.escape(dependency)+r'\b', links):
                    raise SystemExit(f'{spec["id"]}: review the {module} to {dependency} dependency')
        target = output/(spec['id']+'.svg')
        content = render(spec)
        if args.check:
            if not target.exists() or target.read_text(encoding='utf-8') != content:
                raise SystemExit(f'{target.name} needs regeneration: run python docs/diagrams/render.py')
        else:
            target.write_text(content, encoding='utf-8', newline='\n')
    print(f'{len(figures)} architecture SVGs '+('verified.' if args.check else 'generated.'))


if __name__ == '__main__':
    main()
