import re
src = open('main.tex', encoding='utf-8').read()
for m in re.finditer(r'\\citet\{([^}]+)\}', src):
    line_no = src[:m.start()].count('\n') + 1
    # show the surrounding 80 chars
    a = max(0, m.start() - 30)
    b = min(len(src), m.end() + 30)
    print(f'L{line_no}: ...{src[a:b]}...')
