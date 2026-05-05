"""Find dangling \cref / \ref / \Cref references in main.tex."""
import re
src = open('main.tex', encoding='utf-8').read()
defined = set(re.findall(r'\\label\{([^}]+)\}', src))
used = set()
for m in re.finditer(r'\\(?:cref|Cref|ref)\*?\{([^}]+)\}', src):
    for k in m.group(1).split(','):
        used.add(k.strip())
print("Defined labels:", sorted(defined))
print("Used labels   :", sorted(used))
print("Dangling refs :", sorted(used - defined))
print("Unused labels :", sorted(defined - used))
