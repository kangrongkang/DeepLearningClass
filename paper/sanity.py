"""Static sanity check for the paper."""
import re

src = open('main.tex', encoding='utf-8').read()
bib = open('references.bib', encoding='utf-8').read()

print('Sections:', re.findall(r'\\section\*?\{([^}]+)\}', src))
print('Subsections:', len(re.findall(r'\\subsection\{([^}]+)\}', src)))
print('Tables:', len(re.findall(r'\\begin\{table\*?\}', src)))
print('Figures:', len(re.findall(r'\\begin\{figure\*?\}', src)))

cite_calls = re.findall(r'\\cite[ptn]?\{([^}]+)\}', src)
used_keys = set()
for c in cite_calls:
    for k in c.split(','):
        used_keys.add(k.strip())
defined_keys = set(re.findall(r'@\w+\{([^,]+),', bib))

print('Cite keys used :', len(used_keys))
print('Bib keys defined:', len(defined_keys))
print('Undefined keys :', sorted(used_keys - defined_keys))
print('Unused bib keys:', sorted(defined_keys - used_keys))

print('Brace balance  :', src.count('{'), '/', src.count('}'),
      ('OK' if src.count('{') == src.count('}') else 'MISMATCH'))

# Verify every figure file referenced exists
import os
fig_refs = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', src)
missing = [f for f in fig_refs if not os.path.exists(os.path.join('figures', f))]
print('Figure refs    :', len(fig_refs))
print('Missing figures:', missing)
