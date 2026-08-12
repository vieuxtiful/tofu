import pathlib
import re

root = pathlib.Path("frontend/src")
imports = set()
for f in root.rglob("*.tsx"):
    text = f.read_text(encoding="utf-8")
    for m in re.finditer(r'from ["\'](\./?[^"\']+)["\']', text):
        imports.add((str(f.relative_to(root)), m.group(1)))

for src, imp in sorted(imports):
    base = imp.lstrip("./")
    candidates = [
        root / (base + ".tsx"),
        root / (base + ".ts"),
        root / (base + "/index.tsx"),
        root / (base + "/index.ts"),
    ]
    if not any(c.exists() for c in candidates):
        print(f"MISSING: {imp}  (imported from {src})")
