import os, re, sys, shutil, glob, importlib.util

ROOT = os.path.abspath(os.getcwd())
print(f"[cwd] {ROOT}")

# 1) Mostrar qué app_streamlit.py se va a ejecutar si llamas streamlit run app_streamlit.py desde aquí
APP = os.path.join(ROOT, "app_streamlit.py")
print(f"[app_streamlit] exists={os.path.exists(APP)}  path={APP}")

# 2) Buscar TODOS los imports relativos "from xxx import ..."
pat_rel = re.compile(r"from\s+\.\s*(\w+)\s+import", re.I)
pat_pkg = re.compile(r"from\s+qvm_trend\.", re.I)

bad = []
for path in glob.glob("**/*.py", recursive=True):
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        s = fh.read()
    if pat_rel.search(s) or pat_pkg.search(s):
        bad.append(path)

print(f"[relative-or-qvm_trend-imports] {len(bad)} archivo(s)")
for b in bad: print(" -", b)

# 3) Parchear TODOS los .py (con backup .bak)
changed = []
for path in glob.glob("**/*.py", recursive=True):
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        s = fh.read()
    s2 = s
    s2 = re.sub(r"from\s+\.\s*(\w+)\s+import", r"from \1 import", s2)
    s2 = re.sub(r"from\s+qvm_trend\.", "from ", s2)
    if s2 != s:
        shutil.copy2(path, path + ".bak")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(s2)
        changed.append(path)

print(f"[patched] {len(changed)} archivo(s) modificados")
for c in changed: print(" *", c)

# 4) Verificación final: listar cualquier import relativo restante
left = []
for path in glob.glob("**/*.py", recursive=True):
    s = open(path, "r", encoding="utf-8", errors="ignore").read()
    if pat_rel.search(s):
        left.append(path)
print(f"[remaining-relative-imports] {len(left)}")
for l in left: print(" !", l)

# 5) Intentar importar módulos clave desde ESTA carpeta (simula import-time)
sys.path.insert(0, ROOT)
modules = ["scoring","factors","factors_growth_aware","fundamentals","data_io","pipeline","backtests","stats","mc"]
print("\n[import-check]")
for m in modules:
    try:
        spec = importlib.util.find_spec(m)
        if spec is None:
            print(f" - {m}: NOT FOUND in sys.path[0]={sys.path[0]}")
        else:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f" - {m}: OK")
    except Exception as e:
        print(f" - {m}: ERROR -> {e.__class__.__name__}: {e}")

print("\n[hint] Para ejecutar la app contra ESTE árbol:\n"
      f'  set FMP_API_KEY=TU_API_KEY  (o $env:FMP_API_KEY en PowerShell)\n'
      f'  streamlit run "{APP}"')
