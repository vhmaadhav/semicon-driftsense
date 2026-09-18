#!/usr/bin/env python3
"""graphify — build and query a local knowledge graph of this repository.

Writes ``graphify-out/`` with an interactive HTML graph, ``graph.json``,
``GRAPH_REPORT.md`` and a ``wiki/`` for broad navigation. The directory is
git-ignored: it holds generated views, caches and local query memory.

Commands (run from the repository root, or pass the root as the last arg):

    python3 scripts/graphify.py build .            full build (or just: build)
    python3 scripts/graphify.py update .           incremental, AST-only re-scan
    python3 scripts/graphify.py query "question"   scoped subgraph for a question
    python3 scripts/graphify.py path "A" "B" [--undirected]   relationship path
    python3 scripts/graphify.py explain "Concept"  focused concept subgraph
    python3 scripts/graphify.py images desc.json   merge image descriptions

Pure stdlib. Deterministic: same tree, same graph.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import urllib.parse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------

SKIP_DIRS = {
    ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".obsidian",
    ".codex", "__pycache__", "graphify-out", "venv", "venv313", "venv-train",
    "node_modules", "data", "output", "sample", "logs", ".agents/ab.log",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
DOC_EXTS = {".md", ".rst", ".txt"}
CODE_EXTS = {".py"}
CONFIG_EXTS = {".toml", ".yaml", ".yml", ".cfg", ".ini", ".json", ".txt"}
MAX_DOCSYM_EDGES_PER_FILE = 40
MAX_CALL_EDGES_PER_FUNC = 30


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scan_files(root: Path) -> dict[str, dict]:
    """Return {relpath: {kind, sha, size}} for every file we graph."""
    out: dict[str, dict] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        parts = rel.split("/")
        if any(part in SKIP_DIRS for part in parts[:-1]) or rel.split("/")[-1] in SKIP_DIRS:
            continue
        if rel.startswith("weights/") or rel.endswith(".pt"):
            continue
        ext = p.suffix.lower()
        if ext in CODE_EXTS:
            kind = "code"
        elif ext in DOC_EXTS:
            kind = "doc"
        elif ext in IMAGE_EXTS:
            kind = "image"
        elif ext == ".pdf":
            kind = "doc"
        elif ext in CONFIG_EXTS or p.name in {"Dockerfile", "Makefile", ".gitignore"}:
            kind = "config"
        else:
            continue
        try:
            data = p.read_bytes()
        except OSError:
            continue
        out[rel] = {"kind": kind, "sha": sha256(data), "size": len(data)}
    return out


# --------------------------------------------------------------------------
# python AST extraction
# --------------------------------------------------------------------------

def module_name(rel: str) -> str:
    path = rel[:-3] if rel.endswith(".py") else rel
    if path.endswith("/__init__"):
        path = path[: -len("/__init__")]
    return path.replace("/", ".")


def doc1(node, cap: int = 280) -> str:
    """First paragraph of an AST node's docstring, flattened. '' when absent."""
    try:
        d = ast.get_docstring(node) or ""
    except Exception:
        return ""
    if not d:
        return ""
    return " ".join(d.strip().split("\n\n")[0].split())[:cap]


def func_sig(node) -> str:
    try:
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"def {node.name}({ast.unparse(node.args)}){ret}"
    except Exception:
        return f"def {node.name}(...)"


def class_sig(node) -> str:
    try:
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    except Exception:
        return f"class {node.name}"


def span(node) -> int:
    try:
        return node.end_lineno - node.lineno + 1
    except Exception:
        return 0


def md_summary(text: str, cap: int = 220) -> str:
    """First prose line of a markdown file (front matter and headings skipped)."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":  # YAML front matter
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                lines = lines[i + 1:]
                break
    for line in lines:
        s = line.strip()
        if not s or s.startswith(("#", "![", "|", "```")):
            continue
        return " ".join(s.split())[:cap]
    return ""


def resolve_module(mod: str, file_set: set[str]) -> str | None:
    """Map a dotted module name to a project file, or None."""
    if not mod:
        return None
    for cand in (mod.replace(".", "/") + ".py", mod.replace(".", "/") + "/__init__.py"):
        if cand in file_set:
            return cand
    return None


def parse_python(rel: str, tree: ast.AST, file_set: set[str], loc: int = 0):
    """One file -> (nodes, edges, imports, defs).  defs: name -> node_id."""
    nodes, edges = [], []
    fnode = f"py:{rel}"
    nodes.append({"id": fnode, "kind": "file", "label": rel, "file": rel, "loc": loc})
    imports: dict[str, tuple[str, str | None]] = {}  # alias -> (project file, orig symbol | None)
    defs: dict[str, str] = {}      # bare name -> node id (this file)
    local_def_names: set[str] = set()

    # pass 1: collect top-level defs and imports
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nid = f"{fnode}::{node.name}"
            defs[node.name] = nid
            local_def_names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            nid = f"{fnode}::{node.name}"
            defs[node.name] = nid
            local_def_names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                target = resolve_module(alias.name, file_set)
                if target:
                    imports[alias.asname or alias.name.split(".")[0]] = (target, None)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level > 0:  # relative: driftsense/foo.py level=1 -> driftsense/
                base = "/".join(rel.split("/")[:-1][: len(rel.split("/")[:-1]) - (node.level - 1)])
                mod = "/".join(x for x in [base.replace("/", "."), mod] if x)
            if node.module and node.names and node.names[0].name == "*":
                target = resolve_module(mod, file_set)
                if target:
                    imports[f"*{mod}"] = (target, None)
                continue
            target = resolve_module(mod, file_set)
            if target:
                for alias in node.names:
                    # `from pkg import submodule` may name a module, not a symbol
                    sub = resolve_module(f"{mod}.{alias.name}", file_set) if mod else None
                    imports[alias.asname or alias.name] = (sub or target,
                                                           None if sub else alias.name)

    # pass 2: members, inheritance, import edges
    file_defs_list: list[dict] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes.append({
                "id": f"{fnode}::{node.name}", "kind": "function",
                "label": node.name, "file": rel, "line": node.lineno,
                "sig": func_sig(node), "lines": span(node), "desc": doc1(node),
            })
            file_defs_list.append({"id": f"{fnode}::{node.name}", "label": node.name})
            edges.append({"src": fnode, "dst": f"{fnode}::{node.name}", "kind": "contains"})
        elif isinstance(node, ast.ClassDef):
            cid = f"{fnode}::{node.name}"
            nodes.append({
                "id": cid, "kind": "class", "label": node.name,
                "file": rel, "line": node.lineno,
                "sig": class_sig(node), "lines": span(node), "desc": doc1(node),
            })
            file_defs_list.append({"id": cid, "label": node.name})
            edges.append({"src": fnode, "dst": cid, "kind": "contains"})
            for base in node.bases:
                bname = ast.unparse(base).split(".")[-1] if hasattr(ast, "unparse") else ""
                if bname and not bname.startswith("_"):
                    edges.append({"src": cid, "dst": f"base:{bname}", "kind": "inherits"})
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mid = f"{cid}.{sub.name}"
                    nodes.append({
                        "id": mid, "kind": "method", "label": f"{node.name}.{sub.name}",
                        "file": rel, "line": sub.lineno,
                        "sig": func_sig(sub), "lines": span(sub), "desc": doc1(sub),
                    })
                    edges.append({"src": cid, "dst": mid, "kind": "contains"})
                    defs.setdefault(f"{node.name}.{sub.name}", mid)
    nodes[0]["desc"] = doc1(tree)
    nodes[0]["defs"] = file_defs_list[:20]
    for alias, (target, _orig) in imports.items():
        edges.append({"src": fnode, "dst": f"py:{target}", "kind": "imports"})
    return nodes, edges, imports, defs, local_def_names


def calls_in_func(fnode_id: str, fn_node: ast.AST,
                  imports: dict[str, tuple[str, str | None]],
                  file_defs: dict[str, str], global_defs: dict[str, list[str]]) -> list[dict]:
    edges: list[dict] = []
    seen: set[str] = set()

    def add(dst: str):
        if dst and dst != fnode_id and dst not in seen and len(edges) < MAX_CALL_EDGES_PER_FUNC:
            seen.add(dst)
            edges.append({"src": fnode_id, "dst": dst, "kind": "calls"})

    def imported_symbol(alias: str, attr: str | None) -> str | None:
        """Resolve a call through an import alias to a project symbol node."""
        entry = imports.get(alias)
        if not entry:
            return None
        target_file, orig = entry
        symbol = attr or orig
        if symbol:
            for cand_id in global_defs.get(symbol, []):
                if cand_id.startswith(f"py:{target_file}"):
                    return cand_id
        return f"py:{target_file}"

    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in file_defs:
                add(file_defs[name])
            elif name in global_defs:
                add(global_defs[name][0])
            else:
                dst = imported_symbol(name, None)
                if dst:
                    add(dst)
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name):
                dst = imported_symbol(func.value.id, func.attr)
                if dst:
                    add(dst)
    return edges


# --------------------------------------------------------------------------
# markdown extraction
# --------------------------------------------------------------------------

MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
MD_CODE = re.compile(r"`([^`]+)`")


def parse_markdown(rel: str, text: str, file_set: set[str],
                   symbol_index: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    nodes, edges = [], []
    mnode = f"md:{rel}"
    nodes.append({"id": mnode, "kind": "doc", "label": rel, "file": rel,
                  "desc": md_summary(text)})
    here = Path(rel).parent
    for m in MD_LINK.finditer(text):
        target = urllib.parse.unquote(m.group(1).split("#")[0].strip())
        if not target or target.startswith(("http:", "https:", "mailto:")):
            continue
        resolved = (here / target).as_posix()
        if resolved in file_set:
            if resolved.endswith(".py"):
                dst = f"py:{resolved}"
            elif Path(resolved).suffix.lower() in IMAGE_EXTS:
                dst = f"img:{resolved}"
            else:
                dst = f"md:{resolved}"
            edges.append({"src": mnode, "dst": dst, "kind": "references"})
    hits: Counter[str] = Counter()
    for m in MD_CODE.finditer(text):
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", m.group(1)):
            if token in symbol_index:
                hits[token] += 1
    for token, _ in hits.most_common(MAX_DOCSYM_EDGES_PER_FILE):
        edges.append({"src": mnode, "dst": symbol_index[token][0], "kind": "references"})
    return nodes, edges


# --------------------------------------------------------------------------
# graph build
# --------------------------------------------------------------------------

def build_graph(root: Path, descriptions: dict[str, dict] | None = None) -> tuple[dict, dict]:
    files = scan_files(root)
    file_set = set(files)
    py_files = sorted(f for f in files if files[f]["kind"] == "code")

    trees: dict[str, ast.AST] = {}
    locs: dict[str, int] = {}
    per_file: dict[str, tuple] = {}
    for rel in py_files:
        src_path = root / rel
        try:
            src = src_path.read_text(encoding="utf-8", errors="replace")
            trees[rel] = ast.parse(src)
            locs[rel] = src.count("\n") + 1
        except SyntaxError as exc:
            print(f"warn: {rel}: {exc}", file=sys.stderr)
            continue

    global_defs: dict[str, list[str]] = defaultdict(list)
    for rel in py_files:
        if rel not in trees:
            continue
        fnode = f"py:{rel}"
        for node in trees[rel].body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                global_defs[node.name].append(f"{fnode}::{node.name}")
    for name in global_defs:
        global_defs[name].sort()

    nodes: list[dict] = []
    edges: list[dict] = []
    for rel in py_files:
        if rel not in trees:
            continue
        n, e, imports, file_defs, _ = parse_python(rel, trees[rel], file_set,
                                                   locs.get(rel, 0))
        nodes.extend(n)
        edges.extend(e)
        per_file[rel] = (imports, file_defs)

    # calls resolved with the full symbol table
    for rel in py_files:
        if rel not in trees:
            continue
        fnode = f"py:{rel}"
        imports, file_defs = per_file[rel]
        for node in trees[rel].body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                edges.extend(calls_in_func(f"{fnode}::{node.name}", node, imports,
                                           file_defs, global_defs))
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        edges.extend(calls_in_func(f"{fnode}::{node.name}.{sub.name}",
                                                   sub, imports, file_defs,
                                                   global_defs))

    symbol_index: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        if n["kind"] in ("class", "function"):
            symbol_index[n["label"]].append(n["id"])
    for name in symbol_index:
        symbol_index[name].sort()

    # docs and other files
    for rel, info in sorted(files.items()):
        if info["kind"] == "code":
            continue
        if info["kind"] == "doc":
            try:
                text = (root / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            n, e = parse_markdown(rel, text, file_set, symbol_index)
            nodes.extend(n)
            edges.extend(e)
        elif info["kind"] == "image":
            nodes.append({"id": f"img:{rel}", "kind": "image", "label": rel, "file": rel,
                          "desc": (descriptions or {}).get(rel, {}).get("desc", "")})
        else:
            nodes.append({"id": f"cfg:{rel}", "kind": "config", "label": rel, "file": rel})

    graph = {"nodes": nodes, "edges": edges}
    resolve_edge_ends(graph)
    dedup_edges(graph)
    return graph, files


def resolve_edge_ends(graph: dict) -> None:
    """Resolve loose ends: inherits -> class node; unknown -> file node; else drop."""
    by_id = {n["id"]: n for n in graph["nodes"]}
    py_by_stem: dict[str, str] = {}
    for n in graph["nodes"]:
        if n["kind"] == "file" and n["file"].endswith(".py"):
            py_by_stem[module_name(n["file"])] = n["id"]
            py_by_stem.setdefault(n["file"].split("/")[-1][:-3], n["id"])
    kept = []
    for e in graph["edges"]:
        src, dst = e["src"], e["dst"]
        if e["kind"] == "inherits":
            matches = [nid for nid in by_id if nid.endswith("::" + dst.split(":")[-1])
                       and by_id[nid]["kind"] == "class"]
            if len(matches) == 1:
                e["dst"] = matches[0]
            else:
                continue
        if dst not in by_id:
            continue
        if src not in by_id:
            continue
        kept.append(e)
    graph["edges"] = kept


def dedup_edges(graph: dict) -> None:
    seen = set()
    out = []
    for e in graph["edges"]:
        key = (e["src"], e["dst"], e["kind"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    graph["edges"] = out


# --------------------------------------------------------------------------
# analytics: pagerank, degree, communities
# --------------------------------------------------------------------------

def pagerank(nodes: list[dict], edges: list[dict], iters: int = 40, d: float = 0.85):
    ids = [n["id"] for n in nodes]
    idx = {nid: i for i, nid in enumerate(ids)}
    out_adj = [[] for _ in ids]
    indeg = [0] * len(ids)
    for e in edges:
        if e["src"] in idx and e["dst"] in idx:
            s, t = idx[e["src"]], idx[e["dst"]]
            out_adj[s].append(t)
            indeg[t] += 1
    n = len(ids)
    if n == 0:
        return {}
    rank = [1.0 / n] * n
    for _ in range(iters):
        base = (1.0 - d) / n
        contribution = [0.0] * n
        for s, targets in enumerate(out_adj):
            if targets:
                share = d * rank[s] / len(targets)
                for t in targets:
                    contribution[t] += share
        rank = [base + c for c in contribution]
        total = sum(rank) or 1.0
        rank = [r / total for r in rank]
    return {ids[i]: rank[i] for i in range(n)}


def communities(nodes: list[dict], edges: list[dict]) -> dict[str, int]:
    """Deterministic label propagation over the undirected projection."""
    ids = sorted(n["id"] for n in nodes)
    idx = {nid: i for i, nid in enumerate(ids)}
    adj = defaultdict(set)
    for e in edges:
        if e["src"] in idx and e["dst"] in idx:
            a, b = idx[e["src"]], idx[e["dst"]]
            adj[a].add(b)
            adj[b].add(a)
    labels = list(range(len(ids)))  # seed = sorted order
    for _ in range(24):
        changed = False
        for i in range(len(ids)):
            if not adj[i]:
                continue
            counts = Counter(labels[j] for j in adj[i])
            best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            if labels[i] != best:
                labels[i] = best
                changed = True
        if not changed:
            break
    # relabel communities by size then smallest member id, for stable naming
    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(ids[i])
    ordered = sorted(groups.values(), key=lambda m: (-len(m), m[0]))
    out = {}
    for ci, members in enumerate(ordered):
        for m in members:
            out[m] = ci
    return out


def annotate(graph: dict) -> dict:
    pr = pagerank(graph["nodes"], graph["edges"])
    # importance excluding containment edges — what everything else actually touches
    structural = [e for e in graph["edges"] if e["kind"] != "contains"]
    hub = pagerank(graph["nodes"], structural)
    deg = Counter()
    for e in graph["edges"]:
        deg[e["src"]] += 1
        deg[e["dst"]] += 1
    com = communities(graph["nodes"], graph["edges"])
    for n in graph["nodes"]:
        nid = n["id"]
        n["pagerank"] = round(pr.get(nid, 0.0), 8)
        n["hub_rank"] = round(hub.get(nid, 0.0), 8)
        n["degree"] = deg.get(nid, 0)
        n["community"] = com.get(nid, 0)
    graph["nodes"].sort(key=lambda n: n["id"])
    graph["edges"].sort(key=lambda e: (e["src"], e["dst"], e["kind"]))
    return graph


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

PALETTE = ["#5b8def", "#4fc3a1", "#e6b455", "#e07a6a", "#a98ce0", "#67b7dc",
           "#d98cc4", "#8fce6a", "#e0a35c", "#6ac2b0", "#c48cd8", "#b0b0b0"]


def community_meta(graph: dict) -> list[dict]:
    groups = defaultdict(list)
    for n in graph["nodes"]:
        groups[n["community"]].append(n)
    metas = []
    for cid in sorted(groups):
        members = groups[cid]
        hub = max(members, key=lambda n: (n.get("hub_rank", 0.0), n["label"]))
        metas.append({
            "id": cid,
            "size": len(members),
            "hub": hub["label"],
            "color": PALETTE[cid % len(PALETTE)],
            "kinds": dict(Counter(n["kind"] for n in members)),
        })
    return metas


def write_graph_json(graph: dict, root: Path, out: Path) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root.name),
        "counts": {
            "nodes": len(graph["nodes"]),
            "edges": len(graph["edges"]),
            "kinds": dict(Counter(n["kind"] for n in graph["nodes"])),
        },
        "communities": community_meta(graph),
        "nodes": graph["nodes"],
        "edges": graph["edges"],
    }
    (out / "graph.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")


def write_html(graph: dict, out: Path, root_name: str = "") -> None:
    coms = community_meta(graph)
    data = {
        "root": root_name,
        "communities": coms,
        "nodes": [{k: n.get(k) for k in ("id", "kind", "label", "file", "line",
                                         "pagerank", "hub_rank", "degree", "community",
                                         "desc", "sig", "lines", "defs")}
                  for n in graph["nodes"]],
        "edges": graph["edges"],
    }
    page = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    (out / "index.html").write_text(page, encoding="utf-8")


def write_report(graph: dict, root: Path, out: Path) -> None:
    coms = community_meta(graph)
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    gods = sorted(graph["nodes"], key=lambda n: -n.get("hub_rank", 0.0))[:12]
    lines = [
        f"# Graph report — {root.name}",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
        f"{len(graph['nodes'])} nodes, {len(graph['edges'])} edges.",
        "",
        "Open `graphify-out/index.html` for the interactive graph; "
        "see `wiki/index.md` for per-community navigation.",
        "",
        "## God nodes (highest structural PageRank, containment edges excluded)",
        "",
        "| node | kind | degree | hub rank |",
        "| --- | --- | ---: | ---: |",
    ]
    for n in gods:
        lines.append(f"| `{n['label']}` | {n['kind']} | {n['degree']} | {n.get('hub_rank', 0):.5f} |")
    lines += ["", "## Communities", ""]
    singles: list[str] = []
    for c in coms:
        members = [n for n in graph["nodes"] if n["community"] == c["id"]]
        if len(members) == 1:
            singles.append(f"`{members[0]['label']}`")
            continue
        kinds = ", ".join(f"{k}×{v}" for k, v in sorted(c["kinds"].items()))
        lines.append(f"### {c['color']} community {c['id']} — hub `{c['hub']}` ({c['size']} nodes: {kinds})")
        lines.append("")
        core = sorted(members, key=lambda n: -n.get("hub_rank", 0.0))[:14]
        for n in core:
            loc = f" ({n['file']}:{n['line']})" if n.get("line") else (f" ({n['file']})" if n.get("file") else "")
            lines.append(f"- `{n['label']}` {n['kind']}{loc}")
        if len(members) > len(core):
            lines.append(f"- … and {len(members) - len(core)} more")
        lines.append("")
    if singles:
        lines.append(f"### Isolated nodes ({len(singles)})")
        lines.append("")
        lines.append("No graph edges touch these — listed for completeness: "
                     + ", ".join(singles) + ".")
        lines.append("")
    lines += ["## Top cross-file relationships", "",
              "| relation | count |", "| --- | ---: |"]
    rel = Counter()
    for e in graph["edges"]:
        s, d = nodes_by_id.get(e["src"]), nodes_by_id.get(e["dst"])
        if s and d and s.get("file") and d.get("file") and s["file"] != d["file"]:
            rel[(e["kind"], s["file"], d["file"])] += 1
    for (kind, sf, df), cnt in rel.most_common(15):
        lines.append(f"| `{sf}` —{kind}→ `{df}` | {cnt} |")
    imgs = [n for n in graph["nodes"] if n["kind"] == "image"]
    if imgs:
        lines += ["", "## Images", ""]
        for n in imgs:
            desc = n.get("desc") or "*(no description — run `images` merge)*"
            lines.append(f"- `{n['label']}` — {desc}")
    (out / "GRAPH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_wiki(graph: dict, out: Path) -> None:
    coms = community_meta(graph)
    wiki = out / "wiki"
    wiki.mkdir(parents=True, exist_ok=True)
    idx = ["# Wiki", "", "Broad navigation over the graph, one page per community.", ""]
    for c in coms:
        slug = f"community-{c['id']:02d}"
        members = sorted((n for n in graph["nodes"] if n["community"] == c["id"]),
                         key=lambda n: (-n.get("hub_rank", 0.0), n["label"]))
        idx.append(f"- [{slug} — {c['hub']}](wiki/{slug}.md) · {c['size']} nodes")
        page = [f"# Community {c['id']} — hub `{c['hub']}`", "",
                f"{c['size']} nodes; dominant kinds: "
                + ", ".join(f"{k}×{v}" for k, v in sorted(c["kinds"].items())), ""]
        for n in members:
            loc = f"`{n['file']}:{n['line']}`" if n.get("line") else f"`{n['file']}`" if n.get("file") else ""
            page.append(f"- `{n['label']}` ({n['kind']}) {loc} — hub rank {n.get('hub_rank', 0):.5f}")
        (wiki / f"{slug}.md").write_text("\n".join(page) + "\n", encoding="utf-8")
    (out / "wiki" / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# query / path / explain
# --------------------------------------------------------------------------

def load_graph(root: Path) -> dict:
    gj = root / "graphify-out" / "graph.json"
    if not gj.exists():
        sys.exit("no graphify-out/graph.json — run: python3 scripts/graphify.py build .")
    return json.loads(gj.read_text(encoding="utf-8"))


TOKEN_STOP = {"the", "and", "for", "with", "what", "which", "how", "does", "do",
              "is", "are", "of", "in", "on", "to", "a", "an", "this", "that", "where",
              "who", "uses", "use", "used", "from", "into", "when", "why", "can"}


def find_matches(graph: dict, needle: str) -> list[dict]:
    nd = needle.lower().strip()
    exact, sub = [], []
    for n in graph["nodes"]:
        hay = f"{n['label']} {n.get('file', '')}".lower()
        if n["label"].lower() == nd or n["id"].lower() == nd or n.get("file", "").lower() == nd:
            exact.append(n)
        elif nd in hay:
            sub.append(n)
    return sorted(exact, key=lambda n: -n["pagerank"]) + sorted(sub, key=lambda n: -n["pagerank"])


def adjacencies(graph: dict, undirected: bool = False):
    fwd, rev = defaultdict(list), defaultdict(list)
    for e in graph["edges"]:
        fwd[e["src"]].append(e)
        rev[e["dst"]].append(e)
    return fwd, rev


def cmd_query(graph: dict, question: str) -> str:
    tokens = [t for t in re.findall(r"[a-z0-9_]{2,}", question.lower()) if t not in TOKEN_STOP]
    fwd, rev = adjacencies(graph)
    scores: dict[str, float] = defaultdict(float)
    for n in graph["nodes"]:
        hay = f"{n['label']} {n.get('file','')} {n.get('desc','')}".lower()
        for t in tokens:
            if t in hay:
                scores[n["id"]] += 2.0 if t in n["label"].lower() else 1.0
    boosted = defaultdict(float, scores)
    for nid, s in list(scores.items()):
        for e in fwd.get(nid, []) + rev.get(nid, []):
            other = e["dst"] if e["src"] == nid else e["src"]
            boosted[other] += 0.3 * s
    top = sorted(boosted.items(), key=lambda kv: (-kv[1], kv[0]))[:14]
    if not top:
        return f"no nodes match: {question!r}"
    keep = {nid for nid, _ in top}
    lines = [f"subgraph for: {question!r}", ""]
    for nid, sc in top:
        n = next(x for x in graph["nodes"] if x["id"] == nid)
        loc = f" ({n['file']}:{n['line']})" if n.get("line") else (f" ({n['file']})" if n.get("file") else "")
        lines.append(f"- `{n['label']}` {n['kind']}{loc} score {sc:.1f}, pagerank {n['pagerank']:.5f}")
    lines += ["", "edges inside the subgraph:"]
    shown = 0
    for e in graph["edges"]:
        if e["src"] in keep and e["dst"] in keep and shown < 25:
            lines.append(f"- `{e['src']}` —{e['kind']}→ `{e['dst']}`")
            shown += 1
    return "\n".join(lines)


def cmd_path(graph: dict, a: str, b: str, undirected: bool) -> str:
    ma, mb = find_matches(graph, a)[:1], find_matches(graph, b)[:1]
    if not ma or not mb:
        return f"no match for {a!r} or {b!r}"
    src, dst = ma[0]["id"], mb[0]["id"]
    fwd, rev = adjacencies(graph)
    neighbors = (
        (lambda nid: [(e["dst"], e) for e in fwd.get(nid, [])] + [(e["src"], e) for e in rev.get(nid, [])])
        if undirected else
        (lambda nid: [(e["dst"], e) for e in fwd.get(nid, [])])
    )
    prev: dict[str, tuple[str, dict]] = {}
    q = deque([src])
    seen = {src}
    while q:
        cur = q.popleft()
        if cur == dst:
            break
        for nxt, e in sorted(neighbors(cur), key=lambda t: t[0]):
            if nxt not in seen:
                seen.add(nxt)
                prev[nxt] = (cur, e)
                q.append(nxt)
    if dst not in seen and dst != src:
        return f"no path from `{src}` to `{dst}` ({'undirected' if undirected else 'directed'} edges)"
    chain, cur = [], dst
    while cur != src:
        p, e = prev[cur]
        chain.append((p, e, cur))
        cur = p
    lines = [f"`{src}`"]
    for p, e, c in reversed(chain):
        lines.append(f"  —{e['kind']}→  `{c}`")
    out = [f"path ({'undirected' if undirected else 'directed'}), {len(chain)} hop(s):", ""]
    out.append(lines[0])
    out.extend(lines[1:])
    return "\n".join(out)


def cmd_explain(graph: dict, concept: str) -> str:
    matches = find_matches(graph, concept)
    if not matches:
        return f"no node matches {concept!r}"
    n = matches[0]
    nid = n["id"]
    fwd, rev = adjacencies(graph)
    lines = [f"`{n['label']}` — {n['kind']}" +
             (f" — `{n['file']}:{n['line']}`" if n.get("line") else
              (f" — `{n['file']}`" if n.get("file") else "")), ""]
    if n.get("desc"):
        lines += [n["desc"], ""]
    methods = [e["dst"] for e in fwd.get(nid, []) if e["kind"] == "contains"]
    callees = [e["dst"] for e in fwd.get(nid, []) if e["kind"] in ("calls", "imports")]
    callers = [e["src"] for e in rev.get(nid, []) if e["kind"] in ("calls", "imports")]
    referenced = [e["src"] for e in rev.get(nid, []) if e["kind"] == "references"]
    if methods:
        lines.append(f"methods/members ({len(methods)}):")
        for m in methods[:20]:
            mn = next(x for x in graph["nodes"] if x["id"] == m)
            lines.append(f"  - `{mn['label']}`" + (f" `{mn['file']}:{mn['line']}`" if mn.get("line") else ""))
        lines.append("")
    if callees:
        lines.append(f"calls/imports ({len(callees)}):")
        for c in callees[:20]:
            cn = next(x for x in graph["nodes"] if x["id"] == c)
            lines.append(f"  - `{cn['label']}`" + (f" ({cn['file']})" if cn.get("file") else ""))
        lines.append("")
    if callers:
        lines.append(f"callers/importers ({len(callers)}):")
        for c in callers[:20]:
            cn = next(x for x in graph["nodes"] if x["id"] == c)
            lines.append(f"  - `{cn['label']}`" + (f" ({cn['file']})" if cn.get("file") else ""))
        lines.append("")
    if referenced:
        lines.append(f"referenced by ({len(referenced)}):")
        for c in referenced[:12]:
            cn = next(x for x in graph["nodes"] if x["id"] == c)
            lines.append(f"  - `{cn['label']}`")
        lines.append("")
    return "\n".join(lines).rstrip()


# --------------------------------------------------------------------------
# cache / update / image descriptions
# --------------------------------------------------------------------------

def save_cache(root: Path, files: dict, graph: dict) -> None:
    cache = {rel: {"sha": info["sha"]} for rel, info in files.items()}
    (root / "graphify-out" / "cache.json").write_text(json.dumps(cache), encoding="utf-8")


def load_descriptions(root: Path) -> dict:
    p = root / "graphify-out" / "image_descriptions.json"
    if not p.exists():
        return {}
    try:
        return {d["path"]: d for d in json.loads(p.read_text(encoding="utf-8"))}
    except (ValueError, KeyError):
        return {}


def pending_images(root: Path) -> list[str]:
    """Image files that still lack a description (for the vision pass)."""
    graph = load_graph(root)
    return sorted(n["file"] for n in graph["nodes"]
                  if n["kind"] == "image" and not n.get("desc"))


# --------------------------------------------------------------------------
# html template
# --------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>graphify — code explorer</title>
<style>
  :root { --bg:#0d1015; --panel:#151b24; --panel2:#10151d; --ink:#d9dfe8; --mut:#8a93a3;
          --line:#232b38; --accent:#5b8def; --soft:#9ecbff; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--ink);
              font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; }
  #app { display:flex; height:100%; }
  /* ------- sidebar ------- */
  #side { width:320px; min-width:320px; border-right:1px solid var(--line);
          background:var(--panel2); display:flex; flex-direction:column; }
  #searchbox { padding:10px; border-bottom:1px solid var(--line); }
  #q { width:100%; background:#0b0e13; border:1px solid var(--line); color:var(--ink);
       border-radius:7px; padding:7px 9px; outline:none; font:inherit; }
  #q:focus { border-color:var(--accent); }
  #qresults { margin-top:6px; max-height:45vh; overflow:auto; }
  #qresults div { padding:5px 7px; border-radius:6px; cursor:pointer; }
  #qresults div:hover { background:#1a2230; }
  #qresults .d { color:var(--mut); font-size:12px; }
  #tree { flex:1; overflow:auto; padding:10px 8px 30px; }
  #tree details { padding-left:0; }
  #tree details details { padding-left:14px; }
  #tree summary { cursor:pointer; padding:3px 6px; border-radius:6px; list-style:none; }
  #tree summary::-webkit-details-marker { display:none; }
  #tree summary::before { content:'▸ '; color:var(--mut); }
  #tree details[open] > summary::before { content:'▾ '; }
  #tree .leaf { padding:2px 6px 2px 22px; border-radius:6px; cursor:pointer; color:#b9c2d0; }
  #tree .leaf:hover { background:#1a2230; }
  #tree .active { background:#1c2940; color:#fff; }
  #tree .dir > summary { color:#cdd6e4; font-weight:600; }
  #tree .mod > summary { color:var(--soft); }
  #tree .cnt { color:var(--mut); font-size:11px; }
  /* ------- main ------- */
  #main { flex:1; overflow:auto; padding:0 0 80px; }
  #top { position:sticky; top:0; z-index:5; background:rgba(13,16,21,.96);
         border-bottom:1px solid var(--line); padding:10px 22px; display:flex;
         gap:10px; align-items:center; }
  #back { background:#0b0f15; border:1px solid var(--line); color:var(--ink); border-radius:7px;
          padding:5px 12px; cursor:pointer; font:inherit; }
  #back:hover { border-color:var(--accent); }
  #crumb { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
  #crumb span { cursor:pointer; padding:3px 8px; border-radius:6px; }
  #crumb span:hover { background:#1a2230; }
  #crumb .sep { color:var(--mut); cursor:default; }
  #crumb .here { background:#1c2940; color:#fff; cursor:default; }
  #content { max-width:980px; margin:0 auto; padding:26px 26px 40px; }
  h1 { font-size:21px; margin:0 0 4px; }
  h2 { font-size:15px; color:var(--soft); margin:34px 0 10px; text-transform:uppercase;
       letter-spacing:.08em; }
  .lead { color:#c4cdd9; margin:6px 0 0; }
  .meta { color:var(--mut); font-size:12.5px; margin-top:2px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:12px;
          padding:14px 16px; margin:10px 0; cursor:pointer; transition:border-color .15s,
          transform .15s; }
  .card:hover { border-color:var(--accent); transform:translateY(-1px); }
  .card b { color:#fff; }
  .card p { margin:6px 0 0; color:#aeb8c6; }
  .card .meta { margin-top:4px; }
  .badge { display:inline-block; background:#1c2940; color:var(--soft); border-radius:20px;
           padding:1px 9px; font-size:11px; margin-left:8px; vertical-align:middle; }
  .sig { display:block; background:#0b0f15; border:1px solid var(--line); border-radius:8px;
         padding:8px 11px; color:var(--soft); font-size:12.5px; margin:8px 0 0;
         white-space:pre-wrap; word-break:break-all; }
  .row { background:var(--panel); border:1px solid var(--line); border-radius:11px;
         padding:11px 14px; margin:8px 0; cursor:pointer; }
  .row:hover { border-color:var(--accent); background:#17202c; }
  .row .t { color:#fff; font-weight:600; }
  .row > code { color:var(--mut); font-size:12px; margin-left:8px; }
  .row p { margin:4px 0 0; color:#aeb8c6; font-size:13px; }
  .row p.none { color:#5c6675; font-style:italic; }
  .flowlist { margin:6px 0 0; padding-left:22px; }
  .flowlist li { padding:2px 0; color:#b9c2d0; }
  .flowlist .nav { cursor:pointer; color:var(--soft); }
  .flowlist .nav:hover { text-decoration:underline; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  @media (max-width:1100px){ .grid { grid-template-columns:1fr; } }
  .note { background:#141c12; border:1px solid #2c3d24; border-radius:12px; padding:12px 16px;
          color:#c9d6bd; margin:14px 0; }
  .none { color:var(--mut); font-style:italic; }
</style>
</head>
<body>
<div id="app">
  <div id="side">
    <div id="searchbox">
      <input id="q" placeholder="search anything…  (⌘K)" autocomplete="off">
      <div id="qresults"></div>
    </div>
    <div id="tree"></div>
  </div>
  <div id="main">
    <div id="top">
      <button id="back">← back</button>
      <div id="crumb"></div>
    </div>
    <div id="content"></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const byId = {}; DATA.nodes.forEach(n => byId[n.id] = n);
const outEdges = {}, inEdges = {};
DATA.edges.forEach(e => {
  (outEdges[e.src] = outEdges[e.src] || []).push(e);
  (inEdges[e.dst] = inEdges[e.dst] || []).push(e);
});
const KIND_GLYPH = {file:'▤', doc:'❏', image:'▣', config:'⛭', class:'◆', function:'●', method:'○'};
const pyFiles = DATA.nodes.filter(n => n.kind === 'file');
const docs    = DATA.nodes.filter(n => n.kind === 'doc');
const images  = DATA.nodes.filter(n => n.kind === 'image');
const configs = DATA.nodes.filter(n => n.kind === 'config');
const symbols = DATA.nodes.filter(n => ['class','function','method'].includes(n.kind));
const totalLoc = pyFiles.reduce((s,n) => s + (n.loc||0), 0);
const readMe = byId['md:README.md'] || docs[0];

function esc(s){ return String(s == null ? '' : s)
  .replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function shortSig(sig, cap){ cap = cap || 96;
  if (!sig) return '';
  return sig.length > cap ? sig.slice(0, cap) + '…' : sig; }
function summaryOf(n){
  if (n.desc) return esc(n.desc);
  if (n.kind === 'function' || n.kind === 'method') return 'No docstring — see the signature and its callers.';
  return '';
}
function dirOf(n){ const f = n.file || ''; return f.includes('/') ? f.slice(0, f.lastIndexOf('/')) : '· root'; }

/* ---------------- sidebar tree ---------------- */
function buildTree(){
  const dirs = {};
  pyFiles.forEach(f => (dirs[dirOf(f)] = dirs[dirOf(f)] || []).push(f));
  const order = Object.keys(dirs).sort();
  let html = '<details open class="dir"><summary>▤ ' + esc(DATA.root) +
             ' <span class="cnt">' + pyFiles.length + ' files · ' + totalLoc.toLocaleString() +
             ' lines</span></summary>';
  order.forEach(d => {
    const mods = dirs[d].sort((a,b) => a.file.localeCompare(b.file));
    html += '<details class="dir"><summary>' + esc(d === '· root' ? '/ (root)' : d + '/') +
            ' <span class="cnt">' + mods.length + '</span></summary>';
    mods.forEach(m => {
      const syms = symbols.filter(s => s.file === m.file);
      const cls = syms.filter(s => s.kind === 'class');
      const fns = syms.filter(s => s.kind === 'function');
      html += '<details class="mod"><summary data-id="' + esc(m.id) + '">▤ ' +
              esc(m.file.split('/').pop()) + ' <span class="cnt">' + syms.length +
              (m.loc ? ' · ' + m.loc + 'L' : '') + '</span></summary>';
      cls.forEach(c => {
        html += '<div class="leaf" data-id="' + esc(c.id) + '">◆ ' + esc(c.label) + '</div>';
        const methods = syms.filter(s => s.kind === 'method' && s.file === c.file &&
                                        s.label.startsWith(c.label + '.'));
        methods.forEach(mm => {
          html += '<div class="leaf" data-id="' + esc(mm.id) + '" style="padding-left:34px">○ ' +
                  esc(mm.label.split('.').pop()) + '</div>';
        });
      });
      fns.forEach(f => {
        html += '<div class="leaf" data-id="' + esc(f.id) + '">● ' + esc(f.label) + '</div>';
      });
      html += '</details>';
    });
    html += '</details>';
  });
  docs.forEach(d => { html += '<div class="leaf" data-id="' + esc(d.id) + '">❏ ' + esc(d.label) + '</div>'; });
  images.forEach(i => { html += '<div class="leaf" data-id="' + esc(i.id) + '">▣ ' + esc(i.label) + '</div>'; });
  document.getElementById('tree').innerHTML = html;
  wireNav(document.getElementById('tree'));
}

/* ---------------- flow trees (data flow) ---------------- */
function flowTree(fid, depth, seen){
  if (depth > 2 || seen.has(fid)) return '';
  seen = seen || new Set([fid]);
  const es = (outEdges[fid] || []).filter(e => e.kind === 'calls' || e.kind === 'imports');
  if (!es.length) return '';
  let html = '<ul class="flowlist">';
  es.slice(0, 8).forEach(e => {
    const t = byId[e.dst]; if (!t) return;
    html += '<li><span class="nav" data-id="' + esc(t.id) + '">' + esc(t.label) + '</span>' +
            (t.desc ? ' <span style="color:#77808f">— ' + esc(t.desc.slice(0, 90)) + '</span>' : '') +
            flowTree(t.id, depth + 1, new Set([...seen, fid])) + '</li>';
  });
  return html + '</ul>';
}

/* ---------------- views ---------------- */
function crumbSet(parts){
  const c = document.getElementById('crumb');
  let html = '';
  parts.forEach((p, i) => {
    const last = i === parts.length - 1;
    html += '<span class="' + (last ? 'here' : '') + '" data-id="' + esc(p.id || '') + '">' +
            esc(p.label) + '</span>';
    if (!last) html += '<span class="sep">›</span>';
  });
  c.innerHTML = html;
  c.querySelectorAll('span[data-id]').forEach(el => {
    if (el.dataset.id && !el.classList.contains('here'))
      el.onclick = () => navigate(el.dataset.id);
  });
}

function showRepo(){
  crumbSet([{label: DATA.root}]);
  const hubFiles = pyFiles.slice().sort((a,b) => (b.hub_rank||0)-(a.hub_rank||0)).slice(0, 4);
  const mains = symbols.filter(s => s.kind === 'function' && s.label === 'main');
  let html = '<h1>▤ ' + esc(DATA.root) + '</h1>' +
    '<div class="meta">' + pyFiles.length + ' Python files · ' + symbols.length +
    ' functions &amp; classes · ' + totalLoc.toLocaleString() + ' lines · ' +
    docs.length + ' docs · ' + images.length + ' figures</div>' +
    (readMe ? '<p class="lead">' + esc(readMe.desc) + '</p>' : '') +
    '<div class="note"><b>New here?</b> Read top to bottom: what each module does, ' +
    'how data flows through the pipeline, then click any module or function for its ' +
    'own summary. Everything is clickable — the sidebar tree mirrors this content.</div>' +
    '<h2>Start with these</h2>';
  hubFiles.forEach(f => {
    html += '<div class="row" data-id="' + esc(f.id) + '"><span class="t">▤ ' + esc(f.label) +
            '</span><p>' + (summaryOf(f) || '<span class="none">No docstring.</span>') + '</p></div>';
  });
  html += '<h2>How data flows</h2><p class="meta">Entry points (main) and what they call, depth 2.</p>';
  if (!mains.length) html += '<p class="none">No main() entry points found.</p>';
  mains.forEach(m => {
    html += '<div class="card" data-id="' + esc(m.id) + '" style="cursor:default"><b>● ' +
            esc(m.label) + '</b> <span class="meta">' + esc(m.file) + ':' + m.line + '</span>' +
            flowTree(m.id, 0, new Set()) + '</div>';
  });
  html += '<h2>Modules by directory</h2>';
  const dirs = {};
  pyFiles.forEach(f => (dirs[dirOf(f)] = dirs[dirOf(f)] || []).push(f));
  Object.keys(dirs).sort().forEach(d => {
    const mods = dirs[d].sort((a,b) => (b.hub_rank||0)-(a.hub_rank||0));
    html += '<h2 style="margin-top:22px">' + esc(d === '· root' ? '/ (root files)' : d + '/') + '</h2>';
    mods.forEach(m => {
      const symsN = symbols.filter(s => s.file === m.file).length;
      html += '<div class="card" data-id="' + esc(m.id) + '"><b>▤ ' + esc(m.file) + '</b>' +
        '<span class="badge">' + symsN + ' symbols' + (m.loc ? ' · ' + m.loc + ' lines' : '') + '</span>' +
        '<p>' + (summaryOf(m) || '<span class="none">No docstring.</span>') + '</p></div>';
    });
  });
  html += '<h2>Docs &amp; notes</h2>';
  docs.forEach(d => {
    html += '<div class="row" data-id="' + esc(d.id) + '"><span class="t">❏ ' + esc(d.label) +
            '</span><p>' + (summaryOf(d) || '<span class="none">—</span>') + '</p></div>';
  });
  if (images.length){
    html += '<h2>Figures</h2>';
    images.forEach(i => {
      html += '<div class="row" data-id="' + esc(i.id) + '"><span class="t">▣ ' + esc(i.label) +
              '</span><p>' + (summaryOf(i) || '<span class="none">No description.</span>') + '</p></div>';
    });
  }
  document.getElementById('content').innerHTML = html;
  wireNav(document.getElementById('content'));
}

function importsOf(fileId){
  const set = new Map();
  (outEdges[fileId] || []).filter(e => e.kind === 'imports').forEach(e => {
    const t = byId[e.dst]; if (t && t.id !== fileId) set.set(t.id, t);
  });
  return [...set.values()];
}
function importersOf(fileId){
  const set = new Map();
  (inEdges[fileId] || []).filter(e => e.kind === 'imports').forEach(e => {
    const s = byId[e.src]; if (s && s.id !== fileId) set.set(s.id, s);
  });
  return [...set.values()];
}

function showModule(id){
  const m = byId[id];
  crumbSet([{label: DATA.root, id: '__repo__'}, {label: m.file, id: m.id}]);
  const uses = importsOf(m.id), usedBy = importersOf(m.id);
  const syms = symbols.filter(s => s.file === m.file);
  const cls = syms.filter(s => s.kind === 'class');
  const fns = syms.filter(s => s.kind === 'function');
  let html = '<h1>▤ ' + esc(m.file) + '</h1>' +
    '<div class="meta">' + syms.length + ' symbols' + (m.loc ? ' · ' + m.loc + ' lines' : '') +
    ' · ' + uses.length + ' imports · imported by ' + usedBy.length + '</div>' +
    (m.desc ? '<p class="lead">' + esc(m.desc) + '</p>' : '<p class="none">No module docstring.</p>');
  html += '<h2>What it uses (imports)</h2>';
  html += uses.length ? uses.map(u =>
    '<div class="row" data-id="' + esc(u.id) + '"><span class="t">▤ ' + esc(u.label) +
    '</span><p>' + (summaryOf(u) || '<span class="none">—</span>') + '</p></div>').join('')
    : '<p class="none">No in-repo imports — standalone module.</p>';
  html += '<h2>Who uses it</h2>';
  html += usedBy.length ? usedBy.map(u =>
    '<div class="row" data-id="' + esc(u.id) + '"><span class="t">▤ ' + esc(u.label) +
    '</span><p>' + (summaryOf(u) || '<span class="none">—</span>') + '</p></div>').join('')
    : '<p class="none">Nothing imports this — entry point or leaf.</p>';
  html += '<h2>Classes</h2>';
  html += cls.length ? cls.map(c =>
    '<div class="row" data-id="' + esc(c.id) + '"><span class="t">◆ ' + esc(c.label) +
    '</span><code>' + esc(shortSig(c.sig, 110)) + '</code>' +
    '<p>' + (summaryOf(c) || '<span class="none">No docstring.</span>') + '</p></div>').join('')
    : '<p class="none">None.</p>';
  html += '<h2>Functions</h2>';
  html += fns.length ? fns.map(f =>
    '<div class="row" data-id="' + esc(f.id) + '"><span class="t">● ' + esc(f.label) +
    '</span><code>' + esc(shortSig(f.sig, 110)) + '</code>' +
    '<p>' + (summaryOf(f) || '<span class="none">No docstring.</span>') + '</p></div>').join('')
    : '<p class="none">None.</p>';
  document.getElementById('content').innerHTML = html;
  wireNav(document.getElementById('content'));
}

function relRows(title, pairs, emptyMsg){
  if (!pairs.length) return '<h2>' + title + '</h2><p class="none">' + emptyMsg + '</p>';
  return '<h2>' + title + ' (' + pairs.length + ')</h2>' + pairs.map(t =>
    '<div class="row" data-id="' + esc(t.id) + '"><span class="t">' + (KIND_GLYPH[t.kind]||'·') +
    ' ' + esc(t.label) + '</span><code>' + esc(shortSig(t.sig, 96)) + '</code>' +
    '<p>' + (summaryOf(t) || '<span class="none">—</span>') + '</p></div>').join('');
}

function showSymbol(id){
  const n = byId[id];
  const parentFile = 'py:' + n.file;
  crumbSet([{label: DATA.root, id: '__repo__'},
            {label: n.file, id: parentFile},
            {label: n.label, id: n.id}]);
  const callees = (outEdges[id] || [])
    .filter(e => e.kind === 'calls' || e.kind === 'imports')
    .map(e => byId[e.dst]).filter(Boolean);
  const callers = (inEdges[id] || [])
    .filter(e => e.kind === 'calls' || e.kind === 'imports')
    .map(e => byId[e.src]).filter(Boolean);
  const members = (outEdges[id] || []).filter(e => e.kind === 'contains')
    .map(e => byId[e.dst]).filter(Boolean);
  const reffed = (inEdges[id] || []).filter(e => e.kind === 'references')
    .map(e => byId[e.src]).filter(Boolean);
  let html = '<h1>' + (KIND_GLYPH[n.kind] || '·') + ' ' + esc(n.label) + '</h1>' +
    '<div class="meta">' + esc(n.kind) + ' · ' + esc(n.file) + (n.line ? ':' + n.line : '') +
    (n.lines ? ' · ' + n.lines + ' lines' : '') + '</div>' +
    (n.sig ? '<code class="sig">' + esc(n.sig) + '</code>' : '') +
    (n.desc ? '<p class="lead">' + esc(n.desc) + '</p>'
            : '<p class="none">No docstring — the signature and callers below tell the story.</p>');
  if (n.kind === 'class' && members.length)
    html += relRows('Members', members, 'None.');
  html += relRows('Calls →', callees, 'Calls nothing in-repo.');
  html += relRows('Called by ←', callers, 'Nothing calls it in-repo — likely an entry point.');
  html += relRows('Referenced by docs', reffed, '—');
  document.getElementById('content').innerHTML = html;
  wireNav(document.getElementById('content'));
}

function showDocLike(id){
  const n = byId[id];
  crumbSet([{label: DATA.root, id: '__repo__'}, {label: n.label, id: n.id}]);
  const refs = (outEdges[id] || []).filter(e => e.kind === 'references')
    .map(e => byId[e.dst]).filter(Boolean);
  let html = '<h1>' + (KIND_GLYPH[n.kind] || '·') + ' ' + esc(n.label) + '</h1>' +
    '<div class="meta">' + esc(n.kind) + (n.file ? ' · ' + esc(n.file) : '') + '</div>' +
    (n.desc ? '<p class="lead">' + esc(n.desc) + '</p>' : '');
  if (n.defs && n.defs.length)
    html += relRows('Defines', n.defs.map(d => byId[d.id]).filter(Boolean), '—');
  html += relRows('Points at', refs, 'No in-repo links captured.');
  document.getElementById('content').innerHTML = html;
  wireNav(document.getElementById('content'));
}

/* ---------------- navigation ---------------- */
const history = [];
function navigate(id, push){
  if (push !== false && history[history.length-1] !== id) history.push(id);
  const n = byId[id];
  if (!n){ showRepo(); return; }
  if (n.kind === 'file') showModule(id);
  else if (n.kind === 'doc' || n.kind === 'image' || n.kind === 'config') showDocLike(id);
  else showSymbol(id);
  document.getElementById('main').scrollTop = 0;
  markActive(id);
}
document.getElementById('back').onclick = () => {
  history.pop();
  const prev = history[history.length-1];
  if (prev) navigate(prev, false); else showRepo();
  markActive((byId[history[history.length-1]]||{}).id);
};
function wireNav(root){
  root.querySelectorAll('[data-id]').forEach(el => {
    if (el.dataset.id && el.dataset.id !== '__repo__' && !el.onclick)
      el.onclick = (ev) => { ev.stopPropagation(); navigate(el.dataset.id); };
  });
  root.querySelectorAll('[data-id="__repo__"]').forEach(el => { el.onclick = () => { history.length = 0; showRepo(); }; });
}
function markActive(id){
  document.querySelectorAll('#tree .leaf, #tree summary').forEach(el =>
    el.classList.toggle('active', !!id && el.dataset.id === id));
}

/* ---------------- search ---------------- */
const q = document.getElementById('q'), qres = document.getElementById('qresults');
q.addEventListener('input', () => {
  const v = q.value.toLowerCase().trim();
  qres.innerHTML = '';
  if (!v) return;
  DATA.nodes
    .filter(n => n.label.toLowerCase().includes(v) || (n.file||'').toLowerCase().includes(v)
                 || (n.desc||'').toLowerCase().includes(v))
    .sort((a,b) => (b.hub_rank||0)-(a.hub_rank||0)).slice(0, 14)
    .forEach(n => {
      const d = document.createElement('div');
      d.innerHTML = '<div>' + (KIND_GLYPH[n.kind]||'·') + ' ' + esc(n.label) +
                    '</div><div class="d">' + esc(n.kind + (n.file ? ' · ' + n.file : '')) + '</div>';
      d.onclick = () => { navigate(n.id); qres.innerHTML = ''; q.value = ''; };
      qres.appendChild(d);
    });
});
addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k'){ e.preventDefault(); q.focus(); }
});

buildTree();
showRepo();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def do_build(root: Path) -> None:
    out = root / "graphify-out"
    out.mkdir(parents=True, exist_ok=True)
    descriptions = load_descriptions(root)
    graph, files = build_graph(root, descriptions)
    graph = annotate(graph)
    write_graph_json(graph, root, out)
    write_html(graph, out, root.name)
    write_report(graph, root, out)
    write_wiki(graph, out)
    save_cache(root, files, graph)
    pending = pending_images(root)
    kinds = Counter(n["kind"] for n in graph["nodes"])
    print(f"graphify: {len(graph['nodes'])} nodes ({', '.join(f'{k}={v}' for k, v in sorted(kinds.items()))}), "
          f"{len(graph['edges'])} edges -> {out}/")
    print("wrote graph.json, index.html, GRAPH_REPORT.md, wiki/index.md, cache.json")
    if pending:
        print(f"{len(pending)} image(s) still undescribed: {', '.join(pending)}")
        print("describe them (vision pass) and merge with: python3 scripts/graphify.py images <desc.json>")


def do_update(root: Path) -> None:
    out = root / "graphify-out"
    old_cache: dict = {}
    cp = out / "cache.json"
    if cp.exists():
        try:
            old_cache = json.loads(cp.read_text(encoding="utf-8"))
        except ValueError:
            pass
    do_build(root)
    new_cache = {}
    if cp.exists():
        try:
            new_cache = json.loads(cp.read_text(encoding="utf-8"))
        except ValueError:
            pass
    changed = sorted(r for r in new_cache
                     if r in old_cache and new_cache[r]["sha"] != old_cache[r]["sha"])
    added = sorted(r for r in new_cache if r not in old_cache)
    removed = sorted(r for r in old_cache if r not in new_cache)
    print(f"update: {len(changed)} changed, {len(added)} added, {len(removed)} removed"
          + (f" — changed: {', '.join(changed[:8])}" if changed else ""))


def do_images(root: Path, desc_file: Path) -> None:
    raw = json.loads(desc_file.read_text(encoding="utf-8"))
    entries = raw if isinstance(raw, list) else raw.get("images", [])
    dst = root / "graphify-out" / "image_descriptions.json"
    merged = {}
    if dst.exists():
        try:
            for d in json.loads(dst.read_text(encoding="utf-8")):
                merged[d["path"]] = d
        except ValueError:
            pass
    for d in entries:
        if isinstance(d, dict) and d.get("path"):
            merged[d["path"]] = {"path": d["path"], "desc": d.get("desc", "")}
    dst.write_text(json.dumps([merged[k] for k in sorted(merged)], indent=2), encoding="utf-8")
    print(f"merged {len(entries)} description(s) into {dst}")
    do_build(root)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="graphify", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name in ("build", "update"):
        p = sub.add_parser(name)
        p.add_argument("root", nargs="?", default=".")
    p = sub.add_parser("query"); p.add_argument("question"); p.add_argument("root", nargs="?", default=".")
    p = sub.add_parser("path"); p.add_argument("a"); p.add_argument("b")
    p.add_argument("--undirected", action="store_true"); p.add_argument("root", nargs="?", default=".")
    p = sub.add_parser("explain"); p.add_argument("concept"); p.add_argument("root", nargs="?", default=".")
    p = sub.add_parser("images"); p.add_argument("descriptions"); p.add_argument("root", nargs="?", default=".")
    args = ap.parse_args(argv)
    if args.cmd is None:
        ap.print_help()
        return
    root = Path(getattr(args, "root", ".") or ".").resolve()

    if args.cmd == "build":
        do_build(root)
    elif args.cmd == "update":
        do_update(root)
    elif args.cmd == "images":
        do_images(root, Path(args.descriptions))
    else:
        graph = load_graph(root)
        if args.cmd == "query":
            print(cmd_query(graph, args.question))
        elif args.cmd == "path":
            print(cmd_path(graph, args.a, args.b, args.undirected))
        elif args.cmd == "explain":
            print(cmd_explain(graph, args.concept))


if __name__ == "__main__":
    main()
