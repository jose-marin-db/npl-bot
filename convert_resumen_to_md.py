#!/usr/bin/env python3
"""Convierte resumen.txt plano (preguntas + párrafos reenvueltos) a Markdown."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def flush_paragraph(buf: list[str], out: list[str]) -> None:
    if not buf:
        return
    text = " ".join(s.strip() for s in buf)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        out.append(text + "\n\n")


def is_bullet(line: str) -> bool:
    s = line.strip()
    return s.startswith("• ") or s.startswith("•\t")


def is_ordered_item_start(line: str) -> bool:
    return bool(re.match(r"^\s*\d+\.\s", line))


def is_heading_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if s.startswith("¿"):
        return True
    prefixes = (
        "Describa ",
        "Describí ",
        "Diseñe ",
        "Diseña ",
        "Explique ",
        "Explicá ",
        "Mencione ",
        "Cuándo conviene ",
        "En qué capas ",
        "En qué ",
        "Que tipos ",
        "Que es ",
        "Qué es ",
        "Que rol ",
        "Qué rol ",
        "Cual es ",
        "Cuál es ",
        "Se pueden coleccionar",
        "Tengo un problema",
        "Exlicá ",
    )
    if any(s.startswith(p) for p in prefixes):
        return True
    # Preguntas sin ¿ inicial (OCR / mezclas)
    if re.match(r"^(Que|Qué|Cuál|Cual|Cómo|Como)\s+.+\?$", s) and len(s) < 180:
        return True
    return False


def merge_until_question_mark(lines: list[str], start: int) -> tuple[str, int]:
    """Une líneas hasta que el texto acumulado contenga al menos un '?'."""
    parts: list[str] = [lines[start].strip()]
    j = start + 1
    acc = " ".join(parts)
    while j < len(lines) and "?" not in acc:
        nxt = lines[j].strip()
        if not nxt:
            break
        if is_bullet(lines[j]) or is_ordered_item_start(lines[j]):
            break
        if nxt.startswith("¿") and j > start:
            break
        if is_heading_line(lines[j]) and not nxt.startswith("¿"):
            break
        parts.append(nxt)
        acc = " ".join(parts)
        j += 1
    text = re.sub(r"\s+", " ", acc).strip()
    return text, j


def split_leading_question(text: str) -> tuple[str, str]:
    """Si empieza con ¿, separa título (hasta primer ?) y resto del párrafo."""
    t = text.strip()
    if not t.startswith("¿"):
        return "", t
    idx = t.find("?")
    if idx == -1:
        return t.strip(), ""
    head = t[: idx + 1].strip()
    tail = t[idx + 1 :].strip()
    return head, tail


def last_out_is_list_item(out: list[str]) -> bool:
    if not out:
        return False
    last = out[-1]
    return last.startswith("- ") or bool(re.match(r"^\d+\.\s", last))


def bullet_continuation_ok(prev_bullet_line: str, next_line: str) -> bool:
    n = next_line.strip()
    if not n:
        return False
    if is_bullet(next_line) or is_ordered_item_start(next_line) or n.startswith("¿"):
        return False
    if is_heading_line(next_line):
        return False
    # Nuevo párrafo tipo "Ejemplos clásicos:", "Nota:", etc.
    if re.match(r"^[A-ZÁÉÍÓÚÑ][^:]{0,60}:\s*$", n) or re.match(
        r"^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+ [^:]{0,50}:\s", n
    ):
        return False
    first = n[0]
    # Continuación típica de viñeta reenvuelta
    if first.islower() or first in "([\"'¡¿":
        return True
    if prev_bullet_line.rstrip().endswith(("-", "—", ":")):
        return True
    return False


def convert(lines: list[str]) -> str:
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n and not lines[i].strip():
        i += 1
    if i < n:
        out.append("# " + lines[i].strip() + "\n\n")
        i += 1

    para_buf: list[str] = []

    while i < n:
        raw = lines[i]
        line = raw.rstrip("\n")

        if not line.strip():
            flush_paragraph(para_buf, out)
            para_buf.clear()
            i += 1
            continue

        # Encabezados tipo Describa / Qué es ... (sin ¿ al inicio)
        if is_heading_line(line) and not line.strip().startswith("¿"):
            flush_paragraph(para_buf, out)
            para_buf.clear()
            s = line.strip()
            single_line_heading_prefixes = (
                "Describa ",
                "Describí ",
                "Diseñe ",
                "Diseña ",
                "Explique ",
                "Explicá ",
                "Mencione ",
                "Exlicá ",
                "Tengo un problema",
            )
            if s.endswith("?") or any(s.startswith(p) for p in single_line_heading_prefixes):
                out.append("## " + s + "\n\n")
                i += 1
            else:
                block, ni = merge_until_question_mark(lines, i)
                out.append("## " + block + "\n\n")
                i = ni
            continue

        if line.strip().startswith("¿"):
            flush_paragraph(para_buf, out)
            para_buf.clear()
            block, ni = merge_until_question_mark(lines, i)
            i = ni
            head, tail = split_leading_question(block)
            if head:
                out.append("## " + head + "\n\n")
            if tail:
                para_buf.append(tail)
            continue

        if is_bullet(line):
            flush_paragraph(para_buf, out)
            para_buf.clear()
            item = line.strip()[1:].lstrip()
            bullet_line = "- " + item.strip() + "\n"
            out.append(bullet_line)
            i += 1
            while i < n:
                nxt = lines[i]
                if not nxt.strip():
                    out.append("\n")
                    i += 1
                    break
                if not bullet_continuation_ok(out[-1], nxt):
                    break
                out[-1] = out[-1].rstrip("\n") + " " + nxt.strip() + "\n"
                i += 1
            continue

        if is_ordered_item_start(line):
            flush_paragraph(para_buf, out)
            para_buf.clear()
            s = line.strip()
            # Varios ítems en una línea: "1. 2. 3. 4. Texto..."
            if re.search(r"^\d+\.\s+\d+\.\s+", s):
                parts = re.split(r"(?=\s\d+\.\s)", " " + s)
                parts = [p.strip() for p in parts if p.strip()]
                for p in parts:
                    out.append(p + "\n")
                i += 1
            else:
                out.append(s + "\n")
                i += 1
                while i < n:
                    nxt = lines[i]
                    if not nxt.strip():
                        out.append("\n")
                        i += 1
                        break
                    if is_bullet(nxt) or nxt.strip().startswith("¿") or is_heading_line(nxt):
                        break
                    if is_ordered_item_start(nxt):
                        out.append(nxt.strip() + "\n")
                        i += 1
                        continue
                    out[-1] = out[-1].rstrip("\n") + " " + nxt.strip() + "\n"
                    i += 1
            continue

        # Signos sueltos por el PDF
        if line.strip() in {".", ",", '",', '"'}:
            if para_buf:
                para_buf[-1] = para_buf[-1].rstrip() + line.strip()
            elif out and out[-1].endswith("\n"):
                out[-1] = out[-1].rstrip("\n") + line.strip() + "\n"
            i += 1
            continue

        # Párrafo inmediatamente después de una viñeta o ítem numerado
        if (
            not para_buf
            and line.strip()
            and not is_bullet(line)
            and not is_ordered_item_start(line)
            and not line.strip().startswith("¿")
            and not is_heading_line(line)
            and last_out_is_list_item(out)
        ):
            out.append("\n")

        para_buf.append(line)
        i += 1

    flush_paragraph(para_buf, out)
    text = "".join(out)
    return text.strip() + "\n"


def main() -> None:
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("resumen_source.txt")
    outp = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("resumen.md")
    lines = inp.read_text(encoding="utf-8").splitlines()
    md = convert(lines)
    outp.write_text(md, encoding="utf-8")
    print(f"Escrito {outp} ({len(md)} caracteres)")


if __name__ == "__main__":
    main()
