#!/usr/bin/env python
"""Заполнить блок RESULTS в README.md из файлов результатов (перезапускать после новых результатов).

Источники (все необязательны): checkpoints/best.pt (карточка модели), results/ml/compare_main_vs_xtalk.json,
results/delivery/notes.md (выводы, пишутся вручную по текущим цифрам), results/delivery/demo/summary.md,
results/delivery/selectivity.md, results/delivery/methods*/summary.md.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def section(title, body):
    return f"### {title}\n\n{body.strip()}\n"


def num(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}".replace(".", ",")


def model_card(path=ROOT / "checkpoints/best.pt"):
    if not path.exists():
        return None
    import torch
    real = path.resolve()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, loss, a = ck.get("model_cfg", {}), ck.get("loss_cfg", {}), ck.get("args", {})
    rows = [("веса", f"`checkpoints/best.pt` → `{real.relative_to(ROOT)}` (`{a.get('run', '?')}`), шаг {ck.get('step', '?')}"),
            ("параметров", "15,4 млн"),
            ("архитектура", f"U-Net в плоскости ДОЭ (`kind={cfg.get('kind')}`, ширина {cfg.get('width')}, "
                            f"внимание в узком месте), физический вход (`inputs={cfg.get('inputs')}`), "
                            f"**{cfg.get('unroll', 0)} обучаемых стадий уточнения** (learned GS)"),
            ("функция потерь", f"`{loss}` (xtalk — штраф за перекрёстные помехи)"),
            ("обучение", "цепочка: `amp_main` (Colab L4, 30 000 шагов, 1 ч 47 мин, QAT) → `amp_xtalk` (дообучение "
                         "со штрафом за помехи, xtalk = 1) → **`amp_deep6`**: `--init amp_xtalk --add-stages 4` "
                         "(к 2 стадиям добавлены 4 новые), 16 000 шагов на Google Colab L4, лучший чекпойнт по "
                         "Jsel (качество + селективность) — шаг 13 000"),
            ("режимы и время (GPU Mac, MPS)", "основной — сеть + 50 шагов уточнения (`--method nn_refine`, по "
                                             "умолчанию), 115 мс; быстрый — один проход сети (`--method nn`), 24 мс")]
    return "| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows) + "\n"


DEGENERATE_INTRO = """Первая модель (`amp_main`) имела хорошие средние КК и ИСС, но почти во всех плоскостях рисовала
**одну и ту же смесь всех четырёх изображений**. Средние метрики такую смесь почти не наказывают:
смесь умеренно похожа на каждую цель, и КК ≈ 0,5 получается «бесплатно». Глубинного разделения при этом
почти нет. Мы нашли это по матрице КК(цель i, восстановление в плоскости j). У вырожденного решения
она почти плоская: диагональ ≈ всё остальное.

Как измеряем (`doe.metrics.crosstalk`; то же считает `evaluate.py`): **селективность** = среднее
диагонали − среднее вне диагонали. **«Своя лучшая»** — в скольких из 4 плоскостей восстановление
больше всего похоже именно на свою цель (4/4 — идеальное разделение, ≈1/4 — случайное).

Что сделали: добавили в функцию потерь штраф за перекрёстные помехи (`doe/losses.py`,
`crosstalk_penalty`, вес `xtalk = 1`) и дообучили сеть (`amp_xtalk`). При уточнении
(`predict_refined`) действует тот же штраф.

Чистая сеть, test-256 (`results/ml/compare_main_vs_xtalk.json`):
"""

DEGENERATE_OUTRO = """
Цена штрафа — минус 0,026 по среднему КК и минус 0,034 по ИСС. Зато селективность выросла в 1,7 раза
(0,110 → 0,190), и своя цель стала лучшей в 2,9 плоскостях из 4 вместо 2,1. Мы сознательно меняем
немного средних метрик на настоящее разделение по глубине: именно его и должен давать ДОЭ.

Проба на val-32 (`results/ml/hybrid_probe.log`, пакетный режим): ГС-50 даёт селективность 0,225 при КК 0,507,
сеть + 50 шагов уточнения со штрафом — 0,302 при КК 0,496. Поэтому основным режимом сдачи выбрана
«сеть + 50 шагов». Чистая сеть остаётся быстрым режимом.
"""


DEEP_INTRO = """
**Этап 3 — глубокая модель (`amp_deep6`).** У `amp_xtalk` выход сети был хорошей самостоятельной
голограммой, но **плохой стартовой точкой для уточнения**. При одинаковых 50 шагах Adam старт от GS-50
давал КК 0,522 / ИСС 0,386 / селективность 0,303, а старт от сети — 0,502 / 0,356 / 0,247 (test-96,
`results/ml/bestofk_probe.log`, BACKLOG B13). В «лучшем из двух» сеть выбиралась лишь в 18 % случаев.
Поэтому мы добавили 4 обучаемые стадии уточнения (learned GS, всего 6) и дообучили модель от `amp_xtalk`.

Результат: **старт от сети теперь не хуже старта от GS-50.** «Глубокая сеть + 50» против «GS-50 + 50»
практически вровень: КК −0,005, ИСС глоб. −0,007, худшая плоскость и ИСС gauss в пределах ДИ, а
селективность выше на +0,025. Против GS-200 основной режим проигрывает 0,015 по среднему КК, но выигрывает
по худшей плоскости (+0,179), ИСС глоб. (+0,015), ИСС gauss (+0,047) и селективности (+0,131).
Чистая глубокая сеть разделяет плоскости намного лучше прежней (0,292 против 0,190, своя лучшая 3,73
против 2,87 из 4). Цена — оконный ИСС у чистой сети ниже (0,192 против 0,259), поэтому основной режим — с
уточнением.

TEST-256, протокол экспертов, ДИ 95 % (`results/final/compare_deep.md`):
"""

DEEP_OUTRO = ""


DEEP_HEADER = "## Глубокая модель (6 стадий уточнения)"


def deep_section():
    """Hand-level README section (outside the auto block) built from results/final/compare_deep.md."""
    deep = ROOT / "results/final/compare_deep.md"
    if not deep.exists():
        return None
    body = deep.read_text().split("\n", 2)[2]
    body = body.replace("## Парные разности", "**Парные разности (ДИ 95 %):**")
    return (DEEP_HEADER + "\n" + DEEP_INTRO + "\n" + body.strip() + "\n" + DEEP_OUTRO + "\n")


def sync_manual_sections(text):
    """Replace the hand-inserted README sections with the current results/final/*.md (headers demoted)."""
    sec = deep_section()
    if sec:
        a = text.find(DEEP_HEADER)
        if a >= 0:   # replace the existing section up to the next "## " header
            m = re.compile(r"^## ", re.M).search(text, a + 3)
            text = text[:a] + sec + "\n" + text[m.start():]
        else:        # first time: insert right before the reliability section
            b = text.index("## Надёжность выводов")
            text = text[:b] + sec + "\n" + text[b:]
    for header, src in (("## Надёжность выводов", "results/final/robustness.md"),
                        ("## Сравнение с решением другой команды", "results/final/compare_codex.md")):
        f = ROOT / src
        if not f.exists() or header not in text:
            continue
        body = f.read_text().split("\n", 1)[1].strip()          # drop the file's own "# ..." title
        body = re.sub(r"^## ", "### ", body, flags=re.M)
        start = text.index(header)
        title_end = text.index("\n", start)
        nxt = start + 1
        # end of section = next "## " header that is not one of the old sub-sections we are replacing
        m = re.compile(r"^## (?!Парные разности|Граничные случаи|Выводы)", re.M).search(text, title_end)
        end = m.start() if m else len(text)
        text = text[:title_end + 1] + "\n" + f"Источник: `{src}`.\n\n" + body + "\n\n" + text[end:]
    return text


def degenerate_section():
    p = ROOT / "results/ml/compare_main_vs_xtalk.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    lines = [DEGENERATE_INTRO,
             "| модель | КК ср. / худш. | ИСС глоб. | ИСС gauss | селективность | своя лучшая |",
             "|---|---|---|---|---|---|"]
    for k, name in (("test/amp_main", "amp_main (без штрафа)"), ("test/amp_xtalk", "**amp_xtalk (со штрафом; прежняя основная)**")):
        if k in d:
            r = d[k]
            lines.append(f"| {name} | {num(r['pcc_mean'])} / {num(r['pcc_worst'])} | {num(r['ssim_global_mean'])} | "
                         f"{num(r['ssim_gauss_mean'])} | {num(r['selectivity'])} | {num(r['own_best'], 2)}/4 |")
    lines.append(DEGENERATE_OUTRO)
    return "\n".join(lines)


def main():
    parts = []
    card = model_card()
    if card:
        parts.append(section("Модель", card))
    dg = degenerate_section()
    if dg:
        parts.append(section("Вырожденное решение и штраф за помехи", dg))
    notes = ROOT / "results/delivery/notes.md"
    if notes.exists():
        parts.append(section("Выводы по текущим результатам", notes.read_text()))
    for d, title in (("methods", "Сравнение методов на 12 демо-наборах"), ("methods_test", "Сравнение методов на test-32")):
        f = ROOT / "results/delivery" / d / "summary.md"
        if f.exists():
            parts.append(section(f"{title} (`results/delivery/{d}/summary.md`), все методы на одном устройстве",
                                 f.read_text()))
    demo = ROOT / "results/delivery/demo/summary.md"
    if demo.exists():
        parts.append(section("Демо-наборы: сеть + 50 (основной режим) и сеть (`results/delivery/demo/`)",
                             demo.read_text()))
    sel = ROOT / "results/delivery/selectivity.md"
    if sel.exists():
        body = sel.read_text().split("\n", 2)[2]            # drop the "# ..." header line
        parts.append(section("Селективность по глубине (`results/delivery/selectivity.md`)", body))
    links = [p for p in ("results/ml/summary.md", "results/baselines/summary.md") if (ROOT / p).exists()]
    if links:
        parts.append(section("Подробности", "\n".join(f"* `{p}`" for p in links)))
    if not parts:
        parts = ["*Пока нет результатов: запустите `scripts/make_delivery.sh`.*\n"]
    readme = ROOT / "README.md"
    text = sync_manual_sections(readme.read_text())
    new = re.sub(r"(<!-- RESULTS:BEGIN -->\n).*?(<!-- RESULTS:END -->)",
                 lambda m: m.group(1) + "\n".join(parts) + m.group(2), text, flags=re.S)
    readme.write_text(new)
    print(f"README: {len(parts)} раздел(ов) в блоке результатов")


if __name__ == "__main__":
    main()
