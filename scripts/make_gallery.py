"""Собирает results/final/index.html — одностраничную галерею результатов для просмотра в браузере.

Только читает готовые артефакты (results/, data/demo/); ничего не пересчитывает.
Открывать через HTTP-сервер из корня проекта: python -m http.server 8765 → /results/final/index.html
"""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "final" / "index.html"
HEROES = ["photo_text", "halftone_photos", "rocket_smoke"]
DEMOS = ["photo_text", "halftone_photos", "shapes", "dark_objects", "rocket_smoke", "words",
         "cyrillic_NIYAU", "letters_MIFI", "digits_1234", "mixed", "photo_like", "stress"]
MODES = [("nn", "сеть"), ("nn+adam20", "сеть + 20"), ("nn+adam50", "сеть + 50"), ("nn+adam100", "сеть + 100"),
         ("backprop", "обратное распр."), ("backprop+adam100", "Adam со старта обр. распр., 100")]


def rel(p: Path) -> str:
    return "/" + p.relative_to(ROOT).as_posix()


def img(p: Path, w: str = "100%", caption: str = "") -> str:
    if not p.exists():
        return f'<p class="miss">нет файла {html.escape(rel(p))}</p>'
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return f'<figure><a href="{rel(p)}" target="_blank"><img src="{rel(p)}" style="width:{w}"></a>{cap}</figure>'


def eval_table(split: str) -> str:
    p = ROOT / "results" / "ml" / f"eval_amp_xtalk_{split}.json"
    if not p.exists():
        return ""
    m = json.loads(p.read_text())["metrics"]
    rows = []
    for key, name in MODES:
        if key not in m:
            continue
        r = m[key]
        rows.append(f"<tr><td>{name}</td><td>{r['pcc_mean']:.3f}</td><td>{r['pcc_worst']:.3f}</td>"
                    f"<td>{r['ssim_global_mean']:.3f}</td><td>{r['ssim_skimage_mean']:.3f}</td>"
                    f"<td>{r['ssim_gauss_mean']:.3f}</td><td>{r.get('selectivity', float('nan')):.3f}</td>"
                    f"<td>{r.get('own_best', float('nan')):.2f}/4</td></tr>")
    return ("<table><tr><th>режим</th><th>КК ср.</th><th>КК худш.</th><th>ИСС глоб.</th>"
            "<th>ИСС skimage</th><th>ИСС gauss</th><th>селективность</th><th>своя лучшая</th></tr>" + "".join(rows) + "</table>")


def md_block(p: Path) -> str:
    """Таблицы из готовых summary.md — как предварительно отформатированный текст."""
    if not p.exists():
        return ""
    lines = [l for l in p.read_text().splitlines() if l.startswith("|")]
    return f"<pre>{html.escape(chr(10).join(lines))}</pre>"


def main():
    d = ROOT / "results" / "delivery"
    timing = ROOT / "results" / "final" / "timing_mac.json"
    t = json.loads(timing.read_text()) if timing.exists() else {}
    parts = [f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Нейросетевой синтез ДОЭ — результаты</title>
<style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#1d1d1f}}
h1{{font-size:26px}} h2{{margin-top:40px;border-bottom:1px solid #ddd;padding-bottom:4px}}
table{{border-collapse:collapse;margin:8px 0}} td,th{{border:1px solid #ccc;padding:4px 10px;text-align:right}}
td:first-child,th:first-child{{text-align:left}} figure{{margin:8px 0}} figcaption{{color:#666;font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}} .note{{color:#555}} .miss{{color:#b00}}
pre{{background:#f6f6f6;padding:8px;overflow-x:auto;font-size:12px}}
img{{image-rendering:pixelated}}
</style>
<h1>Нейросетевой синтез фазового ДОЭ: 4 изображения на 4 расстояниях</h1>
<p class="note">Протокол экспертов: сравниваются амплитуды |K|, каждое изображение нормировано на свой максимум = 1.
Все цифры — по сохранённому 8-битному фазовому PNG. λ = 532 нм, пиксель 10 мкм, z = 24,1 / 25,3 / 26,5 / 27,7 мм.</p>
<p>Время на одну голограмму (Mac M3 Pro): сеть — <b>{t.get('nn_mps_ms', '?')} мс</b> на GPU (MPS) /
{t.get('nn_cpu_ms', '?')} мс на CPU; сеть + 100 шагов уточнения — {t.get('nn_refine100_mps_ms', '?')} мс на GPU.</p>

<h2>1. Итог на TEST (256 четвёрок, невиданные классы объектов и шрифты) — модель amp_xtalk, основной режим «сеть + 50»</h2>
{eval_table('test')}
<h3>VAL (256)</h3>{eval_table('val')}

<h2>1а. Надёжность: доверительные интервалы, парные сравнения, граничные случаи</h2>
{md_block(ROOT / 'results' / 'final' / 'robustness.md')}

<h2>1б. Сравнение с решением команды Codex (их тест main60, один оценщик, протокол экспертов)</h2>
{md_block(ROOT / 'results' / 'final' / 'compare_codex.md')}

<h2>1в. Найденная проблема: смесь во всех плоскостях → штраф за помехи</h2>
<p class="note">Первая модель рисовала почти одинаковую смесь всех 4 картинок во всех плоскостях. Строки: цели / первая модель / модель со штрафом / GS.</p>
<div class="grid">
{img(ROOT / 'results' / 'final' / 'xtalk_compare_dark_objects.png', caption='dark_objects')}
{img(ROOT / 'results' / 'final' / 'xtalk_compare_shapes.png', caption='shapes')}
{img(ROOT / 'results' / 'final' / 'refine_compare_shapes.png', caption='shapes: новая сеть + 50 шагов против GS')}
{img(ROOT / 'results' / 'final' / 'refine_compare_dark_objects.png', caption='dark_objects: новая сеть + 50 шагов против GS')}
</div>

<h2>2. Скорость и качество (все методы на одном CPU)</h2>
{img(d / 'speed_quality.png', caption='12 демо-наборов')}
{img(d / 'speed_quality_test.png', caption='32 тестовые четвёрки')}

<h2>3. Скан фокуса: изображения проявляются в своих плоскостях</h2>
<div class="grid">
""" + "".join(img(d / f"focus_sweep_{h}.gif", caption=f"{h}: z от 22 до 30 мм") + img(d / f"crosstalk_{h}.png", caption=f"{h}: КК каждой цели в зависимости от z")
            for h in HEROES) + f"""
</div>
<h3>Селективность по глубине (все демо)</h3>{md_block(d / 'selectivity.md')}

<h2>4. Демо-наборы: цели и восстановления</h2>
<p class="note">Для каждого набора: сеть (один проход, быстрый режим) и сеть + 50 шагов уточнения (основной режим).</p>
"""]
    for name in DEMOS:
        parts.append(f"<h3>{name}</h3><div class='grid'>"
                     f"{img(d / 'demo' / name / 'nn' / 'comparison.png', caption='сеть, один проход')}"
                     f"{img(d / 'demo' / name / 'nn_refine50' / 'comparison.png', caption='сеть + 50 шагов (основной режим)')}</div>")
        parts.append(img(d / 'methods' / name / 'methods.png', caption='сравнение методов'))
    parts.append(f"""
<h2>5. Обучение (Colab L4, 30 000 шагов, 1 ч 47 мин)</h2>
{img(ROOT / 'results' / 'ml' / 'colab' / 'ml' / 'amp_main' / 'curves.png')}
<h2>6. Классические методы (потолок качества)</h2>
{img(ROOT / 'results' / 'baselines' / 'curves_v2.png', caption='качество от числа итераций GS и Adam')}
{md_block(ROOT / 'results' / 'baselines' / 'summary.md')}
<h2>7. Данные</h2>
{img(ROOT / 'data' / 'preview.png', caption='случайные обучающие примеры')}
</html>""")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
