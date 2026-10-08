# -*- coding: utf-8 -*-
"""Таблицы S1–S4: значения по отдельным запускам (вынесены из статьи по рекомендации редакции).

Пишет:
  docs/tables_S1-S4.md   — таблицы для просмотра на GitHub;
  results/tables/S*.csv  — те же значения в машиночитаемом виде (разделитель «;», точка в числах).
Все значения читаются из results/; ничего не задаётся вручную.
"""
import csv, os
import numpy as np
import paper_data as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DOCS = os.path.join(ROOT, 'docs')
TAB = os.path.join(ROOT, 'results', 'tables')
os.makedirs(DOCS, exist_ok=True); os.makedirs(TAB, exist_ok=True)
MINUS = '−'


def f3(x): return f'{x:.3f}'.replace('.', ',').replace('-', MINUS)
def sg(x): return f'{x:+.3f}'.replace('.', ',').replace('-', MINUS)
def pct(x): return f'{x * 100:.1f}'.replace('.', ',')
def sig(v): return '1' if v == 1.0 else str(v).replace('.', ',')


def md_table(header, rows, groups=None):
    out = []
    if groups:
        out.append('| ' + ' | '.join(groups) + ' |')
        out.append('|' + '---|' * len(groups))
        out.append('| ' + ' | '.join(header) + ' |')
    else:
        out.append('| ' + ' | '.join(header) + ' |')
        out.append('|' + '---|' * len(header))
    out += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return '\n'.join(out)


def write_csv(name, header, rows):
    with open(os.path.join(TAB, name), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(header)
        w.writerows(rows)


md = ['# Таблицы S1–S4. Значения по отдельным запускам', '',
      'Дополнительные материалы к статье И.А. Кистина, Ш.М. Исаева «Декомпозиция предсказательной '
      'неопределённости байесовской свёрточной сети для обнаружения данных вне обучающего '
      'распределения» (Научно-технический вестник информационных технологий, механики и оптики). '
      'Таблицы построены скриптом `src/supp_tables.py` из файлов `results/`; те же значения в '
      'машиночитаемом виде — в `results/tables/`. Номер запуска совпадает с начальным значением '
      'генератора псевдослучайных чисел. Δ — разность AUROC эпистемической составляющей и полной '
      'энтропии.', '']

# ------------------------------------------------------------------ S1: эксперимент A
R = P.R
rows, crows = [], []
for r in R:
    d = r['delta_epi_vs_total']
    rows.append([str(r['seed']), f3(r['auroc']['epistemic']), f3(r['auroc']['total']),
                 f3(r['auroc']['aleatoric']), f3(r['auroc']['msp']),
                 f'{sg(d[0])} [{sg(d[1])}; {sg(d[2])}]'])
    crows.append([r['seed']] + [f"{r['auroc'][k]:.6f}" for k in ('epistemic', 'total', 'aleatoric', 'msp')]
                 + [f'{d[0]:.6f}', f'{d[1]:.6f}', f'{d[2]:.6f}'])
md += ['## Таблица S1. Эксперимент A: AUROC и парный бутстреп разности по запускам', '',
       md_table(['Запуск', 'Эпистем.', 'Полная', 'Алеатор.', 'max softmax',
                 'Δ, 95 % ДИ'], rows), '',
       'FashionMNIST → MNIST, полные тестовые наборы 10 000 / 10 000, исходный режим обучения '
       '(5 эпох, σ = 1); парный бутстреп, 2000 повторов.', '']
write_csv('S1_experiment_A.csv', ['run', 'auroc_epistemic', 'auroc_total', 'auroc_aleatoric',
                                  'auroc_msp', 'delta_epi_total', 'delta_ci_low', 'delta_ci_high'], crows)

# ------------------------------------------------------------------ S2: повороты по углам
ub, ut = P.shift_units('rotation', 'base'), P.shift_units('rotation', 'trained')
sb, st = sorted(ub), sorted(ut)
rows, crows = [], []
for j, ang in enumerate(P.ANG):
    def cell(u, s):
        x = u[s][j]['d']
        return sg(x['point']) + ('°' if P.cls(x) == '0' else '')
    rows.append([f'{ang}°'] + [cell(ub, s) for s in sb] + [cell(ut, s) for s in st])
    for reg, u, ss in (('base', ub, sb), ('trained', ut, st)):
        for s in ss:
            x = u[s][j]['d']
            crows.append([reg, ang, s, f"{x['point']:.6f}", f"{x['lo']:.6f}", f"{x['hi']:.6f}"])
md += ['## Таблица S2. Повороты: Δ по углам и запускам', '',
       md_table(['Угол'] + [f'и{s}' for s in sb] + [f'л{s}' for s in st], rows), '',
       '«и» — исходный режим (5 эпох, σ = 1), «л» — лучше обученный (20 эпох, σ = 0,1), цифра — '
       'номер запуска. Знак «°» — 95 % ДИ парного бутстрепа (1000 повторов) содержит ноль; в '
       'остальных ячейках разность значима. Подвыборка 4000 / 4000 объектов.', '']
write_csv('S2_rotations_by_angle.csv', ['regime', 'angle_deg', 'run', 'delta_epi_total',
                                        'delta_ci_low', 'delta_ci_high'], crows)

# ------------------------------------------------------------------ S3: виды сдвига по запускам
SHIFTS = [('mnist', 'MNIST'), ('kmnist', 'KMNIST'), ('classes', 'Отложенные классы'),
          ('rotation', 'Повороты'), ('noise', 'Гауссов шум')]
rows, crows = [], []
for reg, lab in (('base', 'исходный'), ('trained', 'лучше обученный')):
    for s in sorted(P.shift_units('mnist', reg)):
        row, crow = [lab, str(s)], [reg, s]
        for key, _ in SHIFTS:
            u = P.shift_units(key, reg).get(s)
            v = float(np.mean([x['d']['point'] for x in u])) if u else None
            row.append(sg(v) if v is not None else '—')
            crow.append(f'{v:.6f}' if v is not None else '')
        rows.append(row); crows.append(crow)
md += ['## Таблица S3. Δ по видам сдвига и запускам', '',
       md_table(['Режим', 'Запуск'] + [n for _, n in SHIFTS], rows), '',
       'Для поворотов и шума — среднее по уровням сдвига (8 углов, 4 уровня СКО шума). '
       'Межнаборные и ковариатные сдвиги — подвыборки 4000 / 4000, отложенные классы — 8000 / 2000.', '']
write_csv('S3_shift_types_by_run.csv', ['regime', 'run'] + [k for k, _ in SHIFTS], crows)

# ------------------------------------------------------------------ S4: режимы обучения
rows, crows = [], []
for r in P.regime_table():
    rows.append([f"{r['key'][0]} эпох, σ = {sig(r['key'][1])}", str(r['n']), pct(r['acc'][0]),
                 f"{sg(r['d_mnist'][0])} ± {f3(r['d_mnist'][1])}",
                 f"{sg(r['d_rot'][0])} ± {f3(r['d_rot'][1])}",
                 f"{r['rot_pos']}/{r['rot_neg']}/{r['rot_cells']}"])
    for j, s in enumerate(r['seeds']):
        crows.append([r['key'][0], r['key'][1], s, f"{r['acc_seeds'][j]:.6f}",
                      f"{r['d_mnist_seeds'][j]:.6f}", f"{r['d_rot_seeds'][j]:.6f}"])
md += ['## Таблица S4. Режимы обучения: точность и Δ', '',
       md_table(['Режим', 'Запусков', 'Точность, %', 'Δ, MNIST', 'Δ, повороты', '+/−/n, повороты'],
                rows), '',
       'Точность — по первым 4000 тестовым объектам FashionMNIST; Δ — среднее ± СКО по запускам '
       '(для поворотов — среднее по восьми углам); «+/−/n» — число комбинаций «угол × запуск», в '
       'которых 95 % ДИ парного бутстрепа разности лежит выше / ниже нуля, из общего числа n.', '']
write_csv('S4_training_regimes.csv', ['epochs', 'sigma', 'run', 'accuracy_first4000',
                                      'delta_mnist', 'delta_rotation_mean'], crows)

open(os.path.join(DOCS, 'tables_S1-S4.md'), 'w', encoding='utf-8').write('\n'.join(md))
print('docs/tables_S1-S4.md и results/tables/S1–S4 записаны')
