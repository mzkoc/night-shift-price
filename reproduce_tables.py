"""
Recompute the D3 tables of the article from the exported best-known schedules.

    python reproduce_tables.py [v9_D3_bks.csv] [--verify]

Input: v9_D3_bks.csv (and, with --verify, v9_D3_bks_keys.json), written by
    python -u weaving_gp_v9.py --export
Output (stdout): the price of the night rule by factor level (Table 5), its decomposition
(Table 6), the day-capacity comparison (Table 7), the factor tests (Online Resource 1,
Table S4) and the electricity-bill differences.  The statistics (bootstrap intervals,
Friedman, Wilcoxon, Holm) are those of weaving_gp_v9.py, imported from it.
--verify regenerates the instances with the code and decodes every exported key vector,
checking that it gives the exported tardiness, drawing-in time and overtime.
"""
import csv, sys, json
import weaving_gp_v9 as G

FACTORS = [('C', ('2', '3', 'inf')), ('nf', ('4', '8')), ('lmax', ('200', '300', '500')), ('delta', ('1.0', '0.8'))]
LABEL = {'C': 'crews C', 'nf': 'families n_f', 'lmax': 'L_max', 'delta': 'tightness delta'}


def load(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    B = {}
    for r in rows:
        B[(r['instance'], r['scheme'], r['policy'])] = r
    return rows, B


def num(r, k):
    return float(r[k]) if r[k] != '' else float('nan')


def price_rows(rows, B, scheme, setup):
    """night rule minus night crew per instance (and the equal crew-hours split where present)."""
    out = []
    for r in rows:
        if r['scheme'] != scheme or r['policy'] != 'night rule' or r['setup'] != setup:
            continue
        c = B[(r['instance'], scheme, 'night crew')]; e = B.get((r['instance'], scheme, 'equal crew-hours'))
        T1, T0 = num(r, 'T_min') / 60, num(c, 'T_min') / 60
        row = dict(inst=r['instance'], meta={k: r[k] for k, _ in FACTORS} | {'seed': r['seed']},
                   dT=T1 - T0, T0=T0, dTp=100 * (T1 - T0) / T0 if T0 > 1e-9 else float('nan'),
                   dv=100 * (num(r, 'v') - num(c, 'v')), dS=(num(r, 'S_min') - num(c, 'S_min')) / 60,
                   O=num(r, 'O_min') / 60, pause=num(r, 'nPause'), night=num(c, 'nightMin_min') / 60,
                   bind=(num(r, 'v') - num(c, 'v')) > 1e-9, late1=num(r, 'nLate'), late0=num(c, 'nLate'),
                   no=num(r, 'orders'), dE=100 * (num(r, 'E_TL') - num(c, 'E_TL')) / num(c, 'E_TL'))
        if e is not None:
            Te = num(e, 'T_min') / 60
            row.update(red=T1 - Te, cap=Te - T0)
        out.append(row)
    return out


def ratio(rs):
    den = sum(r['T0'] for r in rs)
    return 100 * sum(r['dT'] for r in rs) / den if den > 1e-9 else float('nan')


def line(label, rs):
    g = lambda k: [r[k] for r in rs]
    late = f"{100*sum(g('late1'))/sum(g('no')):.0f}/{100*sum(g('late0'))/sum(g('no')):.0f}"
    print(f"  {label:<20}{len(rs):>4}{G.mean_(g('dv')):>7.2f}{G.fci(g('dT')):>22}{ratio(rs):>7.1f}"
          f"{G.median_(g('dTp')):>7.1f}{G.mean_(g('dS')):>8.1f}{late:>8}{G.mean_(g('O')):>6.1f}{G.mean_(g('pause')):>6.1f}"
          f"{100*sum(g('bind'))/len(rs):>6.0f}{G.mean_(g('night')):>8.1f}")


def factor_tests(rs, key):
    for fk, levels in FACTORS:
        oth = [k for k, _ in FACTORS if k != fk]; blk = {}
        for r in rs:
            b = tuple(r['meta'][k] for k in oth) + (r['meta']['seed'],)
            blk.setdefault(b, {})[r['meta'][fk]] = r[key]
        Bl = [[d[lv] for lv in levels] for d in blk.values() if all(lv in d for lv in levels)]
        if len(Bl) < 3:
            continue
        agg = G.median_ if key == 'dTp' else G.mean_
        means = " / ".join(f"{lv}: {agg([b[j] for b in Bl]):.1f}" for j, lv in enumerate(levels))
        if len(levels) == 2:
            print(f"    {key:<4} {fk:<6} {means}  Wilcoxon p={G.wilcoxon([b[0] for b in Bl], [b[1] for b in Bl]):.3f}")
        else:
            _, _, pv = G.friedman(Bl); pairs = [(a, c) for a in range(len(levels)) for c in range(a + 1, len(levels))]
            ad = G.holm([G.wilcoxon([b[a] for b in Bl], [b[c] for b in Bl]) for a, c in pairs])
            print(f"    {key:<4} {fk:<6} {means}  Friedman p={pv:.3f}; pairs (Holm) "
                  + ", ".join(f"{levels[a]}-{levels[c]} {q:.3f}" for (a, c), q in zip(pairs, ad)))


def verify(path_keys, rows):
    keys = json.load(open(path_keys, encoding='utf-8'))
    insts = {i['name']: i for i in G.d3_instances()}
    pol = {'night rule': True, 'night crew': False, 'equal crew-hours': 'EH', 'night rule C+1': 'R+1',
           'night rule C+2': 'R+2'}
    bad = 0
    for r in rows:
        inst = insts[r['instance']]; k = keys[r['instance']][r['scheme']][r['policy']]
        Z = G.evaluate_keys(k, inst, pol[r['policy']])
        if (Z['T'], Z['S'], Z['O']) != (float(r['T_min']), float(r['S_min']), float(r['O_min'])):
            bad += 1
    print(f"\n  verification: {len(rows) - bad}/{len(rows)} exported schedules reproduced exactly by the evaluator")


def main():
    if '--quick' in sys.argv:
        G._quick()                       # test set-up of weaving_gp_v9.py (for v9quick exports)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else 'v9_D3_bks.csv'
    rows, B = load(path)
    hdr = (f"  {'level':<20}{'n':>4}{'dv%':>7}{'dT h [95% CI]':>22}{'dT%':>7}{'med%':>7}{'dS h':>8}{'late%':>8}"
           f"{'O h':>6}{'pause':>6}{'bind':>6}{'night':>8}")
    for sn in ('Lex0', 'Lex10'):
        rs = price_rows(rows, B, sn, 'structured')
        print(f"\nTABLE 5  price of the night rule, {sn}, set F (rule minus crew; dT% ratio of means)\n{hdr}")
        for fk, levels in FACTORS:
            for lv in levels:
                sub = [r for r in rs if r['meta'][fk] == lv]
                if sub: line(f"{LABEL[fk]} {lv}", sub)
        line('all (F)', rs); line('general matrix (R)', price_rows(rows, B, sn, 'general'))
        dE = [r['dE'] for r in rs]
        print(f"  electricity bill, rule minus crew: mean {G.mean_(dE):.2f}%, max |.| {max(abs(x) for x in dE):.2f}%")
        print(f"\n  factor tests, {sn} (Online Resource 1, Table S4):")
        for key in (('dT', 'dTp', 'dS') if sn == 'Lex0' else ('dv', 'dT', 'dS')):
            factor_tests(rs, key)
    print("\nTABLE 6  decomposition, C = 3 (h; p: Wilcoxon against zero)")
    for sn in ('Lex0', 'Lex10'):
        for lab, setup in (('F, C=3', 'structured'), ('R', 'general')):
            rs = [r for r in price_rows(rows, B, sn, setup) if 'red' in r]
            red, cap = [r['red'] for r in rs], [r['cap'] for r in rs]
            print(f"  {sn:<6}{lab:<8}{len(rs):>4}  total {G.fci([r['dT'] for r in rs])}  redistribution {G.fci(red)} "
                  f"p={G.p0(red):.3f}  capacity {G.fci(cap)} p={G.p0(cap):.3f}")
    print("\nTABLE 7  day capacity, Lex0: night rule with C crews minus night crew with C = 3 (h; count not worse)")
    for lab, setup in (('F, C=3', 'structured'), ('R', 'general')):
        S = [r for r in rows if r['scheme'] == 'Lex0' and r['policy'] == 'night crew' and r['setup'] == setup and r['C'] == '3']
        cols = {}
        for c in S:
            T0 = num(c, 'T_min') / 60
            for key, p in (('C=3', 'night rule'), ('C=4', 'night rule C+1'), ('C=5', 'night rule C+2')):
                cols.setdefault(key, []).append(num(B[(c['instance'], 'Lex0', p)], 'T_min') / 60 - T0)
            if setup == 'structured':
                tw = next(r for r in rows if r['scheme'] == 'Lex0' and r['policy'] == 'night rule' and r['setup'] == setup
                          and r['C'] == 'inf' and all(r[k] == c[k] for k in ('nf', 'lmax', 'delta', 'seed')))
                cols.setdefault('C=inf', []).append(num(tw, 'T_min') / 60 - T0)
        print(f"  {lab:<8}{len(S):>4}  " + "   ".join(f"{k}: {G.fci(v)} ({sum(1 for x in v if x <= 1e-6)})" for k, v in cols.items()))
    if '--verify' in sys.argv:
        verify(path.replace('_bks.csv', '_bks_keys.json'), rows)


if __name__ == '__main__':
    main()
