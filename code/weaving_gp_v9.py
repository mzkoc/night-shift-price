"""
================================================================================
 SUSTAINABLE WEAVING SCHEDULING, MODEL v9 (Oct 2026)  --  single file (local or Colab)
================================================================================
 Data: drawing-in family -> design -> order -> weaving beams.
   An order of Q metres of one design is wound, after sizing, on weaving beams of
   at most L_MAX metres (2000 m, L_MAX 300 -> 300x5 + 250x2).  All beams of an order
   share design, drawing-in family and due date; an order is complete when its last
   beam is woven.  Weaving speed of a design (m/h) = 60 * RPM * EFF / (100 * picks/cm).
 Changeover between consecutive beams on a loom:
   same family      -> knotting, 0 min, no crew
   family f -> g    -> drawing-in r_f + T_g (removal of the old harness + drawing-in),
                       performed by one of C drawing-in crews.
 Policies (crew calendars):
   'night rule'  (True)  crews work 08-24 (day and evening shifts), no drawing-in crew at
                         night.  A drawing-in not finished at 24:00 is either finished in
                         OVERTIME (at most OT_MAX min after midnight) or PAUSED until 08:00.
   'night crew'  (False) counterfactual: the C crews work around the clock (+50% crew-hours).
   'equal crew-hours' ('EH') counterfactual with the crew-hours of the night rule spread
                         over three shifts: 2C/3 crews around the clock (C = 3 -> 2 crews).
   'R+1', 'R+2'          night rule with one or two additional day/evening crews (C = 3
                         instances, Lex0): how much day capacity buys back the price.
   Price of the night rule = T(rule) - T(crew) = [T(rule) - T(EH)]  redistribution part
                                                + [T(EH) - T(crew)]  capacity part.
 Crews: at most C drawing-ins at a time (C = 'inf': no limit); first come, first served;
   crew idle time is not modelled (crews also operate looms).
 Looms: identical, never idled on purpose: a loom waits only for its warp beam, for a
   crew, or for the morning.  At time 0 (Monday 08:00) loom k is still weaving its
   current beam until a_k and carries family f0_k.
 Objectives (minimised):
   T  total tardiness of orders          (min)  delivery
   S  drawing-in crew time               (min)  changeover labour
   O  overtime drawing-in after midnight (min)  social (night work of the evening crew)
 Reported indicators: ToU electricity cost E (TL) and carbon of weaving, late orders,
   lateness of late orders, pauses, waiting, makespan, night drawing-in (00-08) of the
   counterfactual policies, minimised in a third lexicographic stage (night_min).
 Goal programming:
   Chebyshev (minmax) GP with satisficing goals G_i = z*_i + GAMMA * R_i, weights per
   profile, efficiency-restoring second stage;
   lexicographic GP: priority 1 tardiness within a tolerance tau of its ideal (tau = 0:
   goal 0, i.e. tardiness itself is minimised, so that a heuristic ideal cannot stop the
   search), priority 2 Chebyshev GP on S, O (goal for O is zero overtime); strict priority
   in all methods (f = LEX_K * d_T[min] + min(A_2, LEX_P2CAP), see make_scalar).
   Priority 3 of every scheme (v9): EFFICIENCY RESTORATION.  Among schedules with the same
   achievement at priorities 1 and 2, the one with the least sum of normalised objectives
   sum_i (z_i - z*_i) / R_i (i = T, S, O) is chosen, so that every reported schedule is
   Pareto-efficient (no schedule found is better in one objective and not worse in the
   others).  CP-SAT: a final stage with the earlier achievements fixed; heuristics: the
   best-known schedule is the lexicographic minimum of (f, restoration) over the pool, and
   a restoration stage (GP-RKO minimising the restoration term subject to f <= f_best)
   follows the runs.  The least night drawing-in (00-08) of the counterfactual policies is
   then sought among schedules that tie at all three priorities.
 Methods (all scored by the same serial schedule-generation decoder):
   CP-SAT (one crew, small instances) | dispatching rules (EDD, ATCS, FAM, NEDD, NATCS)
   | GP-RKO (rule-seeded) | BRKGA (standard) | BRKGA-S (rule-seeded, irace-tuned)
   | SEED-LS (rule seeds + stochastic descent)
 Experiments: D1 exact validation and exact price (small, one crew), D2 scalability and
   algorithm comparison, D3 price map (factorial, 50 looms, three policies, bootstrap
   intervals, factor tests), D5 convergence of the price estimate, D4 case study,
   D6 goal-setting bias: the D3 protocol on the D5 subset with the tardiness goal pinned at
   the payoff-table ideal (G_T = z*_T, no ideal update), against D3 on the same instances.
 Local:  python -m pip install ortools matplotlib
         python -u weaving_gp_v9.py --quick                     (10-15 minute test)
         python -u weaving_gp_v9.py --only=D3,D6,D1,D4 --resume (run F: D3 continues from the
              v8 checkpoint v8_D3_ckpt.pkl (runs, polishing) with the restoration and night
              stages; then D6, D1 and D4 from scratch; about 1.5-2 h)
         python -u weaving_gp_v9.py --only=D3                   (D3 from scratch, about 6 h)
         outputs carry the selection, e.g. v9_D3_D6_D1_D4_results.csv
         options: --workers=18 --cp-workers=16 --resume (continue an interrupted D3 run)
                  --v8ckpt=v8_D3_ckpt.pkl (v8 checkpoint used by --resume when no v9 one exists)
 Tuning: Rscript irace_tune.R        (GP-RKO:  python weaving_gp_v9.py --irace ...)
         Rscript irace_tune_brkga.R  (BRKGA-S: python weaving_gp_v9.py --irace-brkga ...;
                                      writes brkga_s_cfg.json, read at start-up)
================================================================================
"""
import random, math, time, bisect, sys, os, heapq
from ortools.sat.python import cp_model

# =============================================================== 1. PARAMETERS
# ---- weaving ----
RPM, EFF   = 600, 0.90            # rapier loom speed (picks/min) and efficiency
DENS_RANGE = (20, 40)             # weft density of a design (picks/cm) -> 8.1 to 16.2 m/h
L_MAX, L_STEP = 300, 50           # weaving-beam capacity (m) and length step of the split rule
P_LOOM     = 5.0                  # loom power while weaving (kW); 0 while waiting or in changeover
# ---- day clock: t = 0 is 08:00 of day 1 (Monday); minutes ----
DAY = 1440
CREW_END = 960                    # crews work in phase [0, 960) = 08:00-24:00
OT_MAX   = 180                    # max overtime after midnight (min): 11 h a day (8 h shift + 3 h)
#  (start, end, ToU, price TL/kWh, EF kgCO2/kWh, shift)
#  prices: EPDK tariff valid from 4 April 2026, distribution-system users, medium voltage (OG),
#  single-term, industry: three-time energy charge (295.6124 / 495.3736 / 134.3850 kr/kWh)
#  plus the time-invariant distribution charge (118.2457 kr/kWh), taxes excluded.
EPDK_DIST = 1.182457
SEGMENTS_TOU = [(   0,  480, 'Day',   2.956124 + EPDK_DIST, 0.38, 'D'),
                ( 480,  540, 'Day',   2.956124 + EPDK_DIST, 0.38, 'E'),
                ( 540,  840, 'Peak',  4.953736 + EPDK_DIST, 0.45, 'E'),
                ( 840,  960, 'Night', 1.343850 + EPDK_DIST, 0.46, 'E'),
                ( 960, 1320, 'Night', 1.343850 + EPDK_DIST, 0.46, 'N'),
                (1320, 1440, 'Day',   2.956124 + EPDK_DIST, 0.38, 'N')]
FLAT_PRICE = 2.909687 + EPDK_DIST  # single-time energy charge plus distribution (tariff factor only)
# ---- changeovers ----
REMOVAL_FRAC = 0.10               # removal of the old harness r_f = REMOVAL_FRAC * T_f
DI_RANGE     = (180, 540)         # drawing-in times T_f spread evenly over 3 to 9 h
# ---- goal programming ----
OBJS  = ('T', 'S', 'O')            # optimised objectives; energy cost E and carbon are reported indicators
                                  # (set OBJS = ('T', 'S', 'E', 'O') to optimise E as well)
RHO   = 1e-3                      # augmentation coefficient
GAMMA = 0.10                      # satisficing goal: z*_i + GAMMA * range_i
RANGE_FLOOR = {'T': 60.0, 'S': 60.0, 'E': 10.0, 'O': 30.0}
PROFILES = {                      # Chebyshev GP weights (E used only if it is in OBJS)
    'Balanced': {'T': 1.0, 'S': 1.0, 'E': 1.0, 'O': 1.0},
    'Delivery': {'T': 1.0, 'S': 0.3, 'E': 0.3, 'O': 0.3},
    'Labour':   {'T': 0.3, 'S': 1.0, 'E': 0.3, 'O': 1.0},
    'NightSafe': {'T': 0.3, 'S': 0.3, 'E': 0.3, 'O': 1.0},
}
P2 = tuple(i for i in OBJS if i != 'T')   # objectives of priority 2 in the lexicographic GP
LEX_TAUS = (0.0, 0.05, 0.10, 0.20)   # tardiness tolerance of the lexicographic GP
# strict lexicographic priority in the heuristics:  f = LEX_K * d_T + min(A_2, LEX_P2CAP),
# d_T = max(0, T - G_T) in minutes.  T is integer and G_T lies on a grid of 0.05 min (tau is a
# multiple of 0.05 and z*_T an integer), so two different d_T differ by at least 0.05 min,
# i.e. by 0.05 * LEX_K = 500 > LEX_P2CAP - (-1): priority 1 always dominates priority 2.
LEX_K, LEX_P2CAP = 1e4, 400.0
LEX_LEGACY = 1e3                  # v6 weighting f = 1e3 * p1 + p2 (sensitivity check in D2)
# ties: two schedules tie at priorities 1-2 when their scalars differ by at most a few ulp
# (equal objective vectors give bit-identical scalars); restoration values likewise
def ftol(f): return 1e-12 + 8.9e-16 * abs(f)
RESTORE_BUDGET = {'S': 5000, 'L': 50000}   # restoration stage (priority 3), GP-RKO evaluations

# ---- instance generator ----
GEN_TF, GEN_RDD = 0.1, 0.6        # tardiness factor and due-date range (Li et al. 2012 scheme);
                                  # tight due dates: about half (delta 1.0) to three quarters
                                  # (delta 0.8) of the orders late in the best delivery plans
REL_SHARE, REL_MAX = 0.6, 2880    # share of orders released at 0; others U[0, 48 h]
AVAIL_MAX = 2160                  # looms finish their current beam within 36 h
CARRY = 0.75                      # share of looms whose current family follows the order book
CLASSES = {   # looms, crews, orders, order metres, families, designs, beams accepted
    'S': dict(m=(3, 4),   C=1, orders=(3, 4),   Q=(500, 1000),  fam=(2, 3), des=(3, 4),   n=(6, 10)),
    'M': dict(m=(10, 16), C=1, orders=(6, 10),  Q=(1000, 3000), fam=(4, 4), des=(6, 8),   n=(30, 70)),
    'L': dict(m=(40, 60), C=(2, 3), orders=(15, 25), Q=(1000, 4000), fam=(4, 8), des=(10, 16), n=(120, 200)),
    'F': dict(m=(50, 50), C=(2, 3), orders=(15, 25), Q=(1000, 4000), fam=(4, 8), des=(10, 16), n=(60, 400)),
}

# =============================================================== 2. INSTANCES
def split_order(Q, lmax=None, step=None):
    """beams of an order: k = ceil(Q/lmax); the excess k*lmax - Q is removed from the
       last beams in steps of `step` metres (2000, 300 -> 300x5 + 250x2)."""
    lmax = lmax or L_MAX; step = step or L_STEP
    k = -(-Q // lmax); lens = [lmax] * k; ex = k * lmax - Q; i = k - 1
    while ex > 0:
        d = min(step, ex); lens[i] -= d; ex -= d; i = i - 1 if i > 0 else k - 1
    return lens

def speed_mph(dens):
    return 60.0 * RPM * EFF / (100.0 * dens)

def make_setup(fam_T, mode='structured', seed=0):
    """setup[f][g] in minutes; f = None is not used (looms always carry a family)."""
    F = sorted(fam_T); S = {f: {} for f in F}
    if mode == 'structured':
        for f in F:
            for g in F:
                S[f][g] = 0 if f == g else int(round(REMOVAL_FRAC * fam_T[f] + fam_T[g]))
    else:                                                     # general asymmetric matrix
        rnd = random.Random(seed)
        for f in F:
            for g in F:
                S[f][g] = 0 if f == g else rnd.randrange(DI_RANGE[0], DI_RANGE[1] + 1, 30)
        for _ in range(len(F)):                               # repair the triangle inequality
            for f in F:
                for g in F:
                    for h in F:
                        if f != g and g != h and f != h and S[f][h] > S[f][g] + S[g][h]:
                            S[f][h] = S[f][g] + S[g][h]
    return S

def build_instance(name, m, C, orders, fam_T, designs, avail, f0, setup_mode='structured',
                   tariff='ToU', lmax=None, seed=0):
    """orders: list of dict(design, Q, due, rel)  (due, rel in minutes)."""
    beams, order_of = [], {}
    for o, od in enumerate(orders):
        fam, dens = designs[od['design']]
        for L in split_order(od['Q'], lmax):
            p = int(round(L / speed_mph(dens) * 60))
            order_of.setdefault(o, []).append(len(beams))
            beams.append(dict(order=o, design=od['design'], fam=fam, L=L, p=p, rel=od['rel']))
    setup = make_setup(fam_T, setup_mode, seed)
    work = sum(b['p'] for b in beams)
    inst = dict(name=name, m=m, C=C, n=len(beams), beams=beams, order_of=order_of,
                due={o: od['due'] for o, od in enumerate(orders)}, rel={o: od['rel'] for o, od in enumerate(orders)},
                orders=orders, fam_T=fam_T, designs=designs, setup=setup, avail=list(avail), f0=list(f0),
                tariff=tariff, work=work, pool=[], seed=seed)
    inst['ecurve'] = energy_curve(tariff)
    return inst

def gen_instance(cls, seed, name=None, delta=1.0, lmax=None, n_fam=None, C=None, m=None,
                 setup_mode='structured', tariff='ToU', max_tries=500):
    """random instance of a class; seeds are redrawn (deterministically) until the number
       of beams lies in the class range."""
    P = CLASSES[cls]
    for t in range(max_tries):
        rnd = random.Random(1000 * seed + t)
        mm = m or rnd.randint(*P['m'])
        if C == 'inf': C = mm                                   # no crew limit (one crew per loom)
        cc = C or (P['C'] if isinstance(P['C'], int) else max(P['C'][0], min(P['C'][1], round(mm / 20))))
        nf = n_fam or rnd.randint(*P['fam']); nd = max(nf, rnd.randint(*P['des'])); no = rnd.randint(*P['orders'])
        fam_T = {f: int(round((DI_RANGE[0] + (DI_RANGE[1] - DI_RANGE[0]) * f / max(nf - 1, 1)) / 30) * 30) for f in range(nf)}
        designs = {d: (d % nf, rnd.randint(*DENS_RANGE)) for d in range(nd)}
        orders = []
        for o in range(no):
            orders.append(dict(design=rnd.randrange(nd), Q=rnd.randrange(P['Q'][0], P['Q'][1] + 1, 250),
                               rel=0 if rnd.random() < REL_SHARE else int(rnd.uniform(0, REL_MAX) // 30 * 30)))
        nb = sum(len(split_order(od['Q'], lmax)) for od in orders)
        if not (P['n'][0] <= nb <= P['n'][1]): continue
        # loom state at time 0: busy until a_k; a share CARRY of the looms carries a family of
        # the order book (drawn in proportion to its weaving work), the others a random family
        avail = [int(rnd.uniform(0, AVAIL_MAX) // 15 * 15) for _ in range(mm)]
        dem = {}
        for od in orders:
            fdm, dens = designs[od['design']]; dem[fdm] = dem.get(fdm, 0) + od['Q'] / speed_mph(dens)
        fl = sorted(dem)
        f0 = [rnd.choices(fl, weights=[dem[x] for x in fl])[0] if rnd.random() < CARRY else rnd.randrange(nf)
              for _ in range(mm)]
        # due dates: Li et al. (2012) scheme on an estimated makespan
        work = sum(od['Q'] / speed_mph(designs[od['design']][1]) * 60 for od in orders)
        sbar = sum(fam_T.values()) / nf * (1 + REMOVAL_FRAC)
        Pest = sum(avail) / mm + (work + no * sbar) / mm
        for od in orders:
            pmin = min(split_order(od['Q'], lmax)) / speed_mph(designs[od['design']][1]) * 60
            d = rnd.uniform(Pest * (1 - GEN_TF - GEN_RDD / 2), Pest * (1 - GEN_TF + GEN_RDD / 2)) * delta
            od['due'] = int(max(d, od['rel'] + pmin))
        return build_instance(name or f"{cls}{seed}", mm, cc, orders, fam_T, designs, avail, f0,
                              setup_mode=setup_mode, tariff=tariff, lmax=lmax, seed=seed)
    raise RuntimeError("no instance in the beam range")

# =============================================================== 3. ENERGY CURVE
def energy_curve(tariff):
    """per-minute cost, carbon and night-weaving rate on each daily segment, with cumulative
       values, so that the cost of weaving over [a, b) is F(b) - F(a) in O(1)."""
    segs = []; cE = cC = cN = 0.0
    for (a, b, _, pr, ef, sh) in SEGMENTS_TOU:
        price = pr if tariff == 'ToU' else FLAT_PRICE
        ce, cc, cn = P_LOOM / 60.0 * price, P_LOOM / 60.0 * ef, 1.0 if sh == 'N' else 0.0
        segs.append((a, b, ce, cc, cn, cE, cC, cN))
        cE += ce * (b - a); cC += cc * (b - a); cN += cn * (b - a)
    return dict(segs=segs, starts=[s[0] for s in segs], day=(cE, cC, cN))

def _cum(ec, t):
    d, ph = divmod(t, DAY); s = ec['segs'][bisect.bisect_right(ec['starts'], ph) - 1]
    dE, dC, dN = ec['day']
    return (d * dE + s[5] + s[2] * (ph - s[0]), d * dC + s[6] + s[3] * (ph - s[0]), d * dN + s[7] + s[4] * (ph - s[0]))

def weave_acc(ec, a, b):
    e1, c1, n1 = _cum(ec, a); e2, c2, n2 = _cum(ec, b)
    return e2 - e1, c2 - c1, n2 - n1

def night_overlap(a, b):
    """minutes of [a, b) in the night shift 00-08 (phase [960, 1440))."""
    tot = 0; d = a // DAY
    while d * DAY < b:
        lo, hi = max(a, d * DAY + CREW_END), min(b, (d + 1) * DAY)
        if hi > lo: tot += hi - lo
        d += 1
    return tot

# =============================================================== 4. DECODER (single source of truth)
def ctime(x, rule):
    """earliest crew time >= x."""
    if not rule: return x
    d, ph = divmod(x, DAY)
    return x if ph < CREW_END else (d + 1) * DAY

def di_occupation(sig, s, ot_pref, rule):
    """crew occupation of a drawing-in of length s started at sig (crew time):
       returns (crew end, overtime min, paused?)."""
    if not rule: return sig + s, 0, False
    ph = sig % DAY; left = CREW_END - ph
    if s <= left: return sig + s, 0, False
    ov = s - left
    if ot_pref and ov <= OT_MAX: return sig + s, ov, False
    return sig + s + (DAY - CREW_END), 0, True                 # resumes at 08:00

def eh_crews(C):
    """crews of the equal crew-hours policy: the 2C crew-shifts a day of the night rule
       spread evenly over three shifts (C = 3 -> 2 crews around the clock); None if 2C/3 is
       not an integer."""
    return 2 * C // 3 if isinstance(C, int) and (2 * C) % 3 == 0 else None

def pol_crews(inst, pol):
    """(calendar with the night rule?, number of crews) of a policy:
       True night rule, False night crew, 'EH' equal crew-hours, 'R+k' night rule with k
       additional crews."""
    if pol == 'EH': return False, eh_crews(inst['C'])
    if isinstance(pol, str) and pol.startswith('R+'): return True, inst['C'] + int(pol[2:])
    return pol is True, inst['C']

class Sched:
    """serial schedule generation: beams are placed one at a time on given looms, in the
       order of a plan; the drawing-in jobs are served by the crews in that same order,
       each by the crew that can start it first (crew list scheduling)."""
    __slots__ = ('inst', 'rule', 'C', 'lt', 'lf', 'cf', 'rec')
    def __init__(self, inst, rule=True):
        self.inst = inst; self.rule, self.C = pol_crews(inst, rule)
        self.lt = list(inst['avail']); self.lf = list(inst['f0'])
        self.cf = [0] * self.C; self.rec = [None] * inst['n']

    def peek(self, b, k, ot_pref):
        """timing of beam b if appended to loom k now: (sig, crew, crew_end, ov, paused, setup, start)."""
        inst = self.inst; B = inst['beams'][b]; x0 = max(self.lt[k], B['rel'])
        s = inst['setup'][self.lf[k]][B['fam']]
        if s == 0: return (x0, -1, x0, 0, False, 0, x0)
        c = min(range(self.C), key=self.cf.__getitem__)
        sig = ctime(max(x0, self.cf[c]), self.rule)
        e, ov, pz = di_occupation(sig, s, ot_pref, self.rule)
        return (sig, c, e, ov, pz, s, e)

    def place(self, b, k, ot_pref, pk=None):
        sig, c, e, ov, pz, s, st = pk or self.peek(b, k, ot_pref)
        if c >= 0: self.cf[c] = e
        p = self.inst['beams'][b]['p']; ready = max(self.lt[k], self.inst['beams'][b]['rel'])
        self.rec[b] = (k, ready, sig, s, ov, pz, st, st + p, c)
        self.lt[k] = st + p; self.lf[k] = self.inst['beams'][b]['fam']
        return st + p

def score(inst, sch):
    """objectives of a complete schedule."""
    ec = inst['ecurve']; B = inst['beams']
    S = O = E = CO2 = NW = W = 0.0; nDI = nP = nOT = 0; NDI = 0; oc = {}; Cmax = 0
    for b, r in enumerate(sch.rec):
        k, ready, sig, s, ov, pz, st, en, _ = r
        if s > 0:
            nDI += 1; S += s; O += ov; nOT += ov > 0; nP += pz
            NDI += ov if sch.rule else night_overlap(sig, sig + s)  # drawing-in work done during 00-08
        W += st - s - ready - (DAY - CREW_END if pz else 0)          # waiting for a crew / the morning
        e, c, nn = weave_acc(ec, st, en); E += e; CO2 += c; NW += nn
        o = B[b]['order']; oc[o] = max(oc.get(o, 0), en); Cmax = max(Cmax, en)
    td = [max(0, oc[o] - inst['due'][o]) for o in oc]; late = [x for x in td if x > 0]; T = sum(td)
    return dict(T=T, S=S, E=E, O=O, nDI=nDI, nPause=nP, nOT=nOT, nightDI=NDI, wait=W,
                Cmax=Cmax, CO2=CO2, night_weave=NW / max(inst['work'], 1),
                nLate=len(late), meanLate=(sum(late) / len(late) if late else 0.0), maxLate=max(td) if td else 0.0)

def decode(keys, inst, rule=True):
    """keys = [assignment n | priority n | overtime preference n].
       Assignment and priority give each loom its sequence of beams.  The schedule is then
       simulated: knotting needs no crew; drawing-in jobs are served first come, first served
       (by the crew time at which the loom is ready; ties by priority), each by the crew that
       is free first.  A crew never waits while a loom needs it, and a loom waits only when
       all crews are busy, at night, or for its warp beam."""
    n, m, B, S = inst['n'], inst['m'], inst['beams'], inst['setup']
    seqs = [[] for _ in range(m)]
    for b in sorted(range(n), key=lambda b: keys[n + b]): seqs[min(int(keys[b] * m), m - 1)].append(b)
    sch = Sched(inst, rule); cal = sch.rule; ptr = [0] * m; heap = []
    def advance(k):
        sq = seqs[k]
        while ptr[k] < len(sq):
            b = sq[ptr[k]]
            if S[sch.lf[k]][B[b]['fam']] == 0:
                sch.place(b, k, 0); ptr[k] += 1
            else:
                heapq.heappush(heap, (ctime(max(sch.lt[k], B[b]['rel']), cal), keys[n + b], k)); return
    for k in range(m): advance(k)
    while heap:
        _, _, k = heapq.heappop(heap); b = seqs[k][ptr[k]]
        sch.place(b, k, (keys[2 * n + b] >= 0.5) if cal else 0); ptr[k] += 1; advance(k)
    return sch

def evaluate_keys(keys, inst, rule=True):
    return score(inst, decode(keys, inst, rule))

def keys_from(inst, assign, order, otp):
    """keys reproducing a schedule given by loom assignment, placement order and overtime choices."""
    n, m = inst['n'], inst['m']; k = [0.0] * (3 * n)
    for b in range(n): k[b] = (assign[b] + 0.5) / m; k[2 * n + b] = 0.75 if otp[b] else 0.25
    for r, b in enumerate(order): k[n + b] = (r + 0.5) / n
    return k

# =============================================================== 5. GOAL PROGRAMMING
def chebyshev_gp(Z, w, ide, rng, goals=None):
    """minmax GP: A = max_i w_i d_i^+ / R_i + RHO * sum_i w_i d_i^+ / R_i."""
    G = goals or {i: ide[i] + GAMMA * rng[i] for i in OBJS}
    dv = [w[i] * max(0.0, Z[i] - G[i]) / rng[i] for i in OBJS if w.get(i, 0) > 0 and i in G]
    return max(dv) + RHO * sum(dv)

def efficiency_term(Z, w, ide, rng):
    """second stage when all goals are met: weighted normalised distance to the ideal."""
    return sum(w[i] * (Z[i] - ide[i]) / rng[i] for i in OBJS if w.get(i, 0) > 0)

def restoration(Z, ide, rng):
    """priority 3 (efficiency restoration): sum of the normalised objectives T, S, O."""
    return sum((Z[i] - ide[i]) / rng[i] for i in OBJS)

def is_lex(kind): return kind in ('lex', 'lexP')

def lex_goals(ide, rng, tau, pinned=False):
    """goals of the lexicographic GP.  tau = 0: G_T = 0, so priority 1 minimises tardiness
       itself (with an exact ideal this is equivalent to G_T = z*_T; with a heuristic ideal
       it prevents the search from stopping at an overestimated ideal).
       pinned (D6 only): G_T = (1 + tau) z*_T also for tau = 0, the goal-setting of v7."""
    if tau <= 0 and not pinned: gT = 0.0
    else: gT = ide['T'] * (1 + tau) if ide['T'] > 0 else tau * rng['T']
    G = {i: ide[i] + GAMMA * rng[i] for i in OBJS}; G['T'] = gT; G['O'] = 0.0
    return G

def make_scalar(kind, ide, rng, w=None, tau=0.0, legacy=False):
    """returns f(Z) -> float to minimise.  'lex' is strict: f = LEX_K * d_T + min(A_2, cap)
       (legacy=True: the v6 weighting 1e3 * d_T / R_T + A_2, for the sensitivity check)."""
    if kind == 'cheb':
        def f(Z):
            A = chebyshev_gp(Z, w, ide, rng)
            return A if A > 1e-12 else -1.0 / (1.0 + efficiency_term(Z, w, ide, rng))
        return f
    if is_lex(kind):
        G = lex_goals(ide, rng, tau, pinned=(kind == 'lexP')); w2 = {i: 1.0 for i in P2}
        def f(Z):
            dT = max(0.0, Z['T'] - G['T'])
            p2 = chebyshev_gp(Z, w2, ide, rng, goals=G)
            if p2 <= 1e-12: p2 = -1.0 / (1.0 + efficiency_term(Z, w2, ide, rng))
            if legacy: return LEX_LEGACY * dT / rng['T'] + p2
            return LEX_K * dT + min(p2, LEX_P2CAP)
        return f
    if kind in OBJS:                                           # single objective with a tiny tie-break
        return lambda Z: Z[kind] / rng[kind] + 1e-4 * sum(Z[j] / rng[j] for j in OBJS if j != kind)
    raise ValueError(kind)

# =============================================================== 6. DISPATCHING RULES
PEAK = (540, 840)                 # 17:00-22:00 in the 08:00-anchored day
def peak_overlap(a, b):
    tot = 0; d = a // DAY
    while d * DAY < b:
        lo, hi = max(a, d * DAY + PEAK[0]), min(b, d * DAY + PEAK[1])
        if hi > lo: tot += hi - lo
        d += 1
    return tot

RULES = ('EDD', 'ATCS', 'FAM', 'NEDD', 'NATCS')
ATCS_GRID = ((0.5, 0.5), (1.0, 0.5), (2.0, 0.5), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0))

def dispatch(inst, name, rule=True, k1=2.0, k2=0.5, k3=1.0):
    """list scheduling: the loom that becomes free first takes the beam chosen by the rule.
       EDD   earliest due date of the order
       ATCS  apparent tardiness cost with setups (Lee, Bhaskaran and Pinedo 1997)
       FAM   family batching: a beam that needs only knotting first, then EDD
       NEDD  night-protecting EDD: EDD, but a knotting beam whose due date is within one
             mean beam time of the EDD beam is preferred; drawing-in never runs into overtime
       NATCS night-protecting ATCS: ATCS penalising beams whose drawing-in would run past
             midnight if started now; drawing-in never runs into overtime
       Classic rules let the evening crew finish a drawing-in in overtime when allowed."""
    n, B, due = inst['n'], inst['beams'], inst['due']
    sch = Sched(inst, rule); U = set(range(n)); order, assign, otp = [], [None] * n, [0] * n
    pbar = sum(b['p'] for b in B) / n
    sl = [v for f in inst['setup'] for v in inst['setup'][f].values() if v > 0]; sbar = sum(sl) / max(len(sl), 1)
    green = name in ('NEDD', 'NATCS'); pref = 0 if green else 1
    while U:
        k = min(range(inst['m']), key=lambda kk: (sch.lt[kk], kk)); t = sch.lt[k]; lf = sch.lf[k]
        if name in ('EDD', 'FAM', 'NEDD'):
            cand = sorted(U, key=lambda b: (due[B[b]['order']], inst['setup'][lf][B[b]['fam']] > 0, b))
            b = cand[0]
            if name == 'FAM':
                kn = [x for x in cand if inst['setup'][lf][B[x]['fam']] == 0]
                if kn: b = kn[0]
            elif name == 'NEDD' and inst['setup'][lf][B[b]['fam']] > 0:
                kn = [x for x in cand if inst['setup'][lf][B[x]['fam']] == 0
                      and due[B[x]['order']] <= due[B[b]['order']] + pbar]
                if kn: b = kn[0]
        else:
            best = None
            for x in U:
                s = inst['setup'][lf][B[x]['fam']]; tt = max(t, B[x]['rel'])
                I = (1.0 / B[x]['p']) * math.exp(-max(due[B[x]['order']] - B[x]['p'] - s - tt, 0) / (k1 * pbar)) \
                    * math.exp(-s / (k2 * sbar))
                if name == 'NATCS' and s > 0 and sch.rule:
                    sig = sch.peek(x, k, pref)[0]; over = max(0, s - (CREW_END - sig % DAY))
                    I *= math.exp(-k3 * over / s)
                if best is None or I > best[0] + 1e-15: best = (I, x)
            b = best[1]
        sch.place(b, k, pref); U.discard(b); order.append(b); assign[b] = k; otp[b] = pref
    return keys_from(inst, assign, order, otp)

def rule_keys(inst, f_scalar, rule=True):
    """all rules; ATCS/NATCS parameters chosen from a small grid by the scalar.
       Returns {rule: keys}."""
    out = {}
    for nm in RULES:
        if nm in ('ATCS', 'NATCS'):
            cands = [dispatch(inst, nm, rule, k1, k2) for (k1, k2) in ATCS_GRID]
            out[nm] = min(cands, key=lambda kk: f_scalar(evaluate_keys(kk, inst, rule)))
        else:
            out[nm] = dispatch(inst, nm, rule)
    return out

# =============================================================== 7. GP-RKO and BRKGA
# tuned with irace 4.5 on ten separate class-L instances (seeds 101-110), 400 runs, Lex10 scheme,
# objectives T, S, O (run of 19-20 Oct 2026, elite configuration 47 of 4);
# defaults before tuning: psize 30, xbias 0.7, sa_moves 10, sa_T0 0.3, sa_cool 0.85, ils_len 15, pmerge 0.15
RKO_CFG = dict(psize=21, xbias=0.5695, sa_moves=20, sa_T0=0.1999, sa_cool=0.8960, ils_len=9, pmerge=0.0561,
               xover=True, sa=True, ils=True, ltransfer=True, ordermoves=True, seed_rules=True)
ABLATIONS = {'GP-RKO-noX': dict(xover=False), 'GP-RKO-noSA': dict(sa=False), 'GP-RKO-noILS': dict(ils=False),
             'GP-RKO-noMoves': dict(ltransfer=False, ordermoves=False), 'GP-RKO-noSeed': dict(seed_rules=False)}

def rko(inst, fit, budget, seed, seeds_keys=(), cfg=None, tlim=None, marks=None, trace=None):
    """GP-RKO: elite pool shared by BRKGA-style crossover, simulated annealing and iterated
       local search (load transfer, order consolidation/spreading, random reset + descent).
       marks/trace: anytime record of the incumbent (evaluations, f, keys) at the given
       numbers of evaluations (convergence experiment D5)."""
    cfg = dict(RKO_CFG, **(cfg or {})); R = random.Random(seed); n, m = inst['n'], inst['m']; L = 3 * n
    psize = cfg['psize']; t0 = time.time(); ev = 0; mk = [0]; marks = sorted(marks or ())
    init = [list(k) for k in seeds_keys] if cfg['seed_rules'] else []
    init += [[R.random() for _ in range(L)] for _ in range(max(0, psize - len(init)))]
    pool = sorted([(fit(k), k) for k in init], key=lambda x: x[0])[:psize]; ev += len(init)
    def chk(final=False):
        while trace is not None and mk[0] < len(marks) and (ev >= marks[mk[0]] or final):
            trace.append((marks[mk[0]], pool[0][0], list(pool[0][1]))); mk[0] += 1
    def push(f, k):
        pool.append((f, k)); pool.sort(key=lambda x: x[0]); del pool[psize:]; chk()
    orders = list(inst['order_of'].items())
    def out(): return ev >= budget or (tlim and time.time() - t0 > tlim)
    while not out():
        if cfg['xover']:
            _, ek = pool[R.randrange(max(1, psize // 3))]; _, ok = pool[R.randrange(psize)]
            xb = cfg['xbias']; ch = [ek[i] if R.random() < xb else ok[i] for i in range(L)]; push(fit(ch), ch); ev += 1
        if cfg['sa']:
            cf, ck = pool[R.randrange(psize)]; cur = ck[:]; T = cfg['sa_T0']
            for _ in range(cfg['sa_moves']):
                c = cur[:]; c[R.randrange(L)] = R.random(); f = fit(c); ev += 1
                if f < cf or R.random() < math.exp(-(f - cf) / max(T, 1e-6)): cur, cf = c, f
                T *= cfg['sa_cool']
            push(cf, cur)
        if cfg['ils']:
            bf, bk = pool[R.randrange(min(3, psize))]; pe = bk[:]; u = R.random()
            pmg = cfg['pmerge']
            if m > 1 and cfg['ltransfer'] and u < pmg:            # load transfer between two looms
                a, b = R.sample(range(m), 2)
                for i in range(n):
                    if min(int(pe[i] * m), m - 1) == a and R.random() < 0.7: pe[i] = (b + R.random()) / m
            elif m > 1 and cfg['ordermoves'] and u < 2 * pmg:     # consolidate or spread an order
                o, bl = orders[R.randrange(len(orders))]
                if R.random() < 0.5:
                    tgt = min(int(pe[R.choice(bl)] * m), m - 1)
                    for i in bl:
                        if R.random() < 0.5: pe[i] = (tgt + R.random()) / m
                else:
                    for i in bl:
                        if R.random() < 0.5: pe[i] = R.random()
            else:                                                        # random reset of three keys
                for _ in range(3): pe[R.randrange(L)] = R.random()
            pf = fit(pe); ev += 1
            for _ in range(cfg['ils_len']):
                c = pe[:]; u = R.random()
                if u < 0.45:
                    i, j = R.sample(range(n, 2 * n), 2); c[i], c[j] = c[j], c[i]
                elif u < 0.80:
                    c[R.randrange(n)] = R.random()
                else:
                    i = R.randrange(2 * n, 3 * n); c[i] = 1.0 - c[i]
                f = fit(c); ev += 1
                if f < pf: pe, pf = c, f
            push(pf, pe)
        if not (cfg['xover'] or cfg['sa'] or cfg['ils']): break
    chk(final=True)
    return pool, ev

BRKGA_CFG   = dict(p=100, pe=0.20, pm=0.15, rho=0.70)     # standard settings (Goncalves and Resende 2011)
BRKGA_S_CFG = dict(BRKGA_CFG)                              # rule-seeded BRKGA; replaced by the irace result
def _load_brkga_s():
    import json
    if os.path.exists('brkga_s_cfg.json'):
        with open('brkga_s_cfg.json') as fh: d = json.load(fh)
        BRKGA_S_CFG.update(p=int(d['p']), pe=float(d['pe']), pm=float(d['pm']), rho=float(d['rho']))
        return True
    return False

def brkga(inst, fit, budget, seed, p=100, pe=0.20, pm=0.15, rho=0.70, tlim=None, seeds_keys=()):
    """biased random-key GA (Goncalves and Resende 2011), same decoder.  seeds_keys: initial
       chromosomes (the dispatching rules for BRKGA-S); the rest of the population is random."""
    R = random.Random(seed); L = 3 * inst['n']; p = int(p); ne, nm = max(1, int(pe * p)), int(pm * p); t0 = time.time()
    init = [list(k) for k in seeds_keys][:p]; init += [[R.random() for _ in range(L)] for _ in range(p - len(init))]
    pop = sorted([(fit(k), k) for k in init], key=lambda x: x[0]); ev = p
    while ev < budget and not (tlim and time.time() - t0 > tlim):
        nxt = pop[:ne]
        nxt += [(fit(k), k) for k in ([R.random() for _ in range(L)] for _ in range(nm))]; ev += nm
        while len(nxt) < p:
            _, a = pop[R.randrange(ne)]; _, b = pop[R.randrange(ne, p)]
            ch = [a[i] if R.random() < rho else b[i] for i in range(L)]; nxt.append((fit(ch), ch)); ev += 1
        pop = sorted(nxt, key=lambda x: x[0])
    return pop, ev

def seed_descent(inst, fit, budget, seed, seeds_keys):
    """SEED-LS baseline: the best dispatching-rule schedule improved by stochastic descent
       with the inner moves of GP-RKO's local search (swap of two priority keys, reset of an
       assignment key, flip of an overtime key); only improvements are accepted."""
    R = random.Random(seed); n = inst['n']; ev = 0
    cands = [(fit(k), list(k)) for k in seeds_keys]; ev += len(cands)
    cf, cur = min(cands, key=lambda x: x[0])
    while ev < budget:
        c = cur[:]; u = R.random()
        if u < 0.45:
            i, j = R.sample(range(n, 2 * n), 2); c[i], c[j] = c[j], c[i]
        elif u < 0.80:
            c[R.randrange(n)] = R.random()
        else:
            i = R.randrange(2 * n, 3 * n); c[i] = 1.0 - c[i]
        f = fit(c); ev += 1
        if f < cf: cur, cf = c, f
    return [(cf, cur)], ev

# =============================================================== 8. CP-SAT MODEL (one crew)
E_SC = 1200                       # energy in 1/1200 TL: P/60 * price is then an exact integer
def _segments(H, tariff):
    out = []; d = 0
    while d * DAY < H:
        for (a, b, _, pr, ef, sh) in SEGMENTS_TOU:
            lo, hi = d * DAY + a, min(d * DAY + b, H)
            price = pr if tariff == 'ToU' else FLAT_PRICE
            if hi > lo: out.append((lo, hi, int(round(P_LOOM / 60.0 * price * E_SC))))
        d += 1
    return out

CP_H_FACTOR = 1.5
def cp_horizon(inst):
    """horizon of the exact model: CP_H_FACTOR times the longest schedule built by the
       dispatching rules under either policy (checked a posteriori: no solution reaches it)."""
    if 'H_cp' not in inst:
        cm = 0
        for rule in (True, False):
            for nm in RULES:
                kk = dispatch(inst, nm, rule) if nm not in ('ATCS', 'NATCS') else dispatch(inst, nm, rule, 2.0, 0.5)
                cm = max(cm, evaluate_keys(kk, inst, rule)['Cmax'])
        inst['H_cp'] = int(CP_H_FACTOR * cm) + DAY
    return inst['H_cp']

def build_cp(inst, rule=True):
    """Exact model with the decoder's semantics for ONE drawing-in crew.
       Looms: one circuit per loom (ready_j = end of the loom predecessor, or a_k).
       Crew: one circuit over the beams that need drawing-in, in order of the crew time at
       which their looms are ready (first come, first served); a drawing-in starts at
       max(crew time of its readiness, crew time of the end of the crew's previous job).
       Overtime (<= OT_MAX after midnight) or pause to 08:00."""
    assert inst['C'] == 1, "the exact model is written for one crew"
    n, m, B, S = inst['n'], inst['m'], inst['beams'], inst['setup']; H = cp_horizon(inst)
    smax = max(v for f in S for v in S[f].values()); M = cp_model.CpModel()
    def ct(x, nm):
        if not rule: return x
        d = M.NewIntVar(0, H // DAY + 2, f"cd{nm}"); M.AddDivisionEquality(d, x, DAY)
        nt = M.NewBoolVar(f"cn{nm}")
        M.Add(x - DAY * d >= CREW_END).OnlyEnforceIf(nt); M.Add(x - DAY * d <= CREW_END - 1).OnlyEnforceIf(nt.Not())
        y = M.NewIntVar(0, H + 2 * DAY, f"ct{nm}")
        M.Add(y == x).OnlyEnforceIf(nt.Not()); M.Add(y == DAY * (d + 1)).OnlyEnforceIf(nt)
        return y
    pres = [[M.NewBoolVar(f"x{j}_{k}") for k in range(m)] for j in range(n)]
    for j in range(n): M.AddExactlyOne(pres[j])
    ready = [M.NewIntVar(0, H, f"r{j}") for j in range(n)]
    setv = [M.NewIntVar(0, smax, f"s{j}") for j in range(n)]
    start = [M.NewIntVar(0, H, f"st{j}") for j in range(n)]
    end = [M.NewIntVar(0, H + B[j]['p'], f"en{j}") for j in range(n)]
    di_terms = [[] for _ in range(n)]
    for j in range(n): M.Add(end[j] == start[j] + B[j]['p'])
    for k in range(m):
        arcs = []; empty = M.NewBoolVar(f"E{k}"); arcs.append((0, 0, empty))
        for j in range(n):
            M.AddImplication(empty, pres[j][k].Not()); arcs.append((j + 1, j + 1, pres[j][k].Not()))
            f = M.NewBoolVar(f"f{j}_{k}"); arcs.append((0, j + 1, f)); arcs.append((j + 1, 0, M.NewBoolVar(f"l{j}_{k}")))
            M.Add(ready[j] == inst['avail'][k]).OnlyEnforceIf(f)
            sv = S[inst['f0'][k]][B[j]['fam']]; M.Add(setv[j] == sv).OnlyEnforceIf(f)
            if sv > 0: di_terms[j].append(f)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                u = M.NewBoolVar(f"u{i}_{j}_{k}"); arcs.append((i + 1, j + 1, u))
                M.Add(ready[j] == end[i]).OnlyEnforceIf(u)
                sv = S[B[i]['fam']][B[j]['fam']]; M.Add(setv[j] == sv).OnlyEnforceIf(u)
                if sv > 0: di_terms[j].append(u)
        M.AddCircuit(arcs)
    di = [M.NewBoolVar(f"di{j}") for j in range(n)]
    x0 = [M.NewIntVar(0, H, f"x0{j}") for j in range(n)]
    sig = [M.NewIntVar(0, H + DAY, f"sg{j}") for j in range(n)]
    cend = [M.NewIntVar(0, H + 2 * DAY, f"ce{j}") for j in range(n)]
    otv = [M.NewBoolVar(f"o{j}") for j in range(n)]; Oj = []; pzs = []
    for j in range(n):
        M.Add(di[j] == sum(di_terms[j])) if di_terms[j] else M.Add(di[j] == 0)
        M.AddMaxEquality(x0[j], [ready[j], B[j]['rel']])
    cx0 = [ct(x0[j], f"x{j}") for j in range(n)]
    for j in range(n):
        M.Add(sig[j] == x0[j]).OnlyEnforceIf(di[j].Not())
        if rule:
            ds = M.NewIntVar(0, H // DAY + 2, f"ds{j}"); M.AddDivisionEquality(ds, sig[j], DAY)
            phs = M.NewIntVar(0, DAY - 1, f"ph{j}"); M.Add(phs == sig[j] - DAY * ds)
            M.Add(phs <= CREW_END - 1).OnlyEnforceIf(di[j])
            ov = M.NewIntVar(0, smax + DAY, f"ov{j}"); M.AddMaxEquality(ov, [setv[j] + phs - CREW_END, 0])
            op = M.NewBoolVar(f"op{j}"); M.Add(ov >= 1).OnlyEnforceIf(op); M.Add(ov == 0).OnlyEnforceIf(op.Not())
            o = otv[j]; M.AddImplication(o, di[j]); M.AddImplication(o, op); M.Add(ov <= OT_MAX).OnlyEnforceIf(o)
            pz = M.NewBoolVar(f"pz{j}")
            M.AddBoolAnd([di[j], op, o.Not()]).OnlyEnforceIf(pz); M.AddBoolOr([di[j].Not(), op.Not(), o]).OnlyEnforceIf(pz.Not())
            M.Add(cend[j] == sig[j] + setv[j] + (DAY - CREW_END) * pz)
            oj = M.NewIntVar(0, OT_MAX, f"O{j}"); M.Add(oj == ov).OnlyEnforceIf(o); M.Add(oj == 0).OnlyEnforceIf(o.Not())
            Oj.append(oj); pzs.append(pz)
        else:
            M.Add(otv[j] == 0); M.Add(cend[j] == sig[j] + setv[j])
        M.Add(start[j] == x0[j]).OnlyEnforceIf(di[j].Not()); M.Add(start[j] == cend[j]).OnlyEnforceIf(di[j])
    ccend = [ct(cend[j], f"c{j}") for j in range(n)]
    arcs = []; nodi = M.NewBoolVar("nodi"); arcs.append((0, 0, nodi))
    for j in range(n):
        M.AddImplication(nodi, di[j].Not()); arcs.append((j + 1, j + 1, di[j].Not()))
        a0 = M.NewBoolVar(f"a0{j}"); arcs.append((0, j + 1, a0)); arcs.append((j + 1, 0, M.NewBoolVar(f"az{j}")))
        M.Add(sig[j] == cx0[j]).OnlyEnforceIf(a0)
        for i in range(n):
            if i == j: continue
            a = M.NewBoolVar(f"a{i}_{j}"); arcs.append((i + 1, j + 1, a))
            mx = M.NewIntVar(0, H + 2 * DAY, f"mx{i}_{j}"); M.AddMaxEquality(mx, [cx0[j], ccend[i]])
            M.Add(sig[j] == mx).OnlyEnforceIf(a)
            M.Add(cx0[i] <= cx0[j]).OnlyEnforceIf(a)                 # first come, first served
    M.AddCircuit(arcs)
    # objectives
    HS = H + max(b['p'] for b in B) + 3 * DAY
    # energy: cost of weaving over [st, en) = F(en) - F(st), F(t) = DCOST * day(t) + f(phase(t))
    fcum = [0] * (DAY + 1); price = {}
    for (a, b, _, pr, ef, sh) in SEGMENTS_TOU:
        c = int(round(P_LOOM / 60.0 * (pr if inst['tariff'] == 'ToU' else FLAT_PRICE) * E_SC))
        for t in range(a, b): price[t] = c
    for t in range(DAY): fcum[t + 1] = fcum[t] + price[t]
    DCOST = fcum[DAY]; Eparts = []
    SEGC = []                                                  # (a, b, cost/min, cumulative at a)
    for (a, b, _, pr, ef, sh) in SEGMENTS_TOU: SEGC.append((a, b, price[a], fcum[a]))
    def Fvar(x, nm):
        d = M.NewIntVar(0, HS // DAY + 1, f"Fd{nm}"); M.AddDivisionEquality(d, x, DAY)
        ph = M.NewIntVar(0, DAY - 1, f"Fp{nm}"); M.Add(ph == x - DAY * d)
        fv = M.NewIntVar(0, DCOST, f"Ff{nm}"); zs = []
        for q, (a, b, c, cu) in enumerate(SEGC):
            z = M.NewBoolVar(f"Fz{nm}_{q}"); zs.append(z)
            M.Add(ph >= a).OnlyEnforceIf(z); M.Add(ph <= b - 1).OnlyEnforceIf(z)
            M.Add(fv == cu + c * (ph - a)).OnlyEnforceIf(z)
        M.AddExactlyOne(zs)
        return DCOST * d + fv
    if 'E' in OBJS:
        for j in range(n):
            Eparts.append(Fvar(end[j], f"e{j}") - Fvar(start[j], f"s{j}"))
    ub = dict(T=HS * len(inst['order_of']), S=n * smax, E=max(price.values()) * inst['work'], O=n * OT_MAX)
    V = {}
    V['S'] = M.NewIntVar(0, ub['S'], "S"); M.Add(V['S'] == sum(setv))
    if 'E' in OBJS:
        V['E'] = M.NewIntVar(0, ub['E'], "E"); M.Add(V['E'] == sum(Eparts))
    V['O'] = M.NewIntVar(0, ub['O'], "O"); M.Add(V['O'] == (sum(Oj) if Oj else 0))
    tl = []
    for o, bl in inst['order_of'].items():
        Co = M.NewIntVar(0, HS, f"C{o}"); M.AddMaxEquality(Co, [end[i] for i in bl])
        To = M.NewIntVar(0, ub['T'], f"T{o}"); M.AddMaxEquality(To, [Co - inst['due'][o], 0]); tl.append(To)
    V['T'] = M.NewIntVar(0, ub['T'], "T"); M.Add(V['T'] == sum(tl))
    V.update(pres=pres, start=start, sig=sig, di=di, ot=otv, ub=ub)
    return M, V

UNIT = {'T': 1, 'S': 1, 'E': E_SC, 'O': 1}
CP_WORKERS = 16
def _solve(M, tlim):
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = tlim
    sv.parameters.num_search_workers = CP_WORKERS; st = sv.Solve(M); return sv, st

LAST_CPZ = {}
def _extract(sv, V, inst):
    LAST_CPZ.clear(); LAST_CPZ.update({i: sv.Value(V[i]) / UNIT[i] for i in OBJS})
    n, m = inst['n'], inst['m']; assign = [0] * n; key = [0] * n; otp = [0] * n
    for j in range(n):
        assign[j] = next(k for k in range(m) if sv.Value(V['pres'][j][k]))
        key[j] = sv.Value(V['sig'][j]) if sv.Value(V['di'][j]) else sv.Value(V['start'][j])
        otp[j] = sv.Value(V['ot'][j])
    order = sorted(range(n), key=lambda b: (key[b], b))
    return keys_from(inst, assign, order, otp)

def _hint_keys(M, V, inst, keys, rule):
    if keys is None: return
    sch = decode(keys, inst, rule)
    for b, r in enumerate(sch.rec):
        for k in range(inst['m']): M.AddHint(V['pres'][b][k], 1 if k == r[0] else 0)
        M.AddHint(V['start'][b], r[6])

def _good(st): return st in (cp_model.OPTIMAL, cp_model.FEASIBLE)

def cp_minimize(inst, kind, ide, rng, rule=True, w=None, tau=0.0, tlim=60, hint=None):
    """CP-SAT for the same scalars as make_scalar: 'cheb', 'lex' or a single objective
       (lexicographic).  Returns dict(keys, optimal, time) or None."""
    t0 = time.time(); M, V = build_cp(inst, rule); ub = V['ub']; proven = True
    R = {i: max(rng[i] * UNIT[i], 1.0) for i in OBJS}; QS = 1e3 * max(R.values())
    def run(tl):
        nonlocal proven
        M.ClearHints(); _hint_keys(M, V, inst, hint, rule); sv, st = _solve(M, tl)
        if not _good(st): return None, None
        proven = proven and st == cp_model.OPTIMAL; return sv, st
    def cheb(ws, G):
        q = {i: int(round(ws[i] * QS / R[i])) for i in ws}
        Dub = max(q[i] * ub[i] for i in ws) + 1; D = M.NewIntVar(0, Dub, ""); terms = []
        for i in ws:
            dev = M.NewIntVar(0, ub[i], ""); M.AddMaxEquality(dev, [V[i] - int(math.floor(G[i] * UNIT[i])), 0])
            M.Add(D >= q[i] * dev); terms.append(q[i] * dev)
        return D, int(round(1 / RHO)) * D + sum(terms), q
    def eff(ws, q): return sum(q[i] * V[i] for i in ws)
    if kind in OBJS:
        M.Minimize(V[kind]); sv, st = run(2 * tlim / 3)
        if sv is None: return None
        hint = _extract(sv, V, inst); M.Add(V[kind] <= sv.Value(V[kind]))
        q = {j: int(round(QS / R[j])) for j in OBJS if j != kind}; M.Minimize(eff(q, q))
        sv2, _ = run(tlim / 3)
        keys = _extract(sv2, V, inst) if sv2 else hint
        return dict(keys=keys, optimal=proven, time=time.time() - t0)
    if kind == 'cheb':
        G = {i: ide[i] + GAMMA * rng[i] for i in OBJS}; ws = {i: w[i] for i in OBJS if w.get(i, 0) > 0}
    else:
        G = lex_goals(ide, rng, tau, pinned=(kind == 'lexP')); ws = {i: 1.0 for i in P2}
        devT = M.NewIntVar(0, ub['T'], "devT"); M.AddMaxEquality(devT, [V['T'] - int(math.floor(G['T'])), 0])
        M.Minimize(devT); sv, st = run(tlim / 3)
        if sv is None: return None
        hint = _extract(sv, V, inst); M.Add(devT <= sv.Value(devT))
    D, obj, q = cheb(ws, G); M.Minimize(obj); sv, st = run(tlim / 3 if is_lex(kind) else 2 * tlim / 3)
    if sv is None: return None
    keys = _extract(sv, V, inst)
    if sv.Value(D) == 0:
        M.Add(D == 0); M.Minimize(eff(ws, q)); hint = keys; sv2, _ = run(tlim / 3)
        if sv2 is None: return dict(keys=keys, optimal=False, time=time.time() - t0)
        keys = _extract(sv2, V, inst); M.Add(eff(ws, q) <= sv2.Value(eff(ws, q)))
    else:
        M.Add(obj <= sv.Value(obj))
    # priority 3, efficiency restoration: least sum of normalised objectives at the same achievement
    qa = {i: int(round(QS / R[i])) for i in OBJS}; M.Minimize(eff(qa, qa)); hint = keys; sv3, _ = run(tlim / 3)
    if sv3: keys = _extract(sv3, V, inst)
    else: proven = False
    return dict(keys=keys, optimal=proven, time=time.time() - t0)

# =============================================================== 9. PAYOFF TABLE AND RUNS
def run_method(inst, method, scal, rule, seed, budget, rkeys=None, cfg=None, tlim=None, marks=None, trace=None):
    """one run of a method for a scalar spec scal = (kind, w, tau) under a policy.
       Returns dict(keys, f, Z, time, evals); f is always the strict scalar of the final
       schedule (GP-RKO-w1e3 searches with the v6 lexicographic weighting)."""
    ide, rng = inst['ide'], inst['rng']; kind, w, tau = scal
    f = make_scalar(kind, ide, rng, w=w, tau=tau)
    fs = make_scalar(kind, ide, rng, w=w, tau=tau, legacy=True) if method == 'GP-RKO-w1e3' else f
    fit = lambda k: fs(evaluate_keys(k, inst, rule)); t0 = time.time()
    seeds = list((rkeys or {}).values())
    if method == 'BRKGA':
        pop, ev = brkga(inst, fit, budget, seed, tlim=tlim, **BRKGA_CFG); best = pop[0]
    elif method == 'BRKGA-S':
        pop, ev = brkga(inst, fit, budget, seed, tlim=tlim, seeds_keys=seeds, **BRKGA_S_CFG); best = pop[0]
    elif method == 'SEED-LS':
        pop, ev = seed_descent(inst, fit, budget, seed, seeds); best = pop[0]
    elif method.startswith('GP-RKO'):
        pool, ev = rko(inst, fit, budget, seed, seeds_keys=seeds, cfg=cfg, tlim=tlim, marks=marks, trace=trace)
        best = pool[0]
    else:                                                       # a dispatching rule
        best = (fit(rkeys[method]), rkeys[method]); ev = 1
    Z = evaluate_keys(best[1], inst, rule)
    return dict(keys=best[1], f=f(Z), Z=Z, time=time.time() - t0, evals=ev)

def payoff(inst, budget, seeds=(1, 2), use_cp=False, cp_tlim=60):
    """ideal and nadir over both policies (night rule, night crew): for every objective the
       best schedule found by the rules, GP-RKO runs and (small instances) CP-SAT."""
    rows = []; rule_rows = []; floor = RANGE_FLOOR
    tmp_ide = {i: 0.0 for i in OBJS}; tmp_rng = dict(floor); inst['ide'], inst['rng'] = tmp_ide, tmp_rng
    for rule in (True, False):
        for i in OBJS:
            if i == 'O' and not rule: continue
            f = make_scalar(i, tmp_ide, tmp_rng); rk = rule_keys(inst, f, rule); cands = []
            for nm, k in rk.items():
                Zr = evaluate_keys(k, inst, rule); cands.append(Zr); rule_rows.append(Zr)
            for sd in seeds:
                r = run_method(inst, 'GP-RKO', (i, None, 0.0), rule, sd, budget, rkeys=rk); cands.append(r['Z'])
                add_pool(inst, rule, r['keys'], f'payoff-{i}')
            if use_cp:
                best = min(rk.values(), key=lambda kk: f(evaluate_keys(kk, inst, rule)))
                c = cp_minimize(inst, i, tmp_ide, tmp_rng, rule=rule, tlim=cp_tlim, hint=best)
                if c: cands.append(evaluate_keys(c['keys'], inst, rule)); add_pool(inst, rule, c['keys'], f'payoff-{i}')
            rows.append(min(cands, key=lambda Z: (Z[i], sum(Z[j] / floor[j] for j in OBJS))))
    ide = {i: min(r[i] for r in rows) for i in OBJS}
    # nadir: worst value in the payoff table, capped at the worst value of a dispatching-rule
    # schedule (a plan a planner would accept), because extreme payoff rows can inflate it
    nad = {i: min(max(r[i] for r in rows), max(r[i] for r in rule_rows)) for i in OBJS}
    nad = {i: max(nad[i], ide[i]) for i in OBJS}
    rng = {i: max(nad[i] - ide[i], floor[i]) for i in OBJS}
    inst['ide'], inst['nad'], inst['rng'] = ide, nad, rng
    inst['ide0'] = dict(ide)                                   # ideal of the payoff table (before refit)
    return ide, nad, rng

def add_pool(inst, pol, keys, src):
    inst.setdefault('kpool', {}).setdefault(pol, []).append((list(keys), src))
    inst.pop('_bks', None); inst.pop('_nm', None)

def pool_keys(inst):
    """distinct keys found for an instance (any method, scheme or policy)."""
    seen = set()
    for pol, lst in inst.get('kpool', {}).items():
        for keys, src in lst:
            t = tuple(round(x, 9) for x in keys)
            if t in seen: continue
            seen.add(t); yield keys, src

def scheme_list(kinds):
    """'Balanced' etc. -> ('cheb', w, 0); 'Lex5' -> ('lex', None, 0.05);
       'Lex0P', 'Lex10P' (D6) -> ('lexP', None, tau): tardiness goal pinned at (1 + tau) z*_T."""
    out = {}
    for s in kinds:
        if s.startswith('Lex') and s.endswith('P'): out[s] = ('lexP', None, int(s[3:-1]) / 100.0)
        elif s.startswith('Lex'): out[s] = ('lex', None, int(s[3:]) / 100.0)
        else: out[s] = ('cheb', PROFILES[s], 0.0)
    return out

def value(scal, f, inst):
    """attainment in range units: Chebyshev A as is; lexicographic GP: the tardiness deviation
       in tardiness ranges (priority 1), priority 2 entering only as a tie-break."""
    return f / (LEX_K * inst['rng']['T']) if is_lex(scal[0]) else f

def refit_ideal(inst):
    """a posteriori ideal: best value of every objective over all schedules found for the
       instance (pooled keys re-evaluated under the night rule and the night crew; O under the
       night rule only, as in the payoff table).  Goals, ranges and the attainment of every
       recorded run are recomputed from the stored objective vectors.  Returns the
       improvement of the ideal per objective (minutes)."""
    old = dict(inst['ide']); new = dict(old)
    for keys, _ in pool_keys(inst):
        for rule in (True, False):
            Z = evaluate_keys(keys, inst, rule)
            for i in OBJS:
                if i == 'O' and not rule: continue
                new[i] = min(new[i], Z[i])
    nad = {i: max(inst['nad'][i], new[i]) for i in OBJS}
    inst['ide'], inst['nad'] = new, nad
    inst['rng'] = {i: max(nad[i] - new[i], RANGE_FLOOR[i]) for i in OBJS}
    inst.pop('_bks', None); inst.pop('_nm', None)
    for r in RESULTS:
        if r['inst'] == inst['name']:
            sc = scheme_list([r['scheme']])[r['scheme']]
            r['f'] = make_scalar(sc[0], new, inst['rng'], w=sc[1], tau=sc[2])(r['Z']); r['v'] = value(sc, r['f'], inst)
    return {i: old[i] - new[i] for i in OBJS}

# =============================================================== 10. EXPERIMENT CONFIGURATION
WORKERS    = 18                   # parallel processes for heuristic runs (threads minus two)
CP_WORKERS = 16                   # CP-SAT threads (CP-SAT runs alone, before the parallel batches)
REPS       = (1, 2, 3, 4, 5)      # independent replications of the metaheuristics
BUDGET     = {'S': 20000, 'M': 50000, 'L': 100000}     # evaluations per run (equal for all methods)
PAY_BUDGET = {'S': 10000, 'M': 25000, 'L': 40000}      # evaluations per payoff-table run
PAY_SEEDS  = (1, 2)
NIGHT_BUDGET = {'S': 5000, 'L': 20000}                 # third stage: minimum night drawing-in
RUN = dict(D1=True, D2=True, D3=True, D6=True, D5=True, D4=True)
POL = {True: 'night rule', False: 'night crew', 'EH': 'equal crew-hours', 'R+1': 'night rule +1 crew',
       'R+2': 'night rule +2 crews'}
DAYCAP = ('R+1', 'R+2')           # day-capacity policies, Lex0 only, on instances with C = 3
def pols_lex0(inst):
    return pols_of(inst) + (DAYCAP if inst['C'] == 3 else ())
def pols_of(inst):
    """policies run on an instance: equal crew-hours only where 2C/3 is an integer (C = 3)."""
    return (True, False) + (('EH',) if eh_crews(inst['C']) else ())
# D1 exact validation and exact price of the night rule (small, one crew)
D1_SEEDS, D1_SCHEMES = tuple(range(1, 11)), ('Balanced', 'Delivery', 'Lex0', 'Lex10')
D1_METHODS = ('GP-RKO', 'BRKGA', 'BRKGA-S', 'SEED-LS')
CP_TLIM, CP_PAY_TLIM = 180, 90
# D2 scalability and algorithm comparison (night rule)
D2_SEEDS, D2_SCHEMES = {'M': tuple(range(1, 11)), 'L': tuple(range(1, 11))}, ('Balanced', 'Lex10')
D2_METHODS = ('GP-RKO', 'BRKGA', 'BRKGA-S', 'SEED-LS') + tuple(ABLATIONS)
D2_LEGACY = True                  # class L, Lex10: GP-RKO with the v6 weighting 1e3 * p1 + p2
# D3 price map of the night rule: full factorial on mill-size instances, three policies
D3_M = 50
D3_FACTORS = dict(C=(2, 3, 'inf'), nf=(4, 8), lmax=(200, 300, 500), delta=(1.0, 0.8))
D3_SEEDS, D3_REPS, D3_SCHEMES = (1, 2), (1, 2, 3), ('Lex0', 'Lex10')
D3R_SEEDS = tuple(range(1, 7))    # robustness: general asymmetric changeover matrix
D3_BUDGET, D3_PAY_BUDGET, D3_PAY_SEEDS = 100000, 50000, (1,)
D3_POLISH_BUDGET = 300000         # cross-seeded polishing run per instance, scheme and policy (equal for all)
RESUME = False                    # --resume: continue an interrupted D3 from its checkpoint files
FMETA = {'C': 'C_f'}              # factor name -> key in inst['meta']
FLABEL = {'C': 'crews $C$', 'nf': 'families $n_f$', 'lmax': r'beam capacity $L^{\max}$ (m)', 'delta': r'tightness $\delta$'}
D3_SEEDLS_BUDGET = 100000         # SEED-LS polishing run (descent from the best pooled schedule)
# D5 convergence of the price estimate: 12 D3 instances (lmax 300, seed 1), long anytime runs
D5_MARKS = (25000, 50000, 100000, 200000, 400000)
D5_REPS, D5_SCHEME, D5_SELECT = (1, 2, 3, 4, 5), 'Lex0', dict(lmax=300, seed=1)
# D4 case study: tolerance curve of the lexicographic GP, three policies
CASE = dict(seed=7, m=50, C=3, n_fam=4, delta=1.0)
D4_SCHEMES = ('Lex0', 'Lex5', 'Lex10', 'Lex20', 'Lex30', 'Balanced', 'Delivery', 'Labour', 'NightSafe')
D4_NIGHT_BUDGET = 50000
BOOT_B = 2000                     # bootstrap resamples for confidence intervals
OUT_PREFIX = 'v9'
BASE_PREFIX = 'v9'                # checkpoint names do not depend on --only
V8_D3_CKPT = 'v8_D3_ckpt.pkl'     # with --resume, D3 continues from the v8 runs if no v9 checkpoint exists
QUICK = False                     # True: tiny set-up for a short test
TUNE_SEEDS = tuple(range(101, 111))   # separate tuning instances (class L) for irace

def _quick():
    global REPS, BUDGET, PAY_BUDGET, PAY_SEEDS, D1_SEEDS, D2_SEEDS, D3_FACTORS, D3_SEEDS, D3_REPS, D3R_SEEDS
    global CP_TLIM, CP_PAY_TLIM, OUT_PREFIX, D4_SCHEMES, D3_M, D3_BUDGET, D3_PAY_BUDGET, NIGHT_BUDGET
    global D5_MARKS, D5_REPS, D4_NIGHT_BUDGET, BOOT_B, D3_POLISH_BUDGET
    REPS, PAY_SEEDS = (1, 2), (1,); BUDGET = {'S': 1500, 'M': 1500, 'L': 1500}; PAY_BUDGET = {'S': 800, 'M': 800, 'L': 800}
    D1_SEEDS, D2_SEEDS = (1, 2), {'M': (1,), 'L': (1, 2, 3)}
    D3_FACTORS = dict(C=(2, 3, 'inf'), nf=(4,), lmax=(300,), delta=(1.0, 0.8))
    D3_SEEDS, D3_REPS, D3R_SEEDS, CP_TLIM, CP_PAY_TLIM = (1,), (1, 2), (1,), 10, 5
    D4_SCHEMES, OUT_PREFIX = ('Lex0', 'Lex10', 'Balanced'), 'v9quick'; D3_BUDGET, D3_PAY_BUDGET = 1500, 800
    global BASE_PREFIX, D3_SEEDLS_BUDGET
    global RESTORE_BUDGET, V8_D3_CKPT
    BASE_PREFIX = 'v9quick'; D3_SEEDLS_BUDGET = 500; V8_D3_CKPT = 'none.pkl'
    RESTORE_BUDGET = {'S': 300, 'L': 500}
    NIGHT_BUDGET = {'S': 300, 'L': 500}; D4_NIGHT_BUDGET = 500; D5_MARKS = (500, 1000, 2000); D5_REPS = (1, 2)
    D3_POLISH_BUDGET = 1000
    BOOT_B = 300

# =============================================================== 11. STATISTICS (no scipy needed)
def _gammq(a, x):
    """regularised upper incomplete gamma Q(a, x) (Numerical Recipes)."""
    if x <= 0: return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1; d *= x / ap; s += d
            if abs(d) < abs(s) * 1e-12: break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1 - a; c = 1e300; d = 1.0 / b; h = d
    for i in range(1, 500):
        an = -i * (i - a); b += 2; d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c; d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1) < 1e-12: break
    return math.exp(-x + a * math.log(x) - gln) * h

def _ranks(v):
    idx = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and abs(v[idx[j + 1]] - v[idx[i]]) < 1e-12: j += 1
        for q in range(i, j + 1): r[idx[q]] = (i + j) / 2 + 1
        i = j + 1
    return r

def friedman(blocks):
    """blocks: list of lists (one value per method, lower is better). Returns (mean ranks, chi2, p)."""
    N, k = len(blocks), len(blocks[0]); R = [0.0] * k
    for b in blocks:
        for j, r in enumerate(_ranks(b)): R[j] += r
    chi2 = 12.0 / (N * k * (k + 1)) * sum(x * x for x in R) - 3 * N * (k + 1)
    return [x / N for x in R], chi2, _gammq((k - 1) / 2.0, chi2 / 2.0)

def wilcoxon(x, y):
    """two-sided Wilcoxon signed-rank test, normal approximation with tie correction."""
    d = [a - b for a, b in zip(x, y) if abs(a - b) > 1e-12]; n = len(d)
    if n == 0: return 1.0
    r = _ranks([abs(v) for v in d]); Wp = sum(ri for ri, v in zip(r, d) if v > 0)
    mu = n * (n + 1) / 4.0; ties = {}
    for v in r: ties[v] = ties.get(v, 0) + 1
    var = n * (n + 1) * (2 * n + 1) / 24.0 - sum(t ** 3 - t for t in ties.values()) / 48.0
    if var <= 0: return 1.0
    z = (Wp - mu) / math.sqrt(var); return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj

def mstats(v):
    v = [x for x in v if x is not None and x == x]
    if not v: return float('nan'), float('nan'), float('nan')
    mu = sum(v) / len(v); sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return min(v), mu, sd

def _clean(v): return [x for x in v if x is not None and x == x]
def mean_(v):
    v = _clean(v); return sum(v) / len(v) if v else float('nan')
def median_(v):
    v = sorted(_clean(v))
    if not v: return float('nan')
    h = len(v) // 2; return v[h] if len(v) % 2 else (v[h - 1] + v[h]) / 2

def boot_ci(v, seed=12345, alpha=0.05):
    """percentile bootstrap interval of the mean (resampling instances)."""
    v = _clean(v)
    if len(v) < 2: return float('nan'), float('nan')
    R = random.Random(seed); n = len(v); ms = []
    for _ in range(BOOT_B): ms.append(sum(v[R.randrange(n)] for _ in range(n)) / n)
    ms.sort(); return ms[int(alpha / 2 * BOOT_B)], ms[int((1 - alpha / 2) * BOOT_B) - 1]

def fci(v, w=6, d=1):
    """'mean [lo, hi]' of a list."""
    lo, hi = boot_ci(v); return f"{mean_(v):{w}.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"

def p0(v):
    """Wilcoxon signed-rank p-value of a list against zero."""
    v = _clean(v); return wilcoxon(v, [0.0] * len(v)) if v else float('nan')

# =============================================================== 12. RUN BOOKKEEPING
RESULTS = []; INSTREG = {}; D5_OUT = {}

def light(inst):
    return {k: v for k, v in inst.items() if k not in ('kpool', 'pool', '_bks', '_nm')}

def task_run(args):
    """worker: one heuristic run (or all rules) for an instance, scheme and policy."""
    inst, scheme, scal, rule, method, rep, budget, cfg = args
    f = make_scalar(scal[0], inst['ide'], inst['rng'], w=scal[1], tau=scal[2]); rk = rule_keys(inst, f, rule)
    out = []
    if method == 'RULES':
        for nm in RULES:
            r = run_method(inst, nm, scal, rule, 0, 1, rkeys=rk); r.update(method=nm, rep=0); out.append(r)
    else:
        r = run_method(inst, method, scal, rule, rep + seed_offset(rule), budget, rkeys=rk, cfg=cfg)
        r.update(method=method, rep=rep); out.append(r)
    for r in out: r.update(inst=inst['name'], scheme=scheme, rule=rule)
    return out

def task_payoff(args):
    inst, budget, seeds = args
    ide, nad, rng = payoff(inst, budget, seeds)
    return inst['name'], ide, nad, rng, inst.get('kpool', {})

def task_trace(args):
    """worker: one long GP-RKO run with the incumbent recorded at D5_MARKS."""
    inst, scal, rule, rep, marks = args
    f = make_scalar(scal[0], inst['ide'], inst['rng'], w=scal[1], tau=scal[2]); rk = rule_keys(inst, f, rule); tr = []
    run_method(inst, 'GP-RKO', scal, rule, rep, max(marks), rkeys=rk, marks=marks, trace=tr)
    return inst['name'], rule, rep, [(mk, fv, evaluate_keys(k, inst, rule)) for mk, fv, k in tr]

def seed_offset(pol):
    """EH with C = 3 is the same problem as the night crew with C = 2 on the twin instance;
       its runs get other seeds so that the twin comparison is a test of search reliability."""
    return 500 if pol == 'EH' else 0

def task_polish(args):
    """worker: cross-seeded polishing run: GP-RKO started from the best pooled schedules
       (found under any policy or scheme) re-evaluated under this scheme and policy."""
    inst, scheme, scal, pol, seeds, budget, seed, method = args
    r = run_method(inst, method, scal, pol, seed + seed_offset(pol), budget, rkeys={i: k for i, k in enumerate(seeds)})
    r.update(method=method + '-polish', rep=1, inst=inst['name'], scheme=scheme, rule=pol)
    return [r]

def save_pickle(obj, path):
    """atomic pickle (a crash while writing never leaves a broken file)."""
    import pickle
    with open(path + '.tmp', 'wb') as fh: pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + '.tmp', path)

def load_pickle(path):
    import pickle
    with open(path, 'rb') as fh: return pickle.load(fh)

def pmap(fn, tasks, label='', ckpt=None):
    """run tasks in parallel (or serially when WORKERS <= 1).  ckpt: file that keeps the
       finished results (in task order); with RESUME the finished prefix is not run again."""
    t0 = time.time(); out = []
    if ckpt and RESUME and os.path.exists(ckpt):
        out = load_pickle(ckpt)
        print(f"   {label}: resumed, {len(out)}/{len(tasks)} tasks already done", flush=True)
    rest = tasks[len(out):]; step = max(1, len(tasks) // 20); n0 = len(out)
    if WORKERS <= 1 or len(rest) <= 1:
        for t in rest:
            out.append(fn(t))
            if ckpt and len(out) % step == 0: save_pickle(out, ckpt)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for r in ex.map(fn, rest, chunksize=1):
                out.append(r)
                if len(out) % step == 0:
                    if ckpt: save_pickle(out, ckpt)
                    print(f"      {label}: {len(out)}/{len(tasks)} tasks, {time.time()-t0:.0f} s", flush=True)
    if ckpt: save_pickle(out, ckpt)
    print(f"   {label}: {len(tasks)} tasks ({len(tasks) - n0} run now) in {time.time()-t0:.0f} s", flush=True)
    return out

def record(exp, res_list, INST):
    for r in res_list:
        inst = INST[r['inst']]; INSTREG[inst['name']] = inst; scal = scheme_list([r['scheme']])[r['scheme']]
        r['exp'] = exp; r['v'] = value(scal, r['f'], inst); RESULTS.append(r)
        add_pool(inst, r['rule'], r['keys'], r['method'])

def write_csv(path):
    """one row per recorded run (attainment after the a posteriori ideal update)."""
    import csv
    rows = []
    for r in RESULTS:
        inst = INSTREG[r['inst']]; Z = r['Z']
        rows.append(dict(exp=r['exp'], instance=r['inst'], n=inst['n'], m=inst['m'], C=inst['C'], scheme=r['scheme'],
                         policy=POL[r['rule']], method=r['method'], rep=r['rep'], f=round(r['f'], 6), v=round(r['v'], 6),
                         T_h=round(Z['T'] / 60, 3), S_h=round(Z['S'] / 60, 3), E_TL=round(Z['E'], 2), O_h=round(Z['O'] / 60, 3),
                         nLate=Z['nLate'], nOrders=len(inst['orders']), meanLate_h=round(Z['meanLate'] / 60, 2),
                         maxLate_h=round(Z['maxLate'] / 60, 2), nDI=Z['nDI'], nPause=Z['nPause'], nOT=Z['nOT'],
                         nightDI_h=round(Z['nightDI'] / 60, 3), wait_h=round(Z['wait'] / 60, 2), Cmax_d=round(Z['Cmax'] / DAY, 3),
                         CO2_kg=round(Z['CO2'], 1), time_s=round(r['time'], 2), evals=r.get('evals', 0), optimal=r.get('optimal', ''),
                         RT_h=round(inst['rng']['T'] / 60, 2), idealT_h=round(inst['ide']['T'] / 60, 2),
                         idealT0_h=round(inst.get('ide0', inst['ide'])['T'] / 60, 2),
                         **{k: inst.get('meta', {}).get(k, '') for k in ('C_f', 'nf', 'lmax', 'delta', 'tariff', 'setup')}))
    for (nm, pol, rep), tr in sorted(D5_OUT.items(), key=lambda x: (x[0][0], str(x[0][1]), x[0][2])):
        inst = INSTREG.get(nm)
        for mk, fv, Z in tr:
            rows.append(dict(exp='D5', instance=nm, n=inst['n'] if inst else '', m=inst['m'] if inst else '',
                             C=inst['C'] if inst else '', scheme=D5_SCHEME, policy=POL[pol], method=f'GP-RKO@{mk}', rep=rep,
                             f=round(fv, 6), T_h=round(Z['T'] / 60, 3), S_h=round(Z['S'] / 60, 3), O_h=round(Z['O'] / 60, 3),
                             nLate=Z['nLate'], nightDI_h=round(Z['nightDI'] / 60, 3), evals=mk,
                             **{k: (inst or {}).get('meta', {}).get(k, '') for k in ('C_f', 'nf', 'lmax', 'delta', 'tariff', 'setup')}))
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def better(fv, rv, best):
    """lexicographic comparison (priority 1-2 scalar f, then restoration r) with ulp ties."""
    if best is None: return True
    if fv < best['f'] - ftol(best['f']): return True
    return fv <= best['f'] + ftol(best['f']) and rv < best['r'] - ftol(best['r'])

def ties(fv, rv, b):
    """True if (f, r) ties with the best-known (b) at all three priorities."""
    return fv <= b['f'] + ftol(b['f']) and rv <= b['r'] + ftol(b['r'])

def bks(inst, scheme, rule):
    """best-known schedule: every schedule found for the instance (any method, scheme or
       policy) re-evaluated under this scheme and policy; the lexicographic minimum of the
       scalar f (priorities 1-2) and the restoration term (priority 3).  Cached until the
       pool changes."""
    c = inst.setdefault('_bks', {})
    if (scheme, rule) in c: return c[(scheme, rule)]
    scal = scheme_list([scheme])[scheme]; f = make_scalar(scal[0], inst['ide'], inst['rng'], w=scal[1], tau=scal[2])
    best = None
    for keys, src in pool_keys(inst):
        Z = evaluate_keys(keys, inst, rule); fv = f(Z); rv = restoration(Z, inst['ide'], inst['rng'])
        if better(fv, rv, best): best = dict(f=fv, r=rv, v=value(scal, fv, inst), Z=Z, keys=keys, src=src)
    c[(scheme, rule)] = best
    return best

def night_min(inst, scheme, pol):
    """schedule with the least night drawing-in among the pooled schedules that tie with the
       best known for this scheme and policy at all three priorities (reported indicator)."""
    c = inst.setdefault('_nm', {})
    if (scheme, pol) in c: return c[(scheme, pol)]
    b = bks(inst, scheme, pol); scal = scheme_list([scheme])[scheme]
    f = make_scalar(scal[0], inst['ide'], inst['rng'], w=scal[1], tau=scal[2]); best = None
    for keys, _ in pool_keys(inst):
        Z = evaluate_keys(keys, inst, pol)
        if ties(f(Z), restoration(Z, inst['ide'], inst['rng']), b) and (best is None or Z['nightDI'] < best['Z']['nightDI']):
            best = dict(Z=Z, keys=keys)
    c[(scheme, pol)] = best
    return best

def tie_seeds(inst, sn, f, pol, b, by_r=False):
    """seeds for a constrained stage: the best known and further pooled schedules with the
       same achievement (priorities 1-2; with by_r also the same restoration value)."""
    seeds = [b['keys']]
    for keys, _ in pool_keys(inst):
        if len(seeds) >= RKO_CFG['psize'] - 1: break
        if keys is b['keys']: continue
        Z = evaluate_keys(keys, inst, pol); fv = f(Z)
        ok = ties(fv, restoration(Z, inst['ide'], inst['rng']), b) if by_r else fv <= b['f'] + ftol(b['f'])
        if ok: seeds.append(keys)
    return seeds

def task_restore(args):
    """worker: priority 3 (efficiency restoration): least sum of normalised objectives among
       schedules whose achievement at priorities 1-2 is not worse than the best known."""
    inst, scheme, scal, pol, fstar, seeds, budget, seed = args
    f = make_scalar(scal[0], inst['ide'], inst['rng'], w=scal[1], tau=scal[2]); tol = ftol(fstar)
    def fit(k):
        Z = evaluate_keys(k, inst, pol); fv = f(Z)
        return restoration(Z, inst['ide'], inst['rng']) if fv <= fstar + tol else 1e6 + (fv - fstar)
    pool, ev = rko(inst, fit, budget, seed, seeds_keys=seeds)
    return inst['name'], pol, [k for fv, k in pool[:3] if fv < 1e6]

def restore_stage(insts, plan, budget, label, ckpt=None):
    """priority 3 for every scheme and policy of the plan [(scheme, policies-of-instance)]."""
    tasks = []
    for inst in insts:
        for sn, pf in plan:
            sc = scheme_list([sn])[sn]; f = make_scalar(sc[0], inst['ide'], inst['rng'], w=sc[1], tau=sc[2])
            for pol in pf(inst):
                b = bks(inst, sn, pol)
                tasks.append((light(inst), sn, sc, pol, b['f'], tie_seeds(inst, sn, f, pol, b), budget, 1))
    by = {i['name']: i for i in insts}
    for nm, pol, kl in pmap(task_restore, tasks, label, ckpt=ckpt):
        for k in kl: add_pool(by[nm], pol, k, 'restore')

def task_night(args):
    """worker: night indicator for a counterfactual policy: least night drawing-in (00-08)
       among schedules that tie with the best known at all three priorities."""
    inst, scheme, scal, pol, fstar, rstar, seeds, budget, seed = args
    f = make_scalar(scal[0], inst['ide'], inst['rng'], w=scal[1], tau=scal[2]); b = dict(f=fstar, r=rstar)
    def fit(k):
        Z = evaluate_keys(k, inst, pol); fv = f(Z); rv = restoration(Z, inst['ide'], inst['rng'])
        if ties(fv, rv, b): return Z['nightDI'] / 60.0
        return 1e6 + max(0.0, fv - fstar) + max(0.0, rv - rstar)
    pool, ev = rko(inst, fit, budget, seed, seeds_keys=seeds)
    return inst['name'], pol, [k for fv, k in pool[:3] if fv < 1e6]

def stage3(insts, schemes, budget, label, ckpt=None):
    """night stage for the counterfactual policies: GP-RKO minimising night drawing-in subject
       to the best-known achievement at all three priorities, seeded with the tying schedules."""
    tasks = []
    for inst in insts:
        for sn in schemes:
            sc = scheme_list([sn])[sn]; f = make_scalar(sc[0], inst['ide'], inst['rng'], w=sc[1], tau=sc[2])
            for pol in pols_of(inst):
                if pol is True: continue
                b = bks(inst, sn, pol)
                tasks.append((light(inst), sn, sc, pol, b['f'], b['r'], tie_seeds(inst, sn, f, pol, b, by_r=True), budget, 1))
    by = {i['name']: i for i in insts}
    for nm, pol, kl in pmap(task_night, tasks, label, ckpt=ckpt):
        for k in kl: add_pool(by[nm], pol, k, 'night-min')

def runs_of(exp, inst, scheme, rule, method):
    return [r for r in RESULTS if r['exp'] == exp and r['inst'] == inst and r['scheme'] == scheme
            and r['rule'] == rule and r['method'] == method]

def fmtZ(Z):
    return (f"T={Z['T']/60:7.1f}h [late {Z['nLate']}, mean {Z['meanLate']/60:4.1f}h, max {Z['maxLate']/60:4.1f}h] "
            f"S={Z['S']/60:6.1f}h O={Z['O']/60:5.1f}h E={Z['E']:8.0f}TL | nDI={Z['nDI']:3d} "
            f"pause={Z['nPause']:2d} OT={Z['nOT']:2d} nightDI={Z['nightDI']/60:5.1f}h wait={Z['wait']/60:6.1f}h Cmax={Z['Cmax']/DAY:5.2f}d")

# =============================================================== 13. EXPERIMENTS
def heuristic_tasks(insts, schemes, policies, methods, reps, cls_of, budget=None):
    tasks = []
    for inst in insts:
        L = light(inst); bud = budget or BUDGET[cls_of(inst)]
        pols = policies(inst) if callable(policies) else policies
        for sn, sc in scheme_list(schemes).items():
            for rule in pols:
                tasks.append((L, sn, sc, rule, 'RULES', 0, bud, None))
                for mth in methods:
                    if mth == 'GP-RKO-w1e3' and not is_lex(sc[0]): continue
                    for rp in reps:
                        tasks.append((L, sn, sc, rule, mth, rp, bud, ABLATIONS.get(mth)))
    return tasks

def do_payoffs(insts, cls_of, budget=None, seeds=None):
    res = pmap(task_payoff, [(light(i), budget or PAY_BUDGET[cls_of(i)], seeds or PAY_SEEDS) for i in insts], 'payoff tables')
    by = {i['name']: i for i in insts}
    for nm, ide, nad, rng, kp in res:
        by[nm].update(ide=ide, nad=nad, rng=rng, ide0=dict(ide)); by[nm]['kpool'] = kp; by[nm].pop('_bks', None)

def print_payoff(inst):
    u = lambda i, v: f"{v/60:.1f}h" if i != 'E' else f"{v:.0f}TL"
    print(f"   {inst['name']:<10} n={inst['n']:3d} m={inst['m']:2d} C={inst['C']} orders={len(inst['orders']):2d} "
          f"| ideal " + " ".join(f"{i}={u(i, inst['ide'][i])}" for i in OBJS) +
          " | nadir " + " ".join(f"{i}={u(i, inst['nad'][i])}" for i in OBJS))

def refit_all(insts, label):
    """a posteriori ideal update; prints how much the payoff-table ideal was improved."""
    g = {i['name']: refit_ideal(i) for i in insts}
    gT = [g[i['name']]['T'] / 60 for i in insts]; gTr = [100 * g[i['name']]['T'] / i['rng']['T'] for i in insts]
    gS = [g[i['name']]['S'] / 60 for i in insts]
    print(f"   {label}: ideal improved after all runs in {sum(1 for x in gT if x > 1e-9)}/{len(insts)} instances; "
          f"T by mean {mean_(gT):.1f} h (max {max(gT):.1f} h, mean {mean_(gTr):.2f}% of R_T), S by mean {mean_(gS):.1f} h; "
          f"goals and attainments recomputed")
    return g

def method_table(exp, insts, schemes, policies, methods, title):
    """excess attainment over the best-known schedule (in % of the ranges, and in hours of
       tardiness and drawing-in), hit rate, time, Friedman ranks and Wilcoxon tests."""
    print("\n" + "=" * 118 + f"\n{title}\n" + "=" * 118)
    print("  dv = attainment minus best-known attainment, in % of the objective ranges (lexicographic: % of the tardiness range);"
          "\n  dT, dS = excess tardiness and drawing-in time over the best-known schedule (h); hit = runs reaching the best known")
    blocks = []; per = {mm: dict(dv=[], hit=0, nr=0, t=[], dT=[], dS=[]) for mm in methods}
    for inst in insts:
        for sn in schemes:
            for rule in policies:
                b = bks(inst, sn, rule); row = []; tol = ftol(b['f'])
                for mm in methods:
                    rr = runs_of(exp, inst['name'], sn, rule, mm)
                    if not rr: row.append(float('nan')); continue
                    dv = [100 * (r['v'] - b['v']) for r in rr]; P = per[mm]
                    P['dv'] += dv; P['hit'] += sum(1 for r in rr if r['f'] <= b['f'] + tol); P['nr'] += len(dv)
                    P['dT'] += [(r['Z']['T'] - b['Z']['T']) / 60 for r in rr]; P['dS'] += [(r['Z']['S'] - b['Z']['S']) / 60 for r in rr]
                    P['t'] += [r['time'] for r in rr]; row.append(sum(dv) / len(dv))
                if all(x == x for x in row): blocks.append(row)
    print(f"  {'method':<16}{'runs':>6}{'mean dv':>10}{'sd':>8}{'max dv':>9}{'hit %':>8}{'dT h':>8}{'dS h':>8}{'s/run':>8}")
    for mm in methods:
        P = per[mm]
        if not P['nr']: continue
        _, mu, sd = mstats(P['dv'])
        print(f"  {mm:<16}{P['nr']:>6}{mu:>10.3f}{sd:>8.3f}{max(P['dv']):>9.3f}{100*P['hit']/P['nr']:>8.1f}"
              f"{mean_(P['dT']):>8.1f}{mean_(P['dS']):>8.1f}{sum(P['t'])/len(P['t']):>8.1f}")
    if len(blocks) >= 3 and len(methods) >= 2:
        mr, chi2, pv = friedman(blocks)
        print(f"  Friedman over {len(blocks)} blocks (instance x scheme x policy): chi2={chi2:.2f}, p={pv:.2e}")
        print("  mean ranks: " + ", ".join(f"{mm} {r:.2f}" for mm, r in zip(methods, mr)))
        if 'GP-RKO' in methods:
            j0 = methods.index('GP-RKO'); others = [j for j in range(len(methods)) if j != j0]
            ps = [wilcoxon([b[j0] for b in blocks], [b[j] for b in blocks]) for j in others]; ad = holm(ps)
            print("  Wilcoxon GP-RKO vs (Holm-adjusted p): " +
                  ", ".join(f"{methods[j]} {a:.2e}" for j, a in zip(others, ad)))

def lt(rs):
    no = sum(r['no'] for r in rs)
    return f"{100*sum(r['late1'] for r in rs)/no:.0f}/{100*sum(r['late0'] for r in rs)/no:.0f}" if no else ''

def price_rows(insts, scheme):
    """night rule vs night crew (and equal crew-hours) on the best-known schedules."""
    rows = []
    pct = lambda a, b: 100 * (a - b) / b if b > 1e-9 else (0.0 if abs(a - b) < 1e-9 else float('nan'))
    for inst in insts:
        b1, b0 = bks(inst, scheme, True), bks(inst, scheme, False); Z1, Z0 = b1['Z'], b0['Z']
        n0 = night_min(inst, scheme, False)
        row = dict(inst=inst, dv=100 * (b1['v'] - b0['v']), dT=(Z1['T'] - Z0['T']) / 60, dTp=pct(Z1['T'], Z0['T']),
                   dS=(Z1['S'] - Z0['S']) / 60, dEp=pct(Z1['E'], Z0['E']), O=Z1['O'] / 60, nightDI=Z0['nightDI'] / 60,
                   nightMin=n0['Z']['nightDI'] / 60 if n0 else float('nan'), pause=Z1['nPause'],
                   bind=(b1['v'] - b0['v']) > 1e-9, bindT=(Z1['T'] - Z0['T']) > 0, Z1=Z1, Z0=Z0, T0=Z0['T'] / 60,
                   late1=Z1['nLate'], late0=Z0['nLate'], no=len(inst['orders']), eh=False)
        if eh_crews(inst['C']) and 'EH' in inst.get('kpool', {}):
            be = bks(inst, scheme, 'EH'); Ze = be['Z']; ne = night_min(inst, scheme, 'EH')
            row.update(eh=True, ZE=Ze, tim=(Z1['T'] - Ze['T']) / 60, cap=(Ze['T'] - Z0['T']) / 60,
                       timS=(Z1['S'] - Ze['S']) / 60, capS=(Ze['S'] - Z0['S']) / 60,
                       dvt=100 * (b1['v'] - be['v']), dvc=100 * (be['v'] - b0['v']),
                       nightEH=ne['Z']['nightDI'] / 60 if ne else float('nan'), lateE=Ze['nLate'])
        rows.append(row)
    return rows

D3HDR = (f"  {'scheme':<7}{'factor':<7}{'level':>6}{'n':>4}{'dv%':>7}{'dT h [95% CI]':>22}{'dT%':>6}{'med%':>6}{'dS h':>7}"
         f"{'late%':>8}{'O h':>6}{'pause':>6}{'bind':>6}{'bindT':>6}{'night h':>9}{'min':>7}")
def pct_agg(rs):
    """price in percent as a ratio of means: 100 * sum(dT) / sum(T of the night crew)."""
    den = sum(r['T0'] for r in rs)
    return 100 * sum(r['dT'] for r in rs) / den if den > 1e-9 else float('nan')

def prow(sn, fk, lv, rs):
    g = lambda key: [r[key] for r in rs]
    print(f"  {sn:<7}{fk:<7}{str(lv):>6}{len(rs):>4}{mean_(g('dv')):>7.2f}{fci(g('dT')):>22}{pct_agg(rs):>6.1f}{median_(g('dTp')):>6.1f}"
          f"{mean_(g('dS')):>7.1f}{lt(rs):>8}{mean_(g('O')):>6.1f}{mean_(g('pause')):>6.1f}"
          f"{100*sum(r['bind'] for r in rs)/len(rs):>6.0f}{100*sum(r['bindT'] for r in rs)/len(rs):>6.0f}"
          f"{mean_(g('nightDI')):>9.1f}{mean_(g('nightMin')):>7.1f}")

def factor_tests(rows, sn, key='dT'):
    """level differences of a factor, blocked by the other factors and the seed (instances of
       a block share the order book; C and delta change nothing else).  Two levels: Wilcoxon
       signed-rank; three levels: Friedman and pairwise Wilcoxon with Holm's correction."""
    for fk, levels in D3_FACTORS.items():
        if len(levels) < 2: continue
        oth = [k for k in D3_FACTORS if k != fk]; blk = {}
        for r in rows:
            mt = r['inst']['meta']; b = tuple(mt[FMETA.get(k, k)] for k in oth) + (r['inst']['seed'],)
            blk.setdefault(b, {})[mt[FMETA.get(fk, fk)]] = r[key]
        B = [[d[lv] for lv in levels] for d in blk.values() if all(lv in d for lv in levels)]
        if len(B) < 3: continue
        agg = median_ if key == 'dTp' else mean_            # per-instance percentages: medians
        means = (" / ".join(f"{lv}: {agg([b[j] for b in B]):.1f}" for j, lv in enumerate(levels))
                 + (" (medians)" if key == 'dTp' else ''))
        if len(levels) == 2:
            pv = wilcoxon([b[0] for b in B], [b[1] for b in B]); print(f"    {sn:<6} {key:<4} {fk:<6} {means}  (blocks {len(B)}) Wilcoxon p={pv:.3f}")
        else:
            _, chi2, pv = friedman(B); pairs = [(a, c) for a in range(len(levels)) for c in range(a + 1, len(levels))]
            ad = holm([wilcoxon([b[a] for b in B], [b[c] for b in B]) for a, c in pairs])
            print(f"    {sn:<6} {key:<4} {fk:<6} {means}  (blocks {len(B)}) Friedman p={pv:.3f}; pairs (Holm) " +
                  ", ".join(f"{levels[a]}-{levels[c]} {q:.3f}" for (a, c), q in zip(pairs, ad)))

def eh_table(PR, label):
    print(f"  {label}")
    print(f"  {'scheme':<7}{'n':>4}{'total dT h':>22}{'redistrib. (rule-EH)':>24}{'p':>7}{'capacity (EH-crew)':>24}{'p':>7}"
          f"{'redist. sh.':>14}{'dS rule-EH':>11}{'night crew':>11}{'night EH':>9}")
    for sn, rows in PR.items():
        rs = [r for r in rows if r['eh']]
        if not rs: continue
        tot, tim, cap = [r['dT'] for r in rs], [r['tim'] for r in rs], [r['cap'] for r in rs]
        sh = 100 * mean_(tim) / mean_(tot) if abs(mean_(tot)) > 1e-9 else float('nan')
        print(f"  {sn:<7}{len(rs):>4}{fci(tot):>22}{fci(tim):>24}{p0(tim):>7.3f}{fci(cap):>24}{p0(cap):>7.3f}{sh:>12.0f}%"
              f"{mean_([r['timS'] for r in rs]):>11.1f}{mean_([r['nightMin'] for r in rs]):>11.1f}{mean_([r['nightEH'] for r in rs]):>9.1f}")

def noise_table(exp, insts, schemes, pols_fn):
    """search noise per policy: spread of the replications and gap of single runs to the
       best-known schedule (hours of tardiness)."""
    print(f"  {'scheme':<7}{'policy':<18}{'rep spread T h (mean/median/max)':>34}{'run gap to BKS T h (mean/median)':>34}")
    for sn in schemes:
        for pol in (True, False, 'EH'):
            sp, gp = [], []
            for inst in insts:
                if pol not in pols_fn(inst): continue
                rr = runs_of(exp, inst['name'], sn, pol, 'GP-RKO')
                if len(rr) < 2: continue
                T = [r['Z']['T'] / 60 for r in rr]; sp.append(max(T) - min(T))
                bT = bks(inst, sn, pol)['Z']['T'] / 60; gp += [t - bT for t in T]
            if sp:
                print(f"  {sn:<7}{POL[pol]:<18}{mean_(sp):>14.1f}{median_(sp):>10.1f}{max(sp):>10.1f}{mean_(gp):>20.1f}{median_(gp):>14.1f}")

def exp_D1():
    print("\n#### D1  small instances, one crew: exact validation and exact price of the night rule")
    insts = [gen_instance('S', sd, name=f"S{sd}") for sd in D1_SEEDS]; INST = {i['name']: i for i in insts}; t0 = time.time()
    for inst in insts:
        payoff(inst, PAY_BUDGET['S'], PAY_SEEDS, use_cp=True, cp_tlim=CP_PAY_TLIM); print_payoff(inst)
    print(f"   payoff tables (with CP-SAT) in {time.time()-t0:.0f} s")
    cp_res = []
    for inst in insts:
        for sn, sc in scheme_list(D1_SCHEMES).items():
            for rule in (True, False):
                f = make_scalar(sc[0], inst['ide'], inst['rng'], w=sc[1], tau=sc[2]); rk = rule_keys(inst, f, rule)
                hint = min(rk.values(), key=lambda kk: f(evaluate_keys(kk, inst, rule)))
                c = cp_minimize(inst, sc[0], inst['ide'], inst['rng'], rule=rule, w=sc[1], tau=sc[2], tlim=CP_TLIM, hint=hint)
                if c is None: continue
                Z = evaluate_keys(c['keys'], inst, rule)
                cp_res.append(dict(inst=inst['name'], scheme=sn, rule=rule, method='CP-SAT', rep=0, f=f(Z), Z=Z,
                                   keys=c['keys'], time=c['time'], optimal=int(c['optimal'])))
    record('D1', cp_res, INST); print(f"   CP-SAT runs done ({time.time()-t0:.0f} s)")
    out = pmap(task_run, heuristic_tasks(insts, D1_SCHEMES, (True, False), D1_METHODS, REPS, lambda i: 'S'), 'D1 heuristics')
    record('D1', [r for lst in out for r in lst], INST)
    methods = ('CP-SAT',) + D1_METHODS + RULES
    method_table('D1', insts, D1_SCHEMES, (True, False), methods,
                 "TABLE 1  D1 small instances (one crew): methods against the best-known schedule")
    restore_stage(insts, [(sn, lambda i: (True, False)) for sn in D1_SCHEMES], RESTORE_BUDGET['S'], 'D1 restoration')
    stage3(insts, D1_SCHEMES, NIGHT_BUDGET['S'], 'D1 night minimisation')
    print("\n" + "=" * 118 + "\nTABLE 2  D1 exact price of the night rule; '*' marks pairs where CP-SAT proved both optima"
          " (all three priorities)\n" + "=" * 118)
    cpp = [(inst, sn, rl, r) for inst in insts for sn in D1_SCHEMES for rl in (True, False)
           for r in runs_of('D1', inst['name'], sn, rl, 'CP-SAT') if r['optimal']]
    nmatch = sum(1 for inst, sn, rl, r in cpp if bks(inst, sn, rl)['Z']['T'] == r['Z']['T'])
    print(f"  best-known tardiness equals the proven CP-SAT tardiness in {nmatch}/{len(cpp)} proven pairs (instance x scheme x policy)")
    print("  nightDI: night drawing-in of the best-known night-crew schedule; min: least night drawing-in among equally good schedules")
    print(f"  {'inst':<6}{'scheme':<10}{'proven':>7}{'dv%':>8}{'dT(h)':>8}{'dT%':>7}{'late':>8}{'dS(h)':>8}{'dE%':>7}{'O(h)':>6}"
          f"{'pause':>6}{'nightDI':>9}{'min':>6}")
    for sn in D1_SCHEMES:
        nb = npv = 0; rows = price_rows(insts, sn)
        for row in rows:
            nm = row['inst']['name']
            pr = all(any(r['optimal'] for r in runs_of('D1', nm, sn, rl, 'CP-SAT')) for rl in (True, False))
            npv += pr; nb += row['bind']
            print(f"  {nm:<6}{sn:<10}{'*' if pr else '':>7}{row['dv']:>8.2f}{row['dT']:>8.1f}{row['dTp']:>7.1f}"
                  f"{str(row['late1'])+'/'+str(row['late0']):>8}{row['dS']:>8.1f}{row['dEp']:>7.2f}"
                  f"{row['O']:>6.1f}{row['pause']:>6d}{row['nightDI']:>9.1f}{row['nightMin']:>6.1f}")
        print(f"  -> {sn}: rule binds in {nb}/{len(insts)} instances; both optima proven in {npv}/{len(insts)}; "
              f"mean dT {mean_([r['dT'] for r in rows]):.2f} h, mean dS {mean_([r['dS'] for r in rows]):.2f} h")
    return insts

def exp_D2():
    print("\n#### D2  scalability and algorithm comparison (night rule)")
    allin = []
    for cls in ('M', 'L'):
        insts = [gen_instance(cls, sd, name=f"{cls}{sd}") for sd in D2_SEEDS[cls]]; INST = {i['name']: i for i in insts}
        do_payoffs(insts, lambda i, c=cls: c)
        for i in insts: print_payoff(i)
        meths = D2_METHODS + (('GP-RKO-w1e3',) if (D2_LEGACY and cls == 'L') else ())
        out = pmap(task_run, heuristic_tasks(insts, D2_SCHEMES, (True,), meths, REPS, lambda i, c=cls: c), f'D2 {cls}')
        record('D2', [r for lst in out for r in lst], INST); refit_all(insts, f'D2 {cls}')
        method_table('D2', insts, D2_SCHEMES, (True,), D2_METHODS + RULES,
                     f"TABLE 3{cls}  D2 class {cls}: methods against the best-known schedule (equal evaluation budget)")
        if D2_LEGACY and cls == 'L':
            d = {'dv': [], 'dT': [], 'dS': []}
            for inst in insts:
                for sn, sc in scheme_list(D2_SCHEMES).items():
                    if not is_lex(sc[0]): continue
                    a = {r['rep']: r for r in runs_of('D2', inst['name'], sn, True, 'GP-RKO')}
                    b = {r['rep']: r for r in runs_of('D2', inst['name'], sn, True, 'GP-RKO-w1e3')}
                    for rp in set(a) & set(b):
                        d['dv'].append(100 * (b[rp]['v'] - a[rp]['v'])); d['dT'].append((b[rp]['Z']['T'] - a[rp]['Z']['T']) / 60)
                        d['dS'].append((b[rp]['Z']['S'] - a[rp]['Z']['S']) / 60)
            if d['dv']:
                print("\n  Sensitivity of the lexicographic weighting (class L, Lex schemes, paired runs, v6 weighting minus strict):")
                print(f"    dv {fci(d['dv'], 6, 3)} %, Wilcoxon p={p0(d['dv']):.3f};  dT {fci(d['dT'])} h;  dS {fci(d['dS'])} h  ({len(d['dv'])} pairs)")
        allin += insts
    return allin

def d3_instances():
    import itertools
    insts = []; keys = list(D3_FACTORS)
    for cell, combo in enumerate(itertools.product(*[D3_FACTORS[k] for k in keys])):
        fx = dict(zip(keys, combo))
        for sd in D3_SEEDS:
            inst = gen_instance('F', sd, name=f"F{cell:02d}-{sd}", m=D3_M, C=fx['C'], n_fam=fx['nf'], lmax=fx['lmax'],
                                delta=fx['delta'])
            inst['meta'] = dict(C_f=fx['C'], nf=fx['nf'], lmax=fx['lmax'], delta=fx['delta'], tariff='ToU', setup='structured')
            insts.append(inst)
    for sd in D3R_SEEDS:
        for dl in (1.0, 0.8):
            inst = gen_instance('F', sd, name=f"R{sd}-{dl:g}", m=D3_M, C=3, n_fam=8, lmax=300, delta=dl, setup_mode='general')
            inst['meta'] = dict(C_f=3, nf=8, lmax=300, delta=dl, tariff='ToU', setup='general'); insts.append(inst)
    return insts

def d5_subset(insts):
    return [i for i in insts if i['meta']['setup'] == 'structured' and i['meta']['lmax'] == D5_SELECT['lmax']
            and i['seed'] == D5_SELECT['seed']]

def price_snapshot(insts):
    """price (h) per instance and scheme, with the timing / capacity split where available."""
    snap = {}
    for sn in D3_SCHEMES:
        for r in price_rows(insts, sn):
            snap[(sn, r['inst']['name'])] = (r['dT'], r.get('tim', float('nan')), r.get('cap', float('nan')))
    return snap

def twin_key(inst):
    """instances of set F that differ only in the number of crews share their order book."""
    m = inst.get('meta', {})
    return (m['nf'], m['lmax'], m['delta'], inst['seed']) if m.get('setup') == 'structured' else None

def share_twins(insts, label=''):
    """twin pooling: every schedule found for an instance is also a candidate for its twins
       (same order book, other number of crews); kept in kpool['twin'], rebuilt on demand."""
    groups = {}
    for i in insts:
        k = twin_key(i)
        if k is not None: groups.setdefault(k, []).append(i)
    nshared = 0
    for g in groups.values():
        if len(g) < 2: continue
        own = {i['name']: [(k, s) for pol, lst in i.get('kpool', {}).items() if pol != 'twin' for k, s in lst] for i in g}
        for i in g:
            i.setdefault('kpool', {})['twin'] = [x for j in g if j is not i for x in own[j['name']]]
            i.pop('_bks', None); i.pop('_nm', None); nshared += len(i['kpool']['twin'])
    if label: print(f"   {label}: twin pooling over {sum(1 for g in groups.values() if len(g) > 1)} groups ({nshared} shared schedules)")

def twin_of(insts, inst, C):
    k = twin_key(inst)
    return next((j for j in insts if twin_key(j) == k and j['meta']['C_f'] == C), None)

def twin_consistency(insts, scheme, from_runs):
    """EH with C = 3 and the night crew with C = 2 are the same problem (2 crews around the
       clock on the same order book).  from_runs: best tardiness of the runs of each instance
       before twin pooling; otherwise the best-known schedules."""
    rows = []
    for i3 in insts:
        if i3['meta'].get('setup') != 'structured' or i3['meta']['C_f'] != 3: continue
        i2 = twin_of(insts, i3, 2)
        if i2 is None: continue
        if from_runs:
            a = [r['Z']['T'] for r in RESULTS if r['exp'] == 'D3' and r['inst'] == i2['name'] and r['scheme'] == scheme
                 and r['rule'] is False and r['method'].startswith('GP-RKO')]
            b = [r['Z']['T'] for r in RESULTS if r['exp'] == 'D3' and r['inst'] == i3['name'] and r['scheme'] == scheme
                 and r['rule'] == 'EH' and r['method'].startswith('GP-RKO')]
            if not a or not b: continue
            rows.append((min(a) / 60, min(b) / 60))
        else:
            rows.append((bks(i2, scheme, False)['Z']['T'] / 60, bks(i3, scheme, 'EH')['Z']['T'] / 60))
    if not rows: return
    d = [a - b for a, b in rows]
    print(f"  {scheme:<7}{'runs before twin pooling' if from_runs else 'best-known after pooling':<28}{len(rows):>4}"
          f"{mean_([a for a, _ in rows]):>12.1f}{mean_([b for _, b in rows]):>12.1f}{fci(d):>26}"
          f"{sum(1 for x in d if x > 1e-6):>11}{sum(1 for x in d if x < -1e-6):>13}")

def daycap_table(insts):
    """day capacity (Lex0): tardiness of the night rule with C, C+1, C+2 and unlimited crews
       against the night crew with C = 3 crews, on the best-known schedules."""
    print(f"  {'set':<14}{'n':>4}" + "".join(f"{h:>24}" for h in ('rule C=3', 'rule C=4', 'rule C=5', 'rule C=inf'))
          + f"{'EH':>20}   (difference to the night crew, h [95% CI]; count rule <= crew)")
    for lab, S in (('F, C=3', [i for i in insts if i['meta'].get('setup') == 'structured' and i['meta']['C_f'] == 3]),
                   ('R', [i for i in insts if i['meta'].get('setup') == 'general'])):
        if not S: continue
        cols = {'rule C=3': [], 'rule C=4': [], 'rule C=5': [], 'rule C=inf': [], 'EH': []}
        for i in S:
            T0 = bks(i, 'Lex0', False)['Z']['T'] / 60
            for key, pol in (('rule C=3', True), ('rule C=4', 'R+1'), ('rule C=5', 'R+2'), ('EH', 'EH')):
                cols[key].append(bks(i, 'Lex0', pol)['Z']['T'] / 60 - T0)
            ti = twin_of(insts, i, 'inf')
            if ti is not None: cols['rule C=inf'].append(bks(ti, 'Lex0', True)['Z']['T'] / 60 - T0)
        cell = lambda v: (f"{fci(v)} ({sum(1 for x in v if x <= 1e-6)})" if v else '--')
        print(f"  {lab:<14}{len(S):>4}" + "".join(f"{cell(cols[k]):>24}" for k in ('rule C=3', 'rule C=4', 'rule C=5', 'rule C=inf'))
              + f"{cell(cols['EH']):>20}")

def snap_table(pre, post, main, rob, insts, la, lb):
    print(f"  {'scheme':<7}{'set':<22}{'n':>4}{la + ' [95% CI]':>24}{lb + ' [95% CI]':>24}{'change':>9}"
          f"{'redistrib. ' + la[:3] + '/' + lb[:3]:>25}{'capacity ' + la[:3] + '/' + lb[:3]:>24}")
    for sn in D3_SCHEMES:
        for lab, S in (('structured', main), ('general matrix', rob), ('D5 subset', d5_subset(insts))):
            nm = [i['name'] for i in S]
            if not nm: continue
            b = [pre[(sn, x)][0] for x in nm]; a = [post[(sn, x)][0] for x in nm]
            tb = [pre[(sn, x)][1] for x in nm]; ta = [post[(sn, x)][1] for x in nm]
            cb = [pre[(sn, x)][2] for x in nm]; ca = [post[(sn, x)][2] for x in nm]
            print(f"  {sn:<7}{lab:<22}{len(nm):>4}{fci(b):>24}{fci(a):>24}{mean_(a) - mean_(b):>9.1f}"
                  f"{mean_(tb):>12.1f}/{mean_(ta):<12.1f}{mean_(cb):>12.1f}/{mean_(ca):<11.1f}")

def polish(insts, plan, ckpt, label='D3 polishing'):
    """cross-seeded polishing: per instance, scheme and policy one GP-RKO run and one SEED-LS
       run started from the best distinct pooled schedules under that scheme and policy.
       plan: list of (scheme, policies-of-instance function)."""
    tasks = []
    for inst in insts:
        for sn, pf in plan:
            sc = scheme_list([sn])[sn]; f = make_scalar(sc[0], inst['ide'], inst['rng'], w=sc[1], tau=sc[2])
            for pol in pf(inst):
                ranked = sorted(((f(evaluate_keys(k, inst, pol)), k) for k, _ in pool_keys(inst)), key=lambda x: x[0])
                seeds = [k for _, k in ranked[:RKO_CFG['psize']]]
                tasks.append((light(inst), sn, sc, pol, seeds, D3_POLISH_BUDGET, 101, 'GP-RKO'))
                tasks.append((light(inst), sn, sc, pol, seeds[:1], D3_SEEDLS_BUDGET, 102, 'SEED-LS'))
    tasks.sort(key=lambda t: -t[0]['n'] * t[5])                 # long tasks first (better load balance)
    return pmap(task_polish, tasks, label, ckpt=ckpt)

def load_d3_checkpoint():
    """final state of D3 (instances with their pools and ideals), for D5 in a separate run."""
    path = f"{BASE_PREFIX}_D3_ckpt.pkl"
    if not os.path.exists(path): return None
    st = load_pickle(path)
    if st.get('stage', 0) < 6: return None
    insts = st['insts']; share_twins(insts)
    for i in insts: INSTREG[i['name']] = i
    return insts

def exp_D3():
    print("\n#### D3  price map of the night rule (mill size, 50 looms; night rule, night crew, equal crew-hours;")
    print("      day capacity C+1, C+2 under Lex0; Lex0 first, then Lex10 with the updated ideal)")
    CK = f"{BASE_PREFIX}_D3_ckpt"; st = None; from_v8 = False
    if RESUME and os.path.exists(CK + '.pkl'):
        st = load_pickle(CK + '.pkl'); print(f"   resumed from {CK}.pkl (stage {st['stage']})")
    elif RESUME and os.path.exists(V8_D3_CKPT):
        # v8 stages: 1 payoff, 2 Lex0, 3 Lex10, 4 polishing, 5 night stage.  The v8 runs and the
        # polishing are kept; the restoration and night stages of v9 follow (v9 stages 5 and 6).
        st = load_pickle(V8_D3_CKPT); from_v8 = True
        print(f"   resumed from the v8 checkpoint {V8_D3_CKPT} (v8 stage {st['stage']}); runs and polishing are kept")
        st['stage'] = min(st['stage'], 4)
    elif os.path.exists(CK + '.pkl'):
        print(f"   note: {CK}.pkl exists but --resume was not given; starting afresh")
        for ext in ('.pkl', '.runs0.pkl', '.runs10.pkl', '.polish.pkl', '.restore.pkl', '.night.pkl'):
            if os.path.exists(CK + ext): os.remove(CK + ext)
    def save(stage):
        save_pickle(dict(stage=stage, insts=[light(i) | {'kpool': {p: l for p, l in i.get('kpool', {}).items() if p != 'twin'}}
                                             for i in insts],
                         results=[r for r in RESULTS if r['exp'] == 'D3']), CK + '.pkl')
    if st:
        insts = st['insts']; RESULTS.extend(st['results'])
    else:
        insts = d3_instances(); do_payoffs(insts, lambda i: 'L', D3_PAY_BUDGET, D3_PAY_SEEDS); save(1)
    INST = {i['name']: i for i in insts}; INSTREG.update(INST); stage = st['stage'] if st else 1
    if stage >= 2: share_twins(insts)
    lex0 = [s for s in D3_SCHEMES if s == 'Lex0']; others = [s for s in D3_SCHEMES if s != 'Lex0']
    if stage < 2:
        out = pmap(task_run, heuristic_tasks(insts, lex0, pols_lex0, ('GP-RKO',), D3_REPS, lambda i: 'L', D3_BUDGET),
                   'D3 runs Lex0', ckpt=CK + '.runs0.pkl')
        record('D3', [r for lst in out for r in lst], INST)
        share_twins(insts, 'D3 Lex0'); refit_all(insts, 'D3 after Lex0'); save(2)
    if stage < 3 and others:
        out = pmap(task_run, heuristic_tasks(insts, others, pols_of, ('GP-RKO',), D3_REPS, lambda i: 'L', D3_BUDGET),
                   'D3 runs ' + ','.join(others), ckpt=CK + '.runs10.pkl')
        record('D3', [r for lst in out for r in lst], INST)
        share_twins(insts, 'D3 ' + ','.join(others)); refit_all(insts, 'D3 after ' + ','.join(others)); save(3)
    main = [i for i in insts if i['meta']['setup'] == 'structured']; rob = [i for i in insts if i['meta']['setup'] == 'general']
    pre = price_snapshot(insts); polished_now = stage < 4
    if stage < 4:
        plan = [(sn, pols_lex0 if sn == 'Lex0' else pols_of) for sn in D3_SCHEMES]
        out = polish(insts, plan, CK + '.polish.pkl')
        record('D3', [r for lst in out for r in lst], INST)
        share_twins(insts, 'D3 polishing'); refit_all(insts, 'D3 after polishing'); save(4)
    post = price_snapshot(insts)
    if not polished_now:
        print("\n   TABLE 4d (effect of the polishing) is not available after a resume past the polishing stage")
    else:
      print("\n" + "=" * 118 + "\nTABLE 4d  D3 effect of the cross-seeded polishing on the price (h of tardiness; GP-RKO "
          f"{D3_POLISH_BUDGET} and SEED-LS {D3_SEEDLS_BUDGET} evaluations per instance, scheme and policy)\n" + "=" * 118)
      snap_table(pre, post, main, rob, insts, 'before', 'after')
    # priority 3: efficiency restoration, then the night indicator of the counterfactual policies
    if stage < 5:
        plan = [(sn, pols_lex0 if sn == 'Lex0' else pols_of) for sn in D3_SCHEMES]
        restore_stage(insts, plan, RESTORE_BUDGET['L'], 'D3 restoration', ckpt=CK + '.restore.pkl')
        share_twins(insts, 'D3 restoration'); refit_all(insts, 'D3 after restoration'); save(5)
    rest = price_snapshot(insts)
    print("\n" + "=" * 118 + "\nTABLE 4g  D3 effect of the restoration stage on the price (h of tardiness; GP-RKO "
          f"{RESTORE_BUDGET['L']} evaluations per instance, scheme and policy)\n" + "=" * 118)
    print("  before: best-known schedules with ties broken by the restoration term over the pool; after: after the stage")
    snap_table(post, rest, main, rob, insts, 'before', 'after')
    if stage < 6:
        stage3(insts, D3_SCHEMES, NIGHT_BUDGET['L'], 'D3 night minimisation', ckpt=CK + '.night.pkl')
        share_twins(insts); save(6)
    fin = price_snapshot(insts)
    print("\n  price after the night stage (final, h):")
    snap_table(rest, fin, main, rob, insts, 'restored', 'final')
    print("\n" + "=" * 118 + "\nTABLE 4  D3 price of the night rule by factor level (best-known schedules, night rule minus night crew)\n" + "=" * 118)
    print("  dv: attainment difference (% of ranges; lexicographic: in tardiness ranges); dT: tardiness (h, mean and 95%")
    print("  bootstrap interval over instances; %); dS: drawing-in time (h); late%: late orders, night rule / night crew;")
    print("  O: overtime after midnight (h); pause: paused drawing-ins; bind: attainment worse under the rule (%);")
    print("  bindT: tardiness higher under the rule (%); night h: night drawing-in of the best-known night-crew schedule;")
    print("  min: least night drawing-in among night-crew schedules that tie at all three priorities.")
    print("  dT%: ratio of means, 100 * sum(dT) / sum(T night crew); med%: median of the per-instance percentages.")
    print("  Lex0 minimises tardiness first (goal 0); under Lex10 both policies may stop at the tardiness goal, so the")
    print("  price there is the attainment difference dv and dS rather than dT.")
    PR = {}
    for sn in D3_SCHEMES:
        rows = price_rows(main, sn); PR[sn] = rows; print(D3HDR)
        for fk in D3_FACTORS:
            for lv in D3_FACTORS[fk]:
                rs = [r for r in rows if r['inst']['meta'][FMETA.get(fk, fk)] == lv]
                if rs: prow(sn, fk, lv, rs)
        prow(sn, 'all', '', rows); print()
    print("  robustness, general asymmetric changeover matrix:"); print(D3HDR)
    PRR = {sn: price_rows(rob, sn) for sn in D3_SCHEMES}
    for sn in D3_SCHEMES: prow(sn, 'general', '', PRR[sn])
    print("\n  factor tests on the price (blocked by the other factors and the seed):")
    for sn in D3_SCHEMES:
        for key in (('dT', 'dTp', 'dS') if sn == 'Lex0' else ('dv', 'dS', 'dT')): factor_tests(PR[sn], sn, key)
    print("\n" + "=" * 118 + "\nTABLE 4b  D3 decomposition of the price, C = 3: redistribution (rule - equal crew-hours) and capacity "
          "(equal crew-hours - night crew)\n" + "=" * 118)
    print("  hours of tardiness, mean [95% bootstrap interval]; p: Wilcoxon signed-rank test against zero; night: least night drawing-in (h)")
    print("  redistribution = losing the night shift but keeping 3 instead of 2 crews in the day and evening shifts")
    eh_table(PR, 'structured changeovers:'); eh_table(PRR, 'general changeover matrix:')
    print("  C = inf: the crew capacity does not bind, so its price is a pure timing price (Table 4, level inf)")
    print("\n" + "=" * 118 + "\nTABLE 4e  D3 consistency check: equal crew-hours with C = 3 and night crew with C = 2 are the same problem\n" + "=" * 118)
    print("  Lex0 only (under Lex10 the two instances have different tardiness goals); EH runs use other seeds")
    print(f"  {'scheme':<7}{'basis':<28}{'n':>4}{'crew C=2':>12}{'EH C=3':>12}{'crew - EH, h [95% CI]':>26}{'EH better':>11}{'crew better':>13}")
    for sn in [x for x in D3_SCHEMES if x == 'Lex0']:
        twin_consistency(insts, sn, True); twin_consistency(insts, sn, False)
    print("\n" + "=" * 118 + "\nTABLE 4f  D3 day capacity (Lex0): night rule with more day/evening crews against the night crew with C = 3\n" + "=" * 118)
    daycap_table(insts)
    print("\n" + "=" * 118 + "\nTABLE 4c  D3 search noise of GP-RKO per policy (tardiness, h)\n" + "=" * 118)
    noise_table('D3', insts, D3_SCHEMES, pols_lex0)
    print("\n" + "=" * 118 + "\nTABLE 5  D3 dispatching rules under the night rule: excess attainment over the best known (% of ranges)\n" + "=" * 118)
    print(f"  {'scheme':<9}" + "".join(f"{nm:>9}" for nm in RULES) + f"{'GP-RKO':>9}{'best rule dT h':>16}   best rule share")
    for sn in D3_SCHEMES:
        acc = {nm: [] for nm in RULES + ('GP-RKO',)}; wins = {nm: 0 for nm in RULES}; brT = []
        for inst in main:
            b = bks(inst, sn, True); best_rule = None
            for nm in RULES + ('GP-RKO',):
                rr = runs_of('D3', inst['name'], sn, True, nm)
                if rr:
                    d = min(100 * (r['v'] - b['v']) for r in rr); acc[nm].append(d)
                    if nm in RULES and (best_rule is None or d < best_rule[0]): best_rule = (d, nm, rr[0]['Z']['T'])
            if best_rule: wins[best_rule[1]] += 1; brT.append((best_rule[2] - b['Z']['T']) / 60)
        print(f"  {sn:<9}" + "".join(f"{mean_(v):>9.2f}" for v in acc.values()) + f"{mean_(brT):>9.1f} ({median_(brT):.1f})"
              "   " + ", ".join(f"{k} {v}" for k, v in wins.items() if v))
    return insts, PR

def exp_D5(insts=None, PR=None):
    print("\n#### D5  convergence of the search (Lex0, long anytime runs)")
    if insts is None:
        insts = load_d3_checkpoint()
        if insts is not None:
            print(f"   D3 instances and pools read from {BASE_PREFIX}_D3_ckpt.pkl")
            PR = {D5_SCHEME: price_rows([i for i in insts if i['meta']['setup'] == 'structured'], D5_SCHEME)}
    if insts is None:
        print("   no complete D3 checkpoint found: D5 runs on fresh instances without the D3 comparison")
        insts = d5_subset(d3_instances()); do_payoffs(insts, lambda i: 'L', D3_PAY_BUDGET, D3_PAY_SEEDS)
    sel = d5_subset(insts)
    for i in sel: INSTREG[i['name']] = i
    sc = scheme_list([D5_SCHEME])[D5_SCHEME]
    tasks = [(light(i), sc, pol, rp, D5_MARKS) for i in sel for pol in pols_of(i) for rp in D5_REPS]
    tasks.sort(key=lambda t: -t[0]['n'])
    for nm, pol, rep, tr in pmap(task_trace, tasks, 'D5 runs'): D5_OUT[(nm, pol, rep)] = tr
    def at(nm, pol, rep, mk):
        return next((Z, fv) for m_, fv, Z in D5_OUT[(nm, pol, rep)] if m_ == mk)
    def best(nm, pol, mk):
        return min((at(nm, pol, rp, mk) for rp in D5_REPS), key=lambda x: x[1])[0]
    have_d3 = PR is not None and any(i.get('kpool') for i in sel)
    # best-known tardiness per instance and policy: D5 runs and (if available) the pooled D3 schedules
    bestT = {}
    for i in sel:
        for pol in pols_of(i):
            v = min(Z['T'] for rp in D5_REPS for _, _, Z in D5_OUT[(i['name'], pol, rp)])
            if have_d3: v = min(v, bks(i, D5_SCHEME, pol)['Z']['T'])
            bestT[(i['name'], pol)] = v
    print("\n" + "=" * 118 + f"\nTABLE 7  D5 search convergence ({len(sel)} instances, {D5_SCHEME}, {len(D5_REPS)} replications): "
          "tardiness above the best known (h)\n" + "=" * 118)
    pols = [p for p in (True, False, 'EH') if any(p in pols_of(i) for i in sel)]
    print(f"  {'evals':>8}" + "".join(f"{POL[p] + ' single':>24}{'best of reps':>14}" for p in pols)
          + f"{'price single h':>22}{'price best h':>22}")
    conv = []
    for mk in D5_MARKS:
        row = dict(mk=mk, gap={}, pb=[], ps=[])
        line = f"  {mk:>8}"
        for p in pols:
            s_ = [at(i['name'], p, rp, mk)[0]['T'] / 60 - bestT[(i['name'], p)] / 60 for i in sel if p in pols_of(i) for rp in D5_REPS]
            b_ = [best(i['name'], p, mk)['T'] / 60 - bestT[(i['name'], p)] / 60 for i in sel if p in pols_of(i)]
            row['gap'][p] = (mean_(s_), mean_(b_)); line += f"{mean_(s_):>24.1f}{mean_(b_):>14.1f}"
        for i in sel:
            nm = i['name']; row['pb'].append((best(nm, True, mk)['T'] - best(nm, False, mk)['T']) / 60)
            row['ps'] += [(at(nm, True, rp, mk)[0]['T'] - at(nm, False, rp, mk)[0]['T']) / 60 for rp in D5_REPS]
        conv.append(row); print(line + f"{fci(row['ps']):>22}{fci(row['pb']):>22}")
    d3 = None
    if have_d3:
        names = {i['name'] for i in sel}; d3 = [r['dT'] for r in PR[D5_SCHEME] if r['inst']['name'] in names]
        print(f"\n  D3 price on the same instances (pooled and polished best-known schedules): {fci(d3)} h")
        print("  D5 against D3 (best tardiness of the five long runs minus the D3 best-known schedule, h; count D5 better):")
        for p in pols:
            dd = [(best(i['name'], p, D5_MARKS[-1])['T'] - bks(i, D5_SCHEME, p)['Z']['T']) / 60 for i in sel if p in pols_of(i)]
            print(f"    {POL[p]:<18}{fci(dd):>24}   {sum(1 for x in dd if x < -1e-6)}/{len(dd)}")
    return dict(conv=conv, d3=d3, pols=pols)

def exp_D6(d3insts=None):
    """goal-setting bias: the D3 protocol (three Lex runs per policy, Lex10 runs, twin pooling,
       polishing, restoration) on the D5 subset with the tardiness goal pinned at the
       payoff-table ideal, G_T = (1 + tau) z*_T for tau = 0 and 0.10, and no ideal update.
       Compared with D3 (goal 0 under Lex0, updated ideal) on the same instances."""
    print("\n#### D6  goal-setting bias: tardiness goal pinned at the payoff-table ideal (D5 subset)")
    if d3insts is None:
        d3insts = load_d3_checkpoint()
        if d3insts is not None: print(f"   D3 instances read from {BASE_PREFIX}_D3_ckpt.pkl")
    insts = d5_subset(d3_instances())
    for i in insts: i['name'] = 'P-' + i['name']
    INST = {i['name']: i for i in insts}; INSTREG.update(INST)
    do_payoffs(insts, lambda i: 'L', D3_PAY_BUDGET, D3_PAY_SEEDS)          # ideal = payoff table, never updated
    S0, S10 = 'Lex0P', 'Lex10P'
    for sn, lab in ((S0, 'D6 runs Lex0P'), (S10, 'D6 runs Lex10P')):
        out = pmap(task_run, heuristic_tasks(insts, [sn], pols_of, ('GP-RKO',), D3_REPS, lambda i: 'L', D3_BUDGET), lab)
        record('D6', [r for lst in out for r in lst], INST); share_twins(insts, lab)
    out = polish(insts, [(S0, pols_of), (S10, pols_of)], None, 'D6 polishing')
    record('D6', [r for lst in out for r in lst], INST); share_twins(insts, 'D6 polishing')
    restore_stage(insts, [(S0, pols_of), (S10, pols_of)], RESTORE_BUDGET['L'], 'D6 restoration'); share_twins(insts)
    print("\n" + "=" * 118 + "\nTABLE 8  D6 goal-setting bias: price with the tardiness goal pinned at the payoff-table ideal (P)"
          "\n          against D3 (Lex0: goal 0; ideal updated) on the same instances (h of tardiness)\n" + "=" * 118)
    d3 = {}
    if d3insts is not None:
        for i in d5_subset(d3insts): d3[i['name']] = i
        same = sum(1 for i in insts if i['name'][2:] in d3 and abs(i['ide0']['T'] - d3[i['name'][2:]]['ide0']['T']) < 1e-9)
        print(f"  payoff-table ideal of tardiness identical to that of D3 in {same}/{len(insts)} instances")
    rows = []
    for i in insts:
        nm0 = i['name'][2:]; j = d3.get(nm0)
        b1, b0 = bks(i, S0, True), bks(i, S0, False)
        row = dict(nm=nm0, P=(b1['Z']['T'] - b0['Z']['T']) / 60, T1P=b1['Z']['T'] / 60, T0P=b0['Z']['T'] / 60,
                   gT=i['ide']['T'] / 60, crewAtGoal=b0['Z']['T'] <= i['ide']['T'] + 1e-9,
                   dvP=100 * (bks(i, S10, True)['v'] - bks(i, S10, False)['v']))
        if j is not None:
            a1, a0 = bks(j, 'Lex0', True), bks(j, 'Lex0', False)
            row.update(D=(a1['Z']['T'] - a0['Z']['T']) / 60, T1D=a1['Z']['T'] / 60, T0D=a0['Z']['T'] / 60,
                       dvD=100 * (bks(j, 'Lex10', True)['v'] - bks(j, 'Lex10', False)['v']))
        rows.append(row)
        print(f"  {nm0:<8} goal {row['gT']:7.1f}  pinned: rule {row['T1P']:7.1f} crew {row['T0P']:7.1f} price {row['P']:6.1f}"
              f"{' (crew at goal)' if row['crewAtGoal'] else '':<15}" +
              (f"| D3: rule {row['T1D']:7.1f} crew {row['T0D']:7.1f} price {row['D']:6.1f}" if 'D' in row else ''))
    P = [r['P'] for r in rows]
    print(f"\n  Lex0, pinned goal: price {fci(P)} h; night-crew schedules at the pinned goal in "
          f"{sum(r['crewAtGoal'] for r in rows)}/{len(rows)} instances")
    if all('D' in r for r in rows):
        D = [r['D'] for r in rows]; dd = [r['P'] - r['D'] for r in rows]
        print(f"  Lex0, D3 (goal 0, updated ideal): price {fci(D)} h")
        print(f"  pinned minus D3: price {fci(dd)} h, Wilcoxon p={p0(dd):.3f}; relative {100 * (mean_(P) - mean_(D)) / mean_(D):.0f}%")
        print(f"  tardiness, pinned minus D3: night rule {fci([r['T1P'] - r['T1D'] for r in rows])} h, "
              f"night crew {fci([r['T0P'] - r['T0D'] for r in rows])} h")
        print(f"  Lex10 attainment difference (% of R_T): pinned {fci([r['dvP'] for r in rows], 6, 2)}, "
              f"D3 {fci([r['dvD'] for r in rows], 6, 2)}")
    return rows

def exp_D4():
    print("\n#### D4  case study: tolerance curve of the lexicographic GP (Lex0 first, then the other schemes)")
    inst = gen_instance('F', CASE['seed'], name='CASE', m=CASE['m'], C=CASE['C'], n_fam=CASE['n_fam'], delta=CASE['delta'])
    inst['meta'] = dict(C_f=CASE['C'], nf=CASE['n_fam'], lmax=L_MAX, delta=CASE['delta'], tariff='ToU', setup='structured')
    INST = {'CASE': inst}; do_payoffs([inst], lambda i: 'L'); print_payoff(inst)
    print(f"   {len(inst['orders'])} orders, {inst['n']} beams, families (h) " +
          ", ".join(f"{f}:{t/60:g}" for f, t in inst['fam_T'].items()))
    first = [s for s in D4_SCHEMES if s == 'Lex0']; rest = [s for s in D4_SCHEMES if s != 'Lex0']
    for part, lab in ((first, 'D4 runs Lex0'), (rest, 'D4 runs, other schemes')):
        if not part: continue
        out = pmap(task_run, heuristic_tasks([inst], part, pols_of, ('GP-RKO',), REPS, lambda i: 'L'), lab)
        record('D4', [r for lst in out for r in lst], INST); refit_all([inst], lab)
    restore_stage([inst], [(sn, pols_of) for sn in D4_SCHEMES], RESTORE_BUDGET['L'], 'D4 restoration')
    refit_all([inst], 'D4 after restoration')
    stage3([inst], D4_SCHEMES, D4_NIGHT_BUDGET, 'D4 night minimisation')
    print("\n" + "=" * 118 + "\nTABLE 6  D4 case: best-known schedules per scheme and policy\n" + "=" * 118)
    print("  counterfactual policies: nightDI of the best-known schedule; 'night min' = least night drawing-in among schedules")
    print("  that tie with it at all three priorities")
    B = {}
    for sn in D4_SCHEMES:
        for rule in pols_of(inst):
            b = bks(inst, sn, rule); B[(sn, rule)] = b
            rr = runs_of('D4', 'CASE', sn, rule, 'GP-RKO'); _, mu, sd = mstats([100 * (r['v'] - b['v']) for r in rr])
            nm = night_min(inst, sn, rule) if rule is not True else None
            if nm: B[(sn, rule, 'nm')] = nm
            extra = f" | night min {nm['Z']['nightDI']/60:5.1f}h" if nm else ''
            print(f"  {sn:<9}{POL[rule]:<17} {fmtZ(b['Z'])}{extra}  | GP-RKO dv mean {mu:.2f} sd {sd:.2f}")
    return inst, B

def reprice_D3():
    """electricity bill of the D3 best-known schedules under the current tariff (SEGMENTS_TOU),
       read from the final D3 checkpoint; the schedules do not depend on the tariff, because the
       bill is a reported indicator only."""
    print("\n#### D3 electricity bill of the best-known schedules (tariff of SEGMENTS_TOU)")
    insts = load_d3_checkpoint()
    if insts is None: print(f"   no complete {BASE_PREFIX}_D3_ckpt.pkl found"); return
    for i in insts:
        i['ecurve'] = energy_curve(i['tariff']); i.pop('_bks', None); i.pop('_nm', None)
    print(f"   prices (TL/kWh): " + ", ".join(f"{s[2]} {s[3]:.4f}" for s in SEGMENTS_TOU[1:4]))
    for lab, S in (('structured (F)', [i for i in insts if i['meta']['setup'] == 'structured']),
                   ('general (R)', [i for i in insts if i['meta']['setup'] == 'general'])):
        for sn in D3_SCHEMES:
            d, e0 = [], []
            for i in S:
                E1, E0 = bks(i, sn, True)['Z']['E'], bks(i, sn, False)['Z']['E']
                d.append(100 * (E1 - E0) / E0); e0.append(E0)
            print(f"  {lab:<16}{sn:<7} rule minus crew: mean {mean_(d):6.2f}%  [95% CI {fci(d, 6, 2)}]  "
                  f"max |.| {max(abs(x) for x in d):5.2f}%   mean bill of the night crew {mean_(e0):9.0f} TL")
        # spread of the bill over all best-known schedules (schemes x policies) of an instance
        sp = []
        for i in S:
            Es = [bks(i, sn, p)['Z']['E'] for sn in D3_SCHEMES for p in pols_of(i)]
            sp.append(100 * (max(Es) - min(Es)) / min(Es))
        print(f"  {lab:<16}spread of the bill over the best-known schedules of an instance: mean {mean_(sp):.2f}%, max {max(sp):.2f}%")

# =============================================================== 14. FIGURES
SHADE = {'D': '#FFF4DE', 'E': '#FDE9D9', 'N': '#DCE7F2'}
FS = 10                           # base font size of the figures
def _shade(ax, horizon):
    d = 0
    while d * DAY < horizon:
        for (a, b, _, _, _, sh) in SEGMENTS_TOU:
            if d * DAY + a < horizon:
                ax.axvspan((d * DAY + a) / 60, min(d * DAY + b, horizon) / 60, color=SHADE[sh], zorder=0, lw=0)
        d += 1

def crew_gantt(ax, inst, keys, pol, title, horizon):
    sch = decode(keys, inst, pol); cal = sch.rule; _shade(ax, horizon)
    for b, r in enumerate(sch.rec):
        k, ready, sig, s, ov, pz, st, en, c = r
        if s == 0: continue
        if pz:
            first = DAY * (sig // DAY + 1) - (DAY - CREW_END) - sig
            ax.barh(c, first / 60, left=sig / 60, height=0.6, color='#7f8c8d', edgecolor='white', lw=0.4, zorder=3)
            ax.barh(c, (DAY - CREW_END) / 60, left=(sig + first) / 60, height=0.25, color='white', edgecolor='#555', ls=':', lw=0.8, zorder=3)
            ax.barh(c, (s - first) / 60, left=(sig + first + DAY - CREW_END) / 60, height=0.6, color='#7f8c8d', edgecolor='white', lw=0.4, zorder=3)
        else:
            night = ov if cal else night_overlap(sig, sig + s)
            ax.barh(c, (s - night) / 60 if cal else s / 60, left=sig / 60, height=0.6, color='#7f8c8d', edgecolor='white', lw=0.4, zorder=3)
            if cal and ov > 0:
                ax.barh(c, ov / 60, left=(sig + s - ov) / 60, height=0.6, color='#c0392b', edgecolor='white', lw=0.4, zorder=3)
            if not cal and night > 0:
                d0 = sig
                while d0 < sig + s:
                    dd = d0 // DAY; lo, hi = max(sig, dd * DAY + CREW_END), min(sig + s, (dd + 1) * DAY)
                    if hi > lo: ax.barh(c, (hi - lo) / 60, left=lo / 60, height=0.6, color='#2c3e50', edgecolor='white', lw=0.4, zorder=4)
                    d0 = (dd + 1) * DAY
    ax.set_yticks(range(sch.C)); ax.set_yticklabels([f"crew {c+1}" for c in range(sch.C)], fontsize=FS)
    ax.tick_params(axis='x', labelsize=FS); ax.set_xlim(0, horizon / 60); ax.invert_yaxis(); ax.set_title(title, fontsize=FS + 1, loc='left')

def figures(D3PR=None, case=None, D5=None):
    import matplotlib
    if 'google.colab' not in sys.modules: matplotlib.use('Agg')
    import matplotlib.pyplot as plt, matplotlib.patches as mpatches
    if case:
        inst, B = case; sn = 'Lex10' if ('Lex10', True) in B else next(iter(B))[0]
        pols = [p for p in (True, False, 'EH') if (sn, p) in B]
        kk = {p: (B[(sn, p, 'nm')]['keys'] if (sn, p, 'nm') in B else B[(sn, p)]['keys']) for p in pols}
        hz = 0; lo = None
        for p in pols:
            rec = [r for r in decode(kk[p], inst, p).rec if r[3] > 0]
            if rec:
                hz = max(hz, max(r[2] + r[3] + (DAY - CREW_END if r[5] else 0) for r in rec)); l0 = min(r[2] for r in rec)
                lo = l0 if lo is None else min(lo, l0)
        hz = (hz // DAY + 1) * DAY; lo = ((lo or 0) // DAY) * DAY
        nC = {p: pol_crews(inst, p)[1] for p in pols}
        fig, axes = plt.subplots(len(pols), 1, figsize=(12, 1.5 + 0.55 * sum(nC.values()) + 0.4 * len(pols)), sharex=True,
                                 gridspec_kw=dict(height_ratios=[nC[p] + 0.8 for p in pols]))
        titles = {True: f"{sn}, night rule: no drawing-in crew 00-08", False: f"{sn}, counterfactual night crew (least night drawing-in)",
                  'EH': f"{sn}, counterfactual equal crew-hours: {nC.get('EH')} crews around the clock (least night drawing-in)"}
        for ax, p in zip(axes, pols): crew_gantt(ax, inst, kk[p], p, titles[p], hz)
        axes[-1].set_xlim(lo / 60, hz / 60); axes[-1].set_xlabel("hours from Monday 08:00", fontsize=FS)
        hd = [mpatches.Patch(color='#7f8c8d', label='drawing-in'), mpatches.Patch(color='#c0392b', label='overtime after 00:00'),
              mpatches.Patch(facecolor='white', edgecolor='#555', ls=':', label='paused until 08:00'),
              mpatches.Patch(color='#2c3e50', label='drawing-in at night (00-08)'),
              mpatches.Patch(color=SHADE['D'], label='day 08-16'), mpatches.Patch(color=SHADE['E'], label='evening 16-24'),
              mpatches.Patch(color=SHADE['N'], label='night 00-08')]
        fig.legend(handles=hd, loc='upper center', ncol=4, fontsize=FS, frameon=False, bbox_to_anchor=(0.5, 1.10))
        fig.tight_layout(); fig.savefig(f"{OUT_PREFIX}_crews.png", dpi=200, bbox_inches='tight'); plt.close(fig)
        taus = [s for s in D4_SCHEMES if s.startswith('Lex')]
        if taus:
            x = [int(s[3:]) for s in taus]; fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
            sty = {True: ('o-', 'night rule'), False: ('s--', 'night crew'), 'EH': ('^:', 'equal crew-hours')}
            for p in pols:
                mk, lab = sty[p]
                ax[0].plot(x, [B[(s, p)]['Z']['T'] / 60 for s in taus], mk, label=lab)
                ax[1].plot(x, [B[(s, p)]['Z']['S'] / 60 for s in taus], mk, label=lab)
                ax[2].plot(x, [(B[(s, p)]['Z']['O'] if p is True else B.get((s, p, 'nm'), B[(s, p)])['Z']['nightDI']) / 60
                               for s in taus], mk, label='overtime after 00:00' if p is True else lab + ', least night drawing-in')
            for a, yl in zip(ax, ('total tardiness (h)', 'drawing-in time (h)', 'night drawing-in work (h)')):
                a.set_xlabel('tardiness tolerance tau (%)', fontsize=FS); a.set_ylabel(yl, fontsize=FS); a.grid(alpha=0.3)
                a.tick_params(labelsize=FS - 1); a.legend(fontsize=FS - 2)
            fig.tight_layout(); fig.savefig(f"{OUT_PREFIX}_tolerance.png", dpi=200, bbox_inches='tight'); plt.close(fig)
    if D3PR:
        sns = list(D3PR); fks = list(D3_FACTORS)
        fig, ax = plt.subplots(2, len(fks), figsize=(2.8 * len(fks), 5.6), sharey='row')
        for j, fk in enumerate(fks):
            for q, sn in enumerate(sns):
                lv = D3_FACTORS[fk]; xs = [i + 0.08 * (q - (len(sns) - 1) / 2) for i in range(len(lv))]
                mh, lo, hi, mp = [], [], [], []
                for v in lv:
                    rs = [r for r in D3PR[sn] if r['inst']['meta'][FMETA.get(fk, fk)] == v]
                    h = [r['dT'] for r in rs]; c = boot_ci(h); mh.append(mean_(h)); lo.append(mean_(h) - c[0]); hi.append(c[1] - mean_(h))
                    mp.append(pct_agg(rs))
                ax[0][j].errorbar(xs, mh, yerr=[lo, hi], fmt='o-', capsize=3, label=sn)
                ax[1][j].plot(xs, mp, 'o-', label=sn)
            for r in (0, 1):
                ax[r][j].set_xticks(range(len(D3_FACTORS[fk]))); ax[r][j].set_xticklabels(['$\\infty$' if v == 'inf' else str(v) for v in D3_FACTORS[fk]], fontsize=FS - 1)
                ax[r][j].grid(alpha=0.3); ax[r][j].tick_params(labelsize=FS - 1); ax[r][j].axhline(0, color='grey', lw=0.6)
            ax[1][j].set_xlabel(FLABEL.get(fk, fk), fontsize=FS)
        ax[0][0].set_ylabel('price (h of tardiness)\nmean, 95% CI', fontsize=FS); ax[1][0].set_ylabel('price (% of tardiness,\nratio of means)', fontsize=FS)
        ax[0][-1].legend(fontsize=FS - 1)
        fig.tight_layout(); fig.savefig(f"{OUT_PREFIX}_pricemap.png", dpi=200, bbox_inches='tight'); plt.close(fig)
    if D5 and D5.get('conv'):
        cv = D5['conv']; x = [c['mk'] for c in cv]; fig, ax = plt.subplots(figsize=(6.4, 3.8))
        sty = {True: ('#1f77b4', 'o'), False: ('#ff7f0e', 's'), 'EH': ('#2ca02c', '^')}
        for p in D5.get('pols', (True, False)):
            col, m = sty[p]
            ax.plot(x, [c['gap'][p][0] for c in cv], '-', marker=m, color=col, label=f'{POL[p]}, single run')
            ax.plot(x, [c['gap'][p][1] for c in cv], ':', marker=m, color=col, mfc='white', label=f'{POL[p]}, best of {len(D5_REPS)}')
        ax.set_xscale('log'); ax.set_xlabel('evaluations per run', fontsize=FS)
        ax.set_ylabel('tardiness above the best known (h)', fontsize=FS)
        ax.grid(alpha=0.3); ax.legend(fontsize=FS - 2); ax.tick_params(labelsize=FS - 1)
        fig.tight_layout(); fig.savefig(f"{OUT_PREFIX}_convergence.png", dpi=200, bbox_inches='tight'); plt.close(fig)

# =============================================================== 15. MAIN and irace targets
def _tune_prep_one(sd):
    import json
    inst = gen_instance('L', sd, name=f"TUNE{sd}"); payoff(inst, PAY_BUDGET['L'], PAY_SEEDS)
    with open(f"tune_payoff_{sd}.json", 'w') as fh: json.dump(dict(ide=inst['ide'], nad=inst['nad'], rng=inst['rng']), fh)
    return sd, inst['n'], inst['m']

def tune_prep():
    """payoff tables of the tuning instances, computed in parallel before irace starts."""
    for sd, n, m in pmap(_tune_prep_one, list(TUNE_SEEDS), 'tuning payoff tables'):
        print(f"   tuning instance {sd}: {n} beams, {m} looms -> tune_payoff_{sd}.json")

def irace_target(argv, method='GP-RKO'):
    """irace target runner:  python weaving_gp_v9.py --irace <instance-seed> <seed> [--psize 30 ...]
                             python weaving_gp_v9.py --irace-brkga <instance-seed> <seed> [--p 100 --pe 0.2 ...]
       prints the attainment value (lower is better) for the Lex10 scheme under the night rule."""
    import json
    iseed, seed = int(argv[0]), int(argv[1]); cfg = {}; i = 2
    while i < len(argv):
        k = argv[i].lstrip('-'); v = argv[i + 1]; cfg[k] = float(v) if '.' in v else int(v); i += 2
    inst = gen_instance('L', iseed, name=f"TUNE{iseed}"); cache = f"tune_payoff_{iseed}.json"
    if os.path.exists(cache):
        with open(cache) as fh: d = json.load(fh); inst.update(ide=d['ide'], nad=d['nad'], rng=d['rng'])
    else:
        payoff(inst, PAY_BUDGET['L'], PAY_SEEDS)
        with open(cache, 'w') as fh: json.dump(dict(ide=inst['ide'], nad=inst['nad'], rng=inst['rng']), fh)
    sc = scheme_list(['Lex10'])['Lex10']; f = make_scalar(sc[0], inst['ide'], inst['rng'], tau=sc[2])
    rk = rule_keys(inst, f, True)
    if method == 'BRKGA-S': BRKGA_S_CFG.update(cfg); cfg = None
    r = run_method(inst, method, sc, True, seed, BUDGET['L'], rkeys=rk, cfg=cfg)
    print(value(sc, r['f'], inst))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('--irace', '--irace-brkga'):
        WORKERS = 1; irace_target(sys.argv[2:], 'GP-RKO' if sys.argv[1] == '--irace' else 'BRKGA-S'); sys.exit(0)
    if '--tune-prep' in sys.argv:
        for a in sys.argv[1:]:
            if a.startswith('--workers='): WORKERS = int(a.split('=')[1])
        tune_prep(); sys.exit(0)
    if '--quick' in sys.argv or QUICK: _quick()
    if '--reprice' in sys.argv:
        # tariff update: bill of the D3 best-known schedules from the checkpoint, and D4 again
        # (deterministic; only the reported bill changes)
        for a in sys.argv[1:]:
            if a.startswith('--workers='): WORKERS = int(a.split('=')[1])
        OUT_PREFIX = OUT_PREFIX + '_reprice'; t0 = time.time(); reprice_D3(); case = exp_D4()
        write_csv(f"{OUT_PREFIX}_results.csv")
        try: figures(None, case, None)
        except Exception as e: print("figures skipped:", e)
        print(f"\nTotal time {time.time()-t0:.0f} s.  Done."); sys.exit(0)
    for a in sys.argv[1:]:
        if a.startswith('--workers='): WORKERS = int(a.split('=')[1])
        if a.startswith('--cp-workers='): CP_WORKERS = int(a.split('=')[1])
        if a == '--resume': RESUME = True
        if a.startswith('--v8ckpt='): V8_D3_CKPT = a.split('=', 1)[1]
        if a.startswith('--only='):
            sel = a.split('=')[1].split(','); RUN = {k: (k in sel) for k in RUN}
            OUT_PREFIX = OUT_PREFIX + '_' + '_'.join(k for k in RUN if RUN[k])   # e.g. v8_D1_D2_results.csv
    tuned = _load_brkga_s(); t0 = time.time()
    print(f"WEAVING GP MODEL v9 | experiments {', '.join(k for k, v in RUN.items() if v)} | workers {WORKERS}, CP-SAT threads {CP_WORKERS}")
    print(f"replications {REPS}, budgets {BUDGET}, OT_MAX {OT_MAX} min, L_MAX {L_MAX} m")
    print(f"BRKGA-S configuration {BRKGA_S_CFG} ({'irace, brkga_s_cfg.json' if tuned else 'NOT TUNED: standard values'})\n", flush=True)
    D3I = D3PR = D5R = case = None
    if RUN['D1']: exp_D1(); write_csv(f"{OUT_PREFIX}_results.csv")
    if RUN['D2']: exp_D2(); write_csv(f"{OUT_PREFIX}_results.csv")
    if RUN['D3']: D3I, D3PR = exp_D3(); write_csv(f"{OUT_PREFIX}_results.csv")
    if RUN['D6']: exp_D6(D3I); write_csv(f"{OUT_PREFIX}_results.csv")
    if RUN['D5']: D5R = exp_D5(D3I, D3PR); write_csv(f"{OUT_PREFIX}_results.csv")
    if RUN['D4']: case = exp_D4(); write_csv(f"{OUT_PREFIX}_results.csv")
    print(f"\nSaved {OUT_PREFIX}_results.csv")
    try:
        figures(D3PR, case, D5R)
        made = [x for x in ('crews', 'tolerance', 'pricemap', 'convergence') if os.path.exists(f"{OUT_PREFIX}_{x}.png")]
        if made: print("Saved figures " + ", ".join(f"{OUT_PREFIX}_{x}.png" for x in made))
    except Exception as e:
        import traceback; traceback.print_exc(); print("figures skipped:", e)
    print(f"\nTotal time {time.time()-t0:.0f} s.  Done.")
