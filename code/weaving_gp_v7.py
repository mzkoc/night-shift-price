"""
================================================================================
 SUSTAINABLE WEAVING SCHEDULING, MODEL v7 (Oct 2026)  --  single file (local or Colab)
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
   Price of the night rule = T(rule) - T(crew) = [T(rule) - T(EH)]  timing part
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
   lexicographic GP: priority 1 tardiness within a tolerance tau of its ideal, priority 2
   Chebyshev GP on S, O (goal for O is zero overtime); strict priority in all methods
   (f = LEX_K * d_T[min] + min(A_2, LEX_P2CAP), see make_scalar).
 Methods (all scored by the same serial schedule-generation decoder):
   CP-SAT (one crew, small instances) | dispatching rules (EDD, ATCS, FAM, NEDD, NATCS)
   | GP-RKO (rule-seeded) | BRKGA (standard) | BRKGA-S (rule-seeded, irace-tuned)
   | SEED-LS (rule seeds + stochastic descent)
 Experiments: D1 exact validation and exact price (small, one crew), D2 scalability and
   algorithm comparison, D3 price map (factorial, 50 looms, three policies, bootstrap
   intervals, factor tests), D5 convergence of the price estimate, D4 case study.
 Local:  python -m pip install ortools matplotlib
         python -u weaving_gp_v7.py --quick          (10-15 minute test)
         python -u weaving_gp_v7.py                  (full run; --only=D1,D2,D3,D5,D4 to select;
                                                      outputs then carry the selection, e.g. v7_D1_D2_results.csv)
         options: --workers=18 --cp-workers=16
 Tuning: Rscript irace_tune.R        (GP-RKO:  python weaving_gp_v7.py --irace ...)
         Rscript irace_tune_brkga.R  (BRKGA-S: python weaving_gp_v7.py --irace-brkga ...;
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
SEGMENTS_TOU = [(   0,  480, 'Day',   4.80, 0.38, 'D'),
                ( 480,  540, 'Day',   4.80, 0.38, 'E'),
                ( 540,  840, 'Peak',  7.50, 0.45, 'E'),
                ( 840,  960, 'Night', 2.30, 0.46, 'E'),
                ( 960, 1320, 'Night', 2.30, 0.46, 'N'),
                (1320, 1440, 'Day',   4.80, 0.38, 'N')]
FLAT_PRICE = 4.50                 # single-rate tariff for the tariff factor
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
    """(calendar with the night rule?, number of crews) of a policy True / False / 'EH'."""
    if pol == 'EH': return False, eh_crews(inst['C'])
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

def lex_goals(ide, rng, tau):
    gT = ide['T'] * (1 + tau) if ide['T'] > 0 else tau * rng['T']
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
    if kind == 'lex':
        G = lex_goals(ide, rng, tau); w2 = {i: 1.0 for i in P2}
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
        G = lex_goals(ide, rng, tau); ws = {i: 1.0 for i in P2}
        devT = M.NewIntVar(0, ub['T'], "devT"); M.AddMaxEquality(devT, [V['T'] - int(math.floor(G['T'])), 0])
        M.Minimize(devT); sv, st = run(tlim / 3)
        if sv is None: return None
        hint = _extract(sv, V, inst); M.Add(devT <= sv.Value(devT))
    D, obj, q = cheb(ws, G); M.Minimize(obj); sv, st = run(tlim / 3 if kind == 'lex' else 2 * tlim / 3)
    if sv is None: return None
    keys = _extract(sv, V, inst)
    if sv.Value(D) == 0:
        M.Add(D == 0); M.Minimize(eff(ws, q)); hint = keys; sv2, _ = run(tlim / 3)
        if sv2: keys = _extract(sv2, V, inst)
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
    """'Balanced' etc. -> ('cheb', w, 0); 'Lex5' -> ('lex', None, 0.05)."""
    out = {}
    for s in kinds:
        out[s] = ('lex', None, int(s[3:]) / 100.0) if s.startswith('Lex') else ('cheb', PROFILES[s], 0.0)
    return out

def value(scal, f, inst):
    """attainment in range units: Chebyshev A as is; lexicographic GP: the tardiness deviation
       in tardiness ranges (priority 1), priority 2 entering only as a tie-break."""
    return f / (LEX_K * inst['rng']['T']) if scal[0] == 'lex' else f

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
RUN = dict(D1=True, D2=True, D3=True, D5=True, D4=True)
POL = {True: 'night rule', False: 'night crew', 'EH': 'equal crew-hours'}
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
FMETA = {'C': 'C_f'}              # factor name -> key in inst['meta']
# D5 convergence of the price estimate: 12 D3 instances (lmax 300, seed 1), long anytime runs
D5_MARKS = (25000, 50000, 100000, 200000, 400000)
D5_REPS, D5_SCHEME, D5_SELECT = (1, 2, 3, 4, 5), 'Lex0', dict(lmax=300, seed=1)
# D4 case study: tolerance curve of the lexicographic GP, three policies
CASE = dict(seed=7, m=50, C=3, n_fam=4, delta=1.0)
D4_SCHEMES = ('Lex0', 'Lex5', 'Lex10', 'Lex20', 'Lex30', 'Balanced', 'Delivery', 'Labour', 'NightSafe')
D4_NIGHT_BUDGET = 50000
BOOT_B = 2000                     # bootstrap resamples for confidence intervals
OUT_PREFIX = 'v7'
QUICK = False                     # True: tiny set-up for a short test
TUNE_SEEDS = tuple(range(101, 111))   # separate tuning instances (class L) for irace

def _quick():
    global REPS, BUDGET, PAY_BUDGET, PAY_SEEDS, D1_SEEDS, D2_SEEDS, D3_FACTORS, D3_SEEDS, D3_REPS, D3R_SEEDS
    global CP_TLIM, CP_PAY_TLIM, OUT_PREFIX, D4_SCHEMES, D3_M, D3_BUDGET, D3_PAY_BUDGET, NIGHT_BUDGET
    global D5_MARKS, D5_REPS, D4_NIGHT_BUDGET, BOOT_B
    REPS, PAY_SEEDS = (1, 2), (1,); BUDGET = {'S': 1500, 'M': 1500, 'L': 1500}; PAY_BUDGET = {'S': 800, 'M': 800, 'L': 800}
    D1_SEEDS, D2_SEEDS = (1, 2), {'M': (1,), 'L': (1, 2, 3)}
    D3_FACTORS = dict(C=(2, 3, 'inf'), nf=(4,), lmax=(300,), delta=(1.0, 0.8))
    D3_SEEDS, D3_REPS, D3R_SEEDS, CP_TLIM, CP_PAY_TLIM = (1,), (1, 2), (1,), 10, 5
    D4_SCHEMES, OUT_PREFIX = ('Lex0', 'Lex10', 'Balanced'), 'v7quick'; D3_BUDGET, D3_PAY_BUDGET = 1500, 800
    NIGHT_BUDGET = {'S': 300, 'L': 500}; D4_NIGHT_BUDGET = 500; D5_MARKS = (500, 1000, 2000); D5_REPS = (1, 2)
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
        r = run_method(inst, method, scal, rule, rep, budget, rkeys=rk, cfg=cfg); r.update(method=method, rep=rep); out.append(r)
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

def task_night(args):
    """worker: third lexicographic stage for a counterfactual policy: minimum night
       drawing-in (00-08) among schedules whose attainment is not worse than the best known."""
    inst, scheme, scal, pol, fstar, seeds, budget, seed = args
    f = make_scalar(scal[0], inst['ide'], inst['rng'], w=scal[1], tau=scal[2]); tol = 1e-9 * max(1.0, abs(fstar))
    def fit(k):
        Z = evaluate_keys(k, inst, pol); fv = f(Z)
        return Z['nightDI'] / 60.0 if fv <= fstar + tol else 1e6 + (fv - fstar)
    pool, ev = rko(inst, fit, budget, seed, seeds_keys=seeds)
    return inst['name'], pol, [k for fv, k in pool[:3] if fv < 1e6]

def pmap(fn, tasks, label=''):
    """run tasks in parallel (or serially when WORKERS <= 1)."""
    t0 = time.time(); out = []
    if WORKERS <= 1 or len(tasks) <= 1:
        for i, t in enumerate(tasks):
            out.append(fn(t))
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for i, r in enumerate(ex.map(fn, tasks, chunksize=1)):
                out.append(r)
                if (i + 1) % max(1, len(tasks) // 10) == 0:
                    print(f"      {label}: {i+1}/{len(tasks)} tasks, {time.time()-t0:.0f} s", flush=True)
    print(f"   {label}: {len(tasks)} tasks in {time.time()-t0:.0f} s", flush=True)
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

def bks(inst, scheme, rule):
    """best-known schedule: every schedule found for the instance (any method, scheme or
       policy) re-evaluated under this scheme and policy (cached until the pool changes)."""
    c = inst.setdefault('_bks', {})
    if (scheme, rule) in c: return c[(scheme, rule)]
    scal = scheme_list([scheme])[scheme]; f = make_scalar(scal[0], inst['ide'], inst['rng'], w=scal[1], tau=scal[2])
    best = None
    for keys, src in pool_keys(inst):
        Z = evaluate_keys(keys, inst, rule); fv = f(Z)
        if best is None or fv < best['f'] - 1e-12: best = dict(f=fv, v=value(scal, fv, inst), Z=Z, keys=keys, src=src)
    c[(scheme, rule)] = best
    return best

def night_min(inst, scheme, pol):
    """schedule with the least night drawing-in among the pooled schedules that are as good
       as the best known for this scheme and policy (third lexicographic priority)."""
    c = inst.setdefault('_nm', {})
    if (scheme, pol) in c: return c[(scheme, pol)]
    b = bks(inst, scheme, pol); scal = scheme_list([scheme])[scheme]
    f = make_scalar(scal[0], inst['ide'], inst['rng'], w=scal[1], tau=scal[2]); tol = 1e-9 * max(1.0, abs(b['f'])); best = None
    for keys, _ in pool_keys(inst):
        Z = evaluate_keys(keys, inst, pol)
        if f(Z) <= b['f'] + tol and (best is None or Z['nightDI'] < best['Z']['nightDI']): best = dict(Z=Z, keys=keys)
    c[(scheme, pol)] = best
    return best

def stage3(insts, schemes, budget, label):
    """third stage for the counterfactual policies: GP-RKO minimising night drawing-in subject
       to the best-known attainment, seeded with the equally good pooled schedules."""
    tasks = []
    for inst in insts:
        for sn in schemes:
            sc = scheme_list([sn])[sn]; f = make_scalar(sc[0], inst['ide'], inst['rng'], w=sc[1], tau=sc[2])
            for pol in pols_of(inst):
                if pol is True: continue
                b = bks(inst, sn, pol); tol = 1e-9 * max(1.0, abs(b['f'])); seeds = [b['keys']]
                for keys, _ in pool_keys(inst):
                    if len(seeds) >= RKO_CFG['psize'] - 1: break
                    if keys is not b['keys'] and f(evaluate_keys(keys, inst, pol)) <= b['f'] + tol: seeds.append(keys)
                tasks.append((light(inst), sn, sc, pol, b['f'], seeds, budget, 1))
    by = {i['name']: i for i in insts}
    for nm, pol, kl in pmap(task_night, tasks, label):
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
                    if mth == 'GP-RKO-w1e3' and sc[0] != 'lex': continue
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
                b = bks(inst, sn, rule); row = []; tol = 1e-9 * max(1.0, abs(b['f']))
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

D3HDR = (f"  {'scheme':<7}{'factor':<7}{'level':>6}{'n':>4}{'dv%':>7}{'dT h [95% CI]':>22}{'dT%':>6}{'dS h':>7}"
         f"{'late%':>8}{'O h':>6}{'pause':>6}{'bind':>6}{'bindT':>6}{'night h':>9}{'min':>7}")
def prow(sn, fk, lv, rs):
    g = lambda key: [r[key] for r in rs]
    print(f"  {sn:<7}{fk:<7}{str(lv):>6}{len(rs):>4}{mean_(g('dv')):>7.2f}{fci(g('dT')):>22}{mean_(g('dTp')):>6.1f}"
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
        means = " / ".join(f"{lv}: {mean_([b[j] for b in B]):.1f}" for j, lv in enumerate(levels))
        if len(levels) == 2:
            pv = wilcoxon([b[0] for b in B], [b[1] for b in B]); print(f"    {sn:<6} {key:<4} {fk:<6} {means}  (blocks {len(B)}) Wilcoxon p={pv:.3f}")
        else:
            _, chi2, pv = friedman(B); pairs = [(a, c) for a in range(len(levels)) for c in range(a + 1, len(levels))]
            ad = holm([wilcoxon([b[a] for b in B], [b[c] for b in B]) for a, c in pairs])
            print(f"    {sn:<6} {key:<4} {fk:<6} {means}  (blocks {len(B)}) Friedman p={pv:.3f}; pairs (Holm) " +
                  ", ".join(f"{levels[a]}-{levels[c]} {q:.3f}" for (a, c), q in zip(pairs, ad)))

def eh_table(PR, label):
    print(f"  {label}")
    print(f"  {'scheme':<7}{'n':>4}{'total dT h':>22}{'timing (rule-EH)':>24}{'p':>7}{'capacity (EH-crew)':>24}{'p':>7}"
          f"{'timing share':>13}{'dS rule-EH':>11}{'night crew':>11}{'night EH':>9}")
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
    stage3(insts, D1_SCHEMES, NIGHT_BUDGET['S'], 'D1 night minimisation')
    print("\n" + "=" * 118 + "\nTABLE 2  D1 exact price of the night rule; '*' marks pairs where CP-SAT proved both optima\n" + "=" * 118)
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
                    if sc[0] != 'lex': continue
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

def exp_D3():
    print("\n#### D3  price map of the night rule (mill size, 50 looms, three policies)")
    insts = d3_instances(); INST = {i['name']: i for i in insts}
    do_payoffs(insts, lambda i: 'L', D3_PAY_BUDGET, D3_PAY_SEEDS)
    out = pmap(task_run, heuristic_tasks(insts, D3_SCHEMES, pols_of, ('GP-RKO',), D3_REPS, lambda i: 'L', D3_BUDGET), 'D3 runs')
    record('D3', [r for lst in out for r in lst], INST); refit_all(insts, 'D3')
    stage3(insts, D3_SCHEMES, NIGHT_BUDGET['L'], 'D3 night minimisation')
    main = [i for i in insts if i['meta']['setup'] == 'structured']; rob = [i for i in insts if i['meta']['setup'] == 'general']
    print("\n" + "=" * 118 + "\nTABLE 4  D3 price of the night rule by factor level (best-known schedules, night rule minus night crew)\n" + "=" * 118)
    print("  dv: attainment difference (% of ranges; lexicographic: % of the tardiness range); dT: tardiness (h, mean and 95%")
    print("  bootstrap interval over instances; %); dS: drawing-in time (h); late%: late orders, night rule / night crew;")
    print("  O: overtime after midnight (h); pause: paused drawing-ins; bind: attainment worse under the rule (%);")
    print("  bindT: tardiness higher under the rule (%); night h: night drawing-in of the best-known night-crew schedule;")
    print("  min: least night drawing-in among equally good night-crew schedules (third lexicographic stage)")
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
        for key in ('dT', 'dTp', 'dS'): factor_tests(PR[sn], sn, key)
    print("\n" + "=" * 118 + "\nTABLE 4b  D3 decomposition of the price, C = 3: timing (rule - equal crew-hours) and capacity (equal crew-hours - night crew)\n" + "=" * 118)
    print("  hours of tardiness, mean [95% bootstrap interval]; p: Wilcoxon signed-rank test against zero; night: least night drawing-in (h)")
    eh_table(PR, 'structured changeovers:'); eh_table(PRR, 'general changeover matrix:')
    print("  C = inf: the crew capacity does not bind, so its price is a timing price (Table 4, level inf)")
    print("\n" + "=" * 118 + "\nTABLE 4c  D3 search noise of GP-RKO per policy (tardiness, h)\n" + "=" * 118)
    noise_table('D3', insts, D3_SCHEMES, pols_of)
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
    print("\n#### D5  convergence of the price estimate (Lex0, long anytime runs)")
    if insts is None:
        insts = [i for i in d3_instances() if i['meta']['setup'] == 'structured'
                 and i['meta']['lmax'] == D5_SELECT['lmax'] and i['seed'] == D5_SELECT['seed']]
        do_payoffs(insts, lambda i: 'L', D3_PAY_BUDGET, D3_PAY_SEEDS)
    sel = [i for i in insts if i['meta']['setup'] == 'structured' and i['meta']['lmax'] == D5_SELECT['lmax']
           and i['seed'] == D5_SELECT['seed']]
    for i in sel: INSTREG[i['name']] = i
    sc = scheme_list([D5_SCHEME])[D5_SCHEME]
    tasks = [(light(i), sc, pol, rp, D5_MARKS) for i in sel for pol in pols_of(i) for rp in D5_REPS]
    tasks.sort(key=lambda t: -t[0]['n'])
    for nm, pol, rep, tr in pmap(task_trace, tasks, 'D5 runs'): D5_OUT[(nm, pol, rep)] = tr
    def at(nm, pol, rep, mk):
        return next((Z, fv) for m_, fv, Z in D5_OUT[(nm, pol, rep)] if m_ == mk)
    def best(nm, pol, mk):
        return min((at(nm, pol, rp, mk) for rp in D5_REPS), key=lambda x: x[1])[0]
    print("\n" + "=" * 118 + f"\nTABLE 7  D5 price of the night rule against the search budget ({len(sel)} instances, {D5_SCHEME}, "
          f"{len(D5_REPS)} replications)\n" + "=" * 118)
    print("  best: best of the replications per policy; single: paired single runs (replication r under both policies);")
    print("  spread: max - min tardiness over the replications (h); timing / capacity: C = 3 instances, best of the replications")
    print(f"  {'evals':>8}{'price best h [95% CI]':>26}{'price single h [95% CI]':>28}{'spread rule':>12}{'spread crew':>12}"
          f"{'timing':>9}{'capacity':>10}")
    conv = []
    for mk in D5_MARKS:
        pb, ps, s1, s0, ti, ca = [], [], [], [], [], []
        for i in sel:
            nm = i['name']; pb.append((best(nm, True, mk)['T'] - best(nm, False, mk)['T']) / 60)
            ps += [(at(nm, True, rp, mk)[0]['T'] - at(nm, False, rp, mk)[0]['T']) / 60 for rp in D5_REPS]
            for pol, acc in ((True, s1), (False, s0)):
                T = [at(nm, pol, rp, mk)[0]['T'] / 60 for rp in D5_REPS]; acc.append(max(T) - min(T))
            if 'EH' in pols_of(i):
                te = best(nm, 'EH', mk)['T']; ti.append((best(nm, True, mk)['T'] - te) / 60); ca.append((te - best(nm, False, mk)['T']) / 60)
        conv.append(dict(mk=mk, pb=pb, ps=ps))
        print(f"  {mk:>8}{fci(pb):>26}{fci(ps):>28}{mean_(s1):>12.1f}{mean_(s0):>12.1f}{mean_(ti):>9.1f}{mean_(ca):>10.1f}")
    d3 = None
    if PR and D5_SCHEME in PR:
        names = {i['name'] for i in sel}; d3 = [r['dT'] for r in PR[D5_SCHEME] if r['inst']['name'] in names]
        if d3: print(f"  D3 estimate on the same instances ({len(D3_REPS)} replications of {D3_BUDGET} evaluations, pooled): {fci(d3)} h")
    return dict(conv=conv, d3=d3)

def exp_D4():
    print("\n#### D4  case study: tolerance curve of the lexicographic GP")
    inst = gen_instance('F', CASE['seed'], name='CASE', m=CASE['m'], C=CASE['C'], n_fam=CASE['n_fam'], delta=CASE['delta'])
    inst['meta'] = dict(C_f=CASE['C'], nf=CASE['n_fam'], lmax=L_MAX, delta=CASE['delta'], tariff='ToU', setup='structured')
    INST = {'CASE': inst}; do_payoffs([inst], lambda i: 'L'); print_payoff(inst)
    print(f"   {len(inst['orders'])} orders, {inst['n']} beams, families (h) " +
          ", ".join(f"{f}:{t/60:g}" for f, t in inst['fam_T'].items()))
    out = pmap(task_run, heuristic_tasks([inst], D4_SCHEMES, pols_of, ('GP-RKO',), REPS, lambda i: 'L'), 'D4 runs')
    record('D4', [r for lst in out for r in lst], INST); refit_all([inst], 'D4')
    stage3([inst], D4_SCHEMES, D4_NIGHT_BUDGET, 'D4 night minimisation')
    print("\n" + "=" * 118 + "\nTABLE 6  D4 case: best-known schedules per scheme and policy\n" + "=" * 118)
    print("  counterfactual policies: nightDI of the best-known schedule; 'night min' = least night drawing-in at the same attainment")
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
                    mp.append(mean_([r['dTp'] for r in rs]))
                ax[0][j].errorbar(xs, mh, yerr=[lo, hi], fmt='o-', capsize=3, label=sn)
                ax[1][j].plot(xs, mp, 'o-', label=sn)
            for r in (0, 1):
                ax[r][j].set_xticks(range(len(D3_FACTORS[fk]))); ax[r][j].set_xticklabels([str(v) for v in D3_FACTORS[fk]], fontsize=FS - 1)
                ax[r][j].grid(alpha=0.3); ax[r][j].tick_params(labelsize=FS - 1); ax[r][j].axhline(0, color='grey', lw=0.6)
            ax[1][j].set_xlabel(fk, fontsize=FS)
        ax[0][0].set_ylabel('price (h of tardiness)\nmean, 95% CI', fontsize=FS); ax[1][0].set_ylabel('price (% of tardiness)', fontsize=FS)
        ax[0][-1].legend(fontsize=FS - 1)
        fig.tight_layout(); fig.savefig(f"{OUT_PREFIX}_pricemap.png", dpi=200, bbox_inches='tight'); plt.close(fig)
    if D5 and D5.get('conv'):
        cv = D5['conv']; x = [c['mk'] for c in cv]; fig, ax = plt.subplots(figsize=(6, 3.6))
        for key, lab, mk in (('pb', 'best of the replications', 'o-'), ('ps', 'single runs (paired)', 's--')):
            m = [mean_(c[key]) for c in cv]; ci = [boot_ci(c[key]) for c in cv]
            ax.plot(x, m, mk, label=lab); ax.fill_between(x, [c[0] for c in ci], [c[1] for c in ci], alpha=0.15)
        if D5.get('d3'): ax.axhline(mean_(D5['d3']), color='k', ls=':', lw=1, label='D3 estimate (pooled)')
        ax.set_xscale('log'); ax.set_xlabel('evaluations per run', fontsize=FS); ax.set_ylabel('price of the night rule (h)', fontsize=FS)
        ax.grid(alpha=0.3); ax.legend(fontsize=FS - 1); ax.tick_params(labelsize=FS - 1)
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
    """irace target runner:  python weaving_gp_v7.py --irace <instance-seed> <seed> [--psize 30 ...]
                             python weaving_gp_v7.py --irace-brkga <instance-seed> <seed> [--p 100 --pe 0.2 ...]
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
    for a in sys.argv[1:]:
        if a.startswith('--workers='): WORKERS = int(a.split('=')[1])
        if a.startswith('--cp-workers='): CP_WORKERS = int(a.split('=')[1])
        if a.startswith('--only='):
            sel = a.split('=')[1].split(','); RUN = {k: (k in sel) for k in RUN}
            OUT_PREFIX = OUT_PREFIX + '_' + '_'.join(k for k in RUN if RUN[k])   # e.g. v7_D1_D2_results.csv
    tuned = _load_brkga_s(); t0 = time.time()
    print(f"WEAVING GP MODEL v7 | experiments {', '.join(k for k, v in RUN.items() if v)} | workers {WORKERS}, CP-SAT threads {CP_WORKERS}")
    print(f"replications {REPS}, budgets {BUDGET}, OT_MAX {OT_MAX} min, L_MAX {L_MAX} m")
    print(f"BRKGA-S configuration {BRKGA_S_CFG} ({'irace, brkga_s_cfg.json' if tuned else 'NOT TUNED: standard values'})\n", flush=True)
    D3I = D3PR = D5R = case = None
    if RUN['D1']: exp_D1(); write_csv(f"{OUT_PREFIX}_results.csv")
    if RUN['D2']: exp_D2(); write_csv(f"{OUT_PREFIX}_results.csv")
    if RUN['D3']: D3I, D3PR = exp_D3(); write_csv(f"{OUT_PREFIX}_results.csv")
    if RUN['D5']: D5R = exp_D5(D3I, D3PR); write_csv(f"{OUT_PREFIX}_results.csv")
    if RUN['D4']: case = exp_D4(); write_csv(f"{OUT_PREFIX}_results.csv")
    print(f"\nSaved {OUT_PREFIX}_results.csv")
    try:
        figures(D3PR, case, D5R); print(f"Saved figures {OUT_PREFIX}_crews.png, _tolerance.png, _pricemap.png, _convergence.png")
    except Exception as e:
        import traceback; traceback.print_exc(); print("figures skipped:", e)
    print(f"\nTotal time {time.time()-t0:.0f} s.  Done.")
