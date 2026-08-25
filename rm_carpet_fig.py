"""Emit the tikz for the survival-carpet panels of example 39,
which are pasted into leveraged_learning.tex rather than read at
compile time.  Needs figures/data/rmlabel_n{n}.dat for the two
increment curves; run rm_label_data.py first.


Bands are cut at the 1-F curve, so the shaded region is exactly the
limiting object: height 1-F(t) = gamma(t), area to the left g(t).  The
curve enters band d's slab at its top-left corner (R_{d-1}, 1-F(R_{d-1}))
and leaves at the bottom-right (R_d, 1-F(R_d)), so the horizontal slab
boundaries, the vertical rate lines and the curve all meet at the same
points.
"""
import os
from math import comb, sqrt

F = lambda x: 1 - (1 - x) ** 2
OUT = []


def band_data(n):
    Q = 2 ** n
    K = [sum(comb(n, i) for i in range(d + 1)) for d in range(n + 1)]
    R = [k / Q for k in K]
    w, prev = [], 0.0
    for d in range(n + 1):
        w.append(F(R[d]) - prev)
        prev = F(R[d])
    # band d spans y in [1-F(R_d), 1-F(R_{d-1})], right edge is the curve
    bands = []
    for d in range(n + 1):
        yb = 1 - F(R[d])
        yt = 1 - (F(R[d - 1]) if d else 0.0)
        bands.append((d, yb, yt, R[d]))
    return K, R, w, bands


def steps(path, key, ymax=1.06):
    """step path from a .dat file, as tikz coordinates"""
    rows = []
    with open(path) as fh:
        head = next(fh).split()
        col = head.index(key)
        for line in fh:
            f = [float(x) for x in line.split()]
            rows.append((f[0], f[col]))
    pts = []
    for i, (t, y) in enumerate(rows):
        t2 = rows[i + 1][0] if i + 1 < len(rows) else 1.0
        pts.append((t, min(y, ymax)))
        pts.append((t2, min(y, ymax)))
    return pts


def fmt(pts, indent='    ', width=68):
    out, line = [], indent
    for x, y in pts:
        piece = '(%.4f,%.4f) -- ' % (x, y)
        if len(line) + len(piece) > width:
            out.append(line.rstrip())
            line = indent
        line += piece
    out.append(line.rstrip().rstrip('- ').rstrip())
    return '\n'.join(out)


def panel(n, xs, ys, curves, xlabels, arrows=False, ylabels=True,
          gammaside=False, callouts=False,
          tints=('cf2!30', 'cf2!18')):
    K, R, w, bands = band_data(n)
    L = []
    L.append('\\begin{tikzpicture}[x=%scm, y=%scm]' % (xs, ys))
    L.append('  %% same reserved extent in every panel, so they line up')
    L.append('  \\path (-0.175,-0.10) rectangle (1.12,1.14);')
    L.append('  %% bands cut at the curve: slab d spans the rate gap')
    for k, (d, yb, yt, rd) in enumerate(bands):
        if yt - yb < 1e-9:
            continue
        L.append('  \\fill[%s, draw=black!40, line width=0.2pt]'
                 % tints[k % 2])
        L.append('    (0,%.6f) -- plot[domain=%.6f:%.6f, samples=24]'
                 % (yb, yb, yt))
        L.append('    ({1-sqrt(\\x)},\\x) -- (0,%.6f) -- cycle;' % yt)
    L.append('  %% the rates, marked up to the curve they generate')
    for d, yb, yt, rd in bands:
        if rd < 1e-6 or yb < 1e-4:
            continue
        L.append('  \\draw[black!45, densely dashed, line width=0.3pt]'
                 ' (%.6f,0) -- (%.6f,%.6f);' % (rd, rd, yb))
    L.append('  \\draw[clim, thick, domain=0:1, samples=90]'
             ' plot (\\x, {(1-\\x)^2});')
    L.append('  \\draw[->, black!70] (0,0) -- (1.06,0)'
             ' node[right, font=\\scriptsize, black!70] {$t$};')
    L.append('  \\draw[->, black!70] (0,0) -- (0,1.12);')
    L.append('  \\foreach \\y in {0, 1}'
             ' \\draw[black!70] (-0.007,\\y) -- (0.007,\\y);')
    if ylabels:
        L.append('  \\foreach \\y/\\lab in {0/0, 1/1}')
        L.append('    \\node[font=\\scriptsize, black!70, left]'
                 ' at (-0.009,\\y) {$\\lab$};')
    if gammaside:
        L.append('  \\node[font=\\scriptsize, black!70, rotate=90]'
                 ' at (-0.055,0.5) {$\\gamma$};')
    L.append('  %% w_d partition on the vertical axis')
    ws = ', '.join('%.6f' % yb for d, yb, yt, rd in bands if 1e-4 < yb < 0.999)
    L.append('  \\foreach \\y in {%s}' % ws)
    L.append('    \\draw[black!70, line width=0.5pt] (-0.014,\\y) -- (0,\\y);')
    if arrows:
        L.append('  %% the slabs are the class weights, read up the axis')
        for d, yb, yt, rd in bands:
            if yt - yb < 0.02:
                continue
            L.append('  \\draw[{Stealth[length=3pt]}-{Stealth[length=3pt]},'
                     ' black!65, line width=0.35pt]')
            L.append('    (-0.038,%.6f) -- (-0.038,%.6f);' % (yb, yt))
            L.append('  \\node[font=\\scriptsize, black!70, anchor=east]'
                     ' at (-0.044,%.6f) {$+w_%d$};' % ((yb + yt) / 2, d))
        # slabs too thin to carry an arrow are left unlabelled; at n = 4
        # that is only w_4, four thousandths of the belief
    if xlabels:
        # rates named, staggered over two rows so neighbours do not touch
        marks = [(0.0, '0')] + [(rd, '1' if rd >= 1.0 else
                                 'R_{%d,%d}' % (n, d))
                                for d, yb, yt, rd in bands]
        rows, last = [], [-9.0, -9.0]
        for x, lab in marks:
            r = 0 if x - last[0] >= 0.16 else 1
            if x - last[r] < 0.16:
                continue
            last[r] = x
            rows.append((x, lab, r))
        L.append('  \\foreach \\x in {%s}'
                 % ', '.join('%.6f' % x for x, _, _ in rows))
        L.append('    \\draw[black!70] (\\x,-0.016) -- (\\x,0.016);')
        for x, lab, r in rows:
            L.append('  \\node[font=\\scriptsize, black!70, below]'
                     ' at (%.6f,%.4f) {$%s$};'
                     % (x, -0.022 - 0.055 * r, lab))
    else:
        rs = ', '.join('%.6f' % rd for d, yb, yt, rd in bands if rd > 1e-5)
        L.append('  \\foreach \\x in {%s}' % rs)
        L.append('    \\draw[black!70] (\\x,-0.014) -- (\\x,0.014);')
    if curves:
        dat = 'figures/data/rmlabel_n%d.dat' % n
        up, dn = steps(dat, 'true'), steps(dat, 'mix')
        L.append('  %% what an answer is worth: not knowing D, and knowing it')
        L.append('  \\fill[black!22]')
        L.append(fmt(up + dn[::-1]) + ' -- cycle;')
        L.append('  \\draw[black!60, densely dashed, line width=0.45pt]')
        L.append(fmt(dn) + ';')
        L.append('  \\draw[black!80, line width=0.5pt]')
        L.append(fmt(up) + ';')
    if callouts:
        L.append('  %% which curve is which')
        L.append('  \\node[font=\\scriptsize, black!85, anchor=west]'
                 ' at (0.50,0.93) {$\\gamma(t)$};')
        L.append('  \\draw[->, black!75, line width=0.35pt]'
                 ' (0.505,0.910) -- (0.415,0.725);')
        L.append('  \\node[font=\\scriptsize, black!70, anchor=west]'
                 ' at (0.745,0.45)'
                 ' {$\\sum_d w_d\\,\\gamma^{(d)}_{n,\\ell}$};')
        L.append('  \\draw[->, black!75, line width=0.35pt]'
                 ' (0.745,0.425) -- (0.705,0.235);')
    L.append('  \\node[font=\\small, black!70, anchor=east]'
             ' at (1.0,0.88) {$n = %d$};' % n)
    L.append('\\end{tikzpicture}')
    return '\n'.join(L)


print(panel(4, '10.5', '3.7', True, True, arrows=True,
            ylabels=False, callouts=True))
print()
print(panel(8, '10.5', '3.2', True, '', gammaside=True))
print()
print(panel(12, '10.5', '3.0', True, '', gammaside=True))
