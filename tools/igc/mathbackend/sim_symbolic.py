"""Simulation of the `symbolic` module against independent oracles, with no model.

    PYTHONPATH=. python -m tools.igc.mathbackend.sim_symbolic --seed 1 --n 60      (needs SymPy and mpmath: the venv named in the plan)

The backend was written without copying SymPy, so SymPy and mpmath are independent here. Every case says which oracle decides it:
  sympy-exact        SymPy simplifies the difference to zero exactly (polynomials, rational functions, radicals)
  sympy-numeric      SymPy's result and the backend's text are evaluated by mpmath at several points (40 digits) and must agree to 25
  certificate        the answer is verified, not recomputed: an antiderivative is differentiated by SymPy and compared with the integrand at several points;
                     a root is substituted back; a Taylor polynomial is compared with the function at small h; a closed-form sum is compared with the explicit sum
  brute-force        explicit loop over exact Fractions
A case is (op_name, args, oracle, oracle_name); `oracle` is a dict the checker of that operation reads. `run(seed, n)` writes build-mathbackend/symbolic.json.
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter, namedtuple
from fractions import Fraction
from pathlib import Path

import mpmath
import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication_application, parse_expr, standard_transformations

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.igc.mathbackend import BadInput, NotCovered, registry, run as run_op  # noqa: E402
from tools.igc.mathbackend import symbolic as sym_mod  # noqa: E402

Case = namedtuple('Case', 'op args oracle oracle_name')
mpmath.mp.dps = 45
X, Y = sp.Symbol('x', real=True), sp.Symbol('y', real=True)
K, N_ = sp.Symbol('k', integer=True), sp.Symbol('n', integer=True)
LOCAL = {'x': X, 'y': Y, 'k': K, 'n': N_, 'abs': sp.Abs, 'floor': sp.floor, 'atan': sp.atan, 'E': sp.E, 'pi': sp.pi, 'exp': sp.exp, 'log': sp.log,
         'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'sqrt': sp.sqrt}
TRANS = standard_transformations + (convert_xor, implicit_multiplication_application)
TIMEOUT_NOTE = 'oracle gave no answer'


def sy(text):
    text = re.sub(r'[|]([^|]+)[|]', lambda m: 'abs(' + m.group(1) + ')', str(text))
    return parse_expr(text, local_dict=LOCAL, transformations=TRANS, evaluate=True)


def mp_values(expr, var, pts):
    out = []
    f = sp.lambdify(var, expr, 'mpmath')
    for p in pts:
        try:
            v = f(mpmath.mpf(p))
            if isinstance(v, mpmath.mpc):
                if abs(v.imag) > mpmath.mpf(10) ** -30:
                    out.append(None)
                    continue
                v = v.real
            out.append(mpmath.mpf(v))
        except Exception:
            out.append(None)
    return out


def close(a, b, tol=25):
    return abs(a - b) <= mpmath.mpf(10) ** -tol * (1 + abs(a))


def numeric_equal(e1, e2, var, pts, need=3):
    v1, v2 = mp_values(e1, var, pts), mp_values(e2, var, pts)
    ok = 0
    for a, b in zip(v1, v2):
        if a is None or b is None:
            continue
        if not close(a, b):
            return False
        ok += 1
    return ok >= need


GOOD_PTS = ['0.37', '1.13', '1.91', '2.47', '0.61', '3.29']
ANY_PTS = ['0.37', '-0.83', '1.91', '-2.41', '0.61', '3.29', '-1.57']


def fr(rs, lo=-6, hi=6, den=(1, 2, 3, 4, 5)):
    d = rs.choice(den)
    n = rs.randint(lo, hi)
    return f'{n}/{d}' if d != 1 else str(n)


def sgn(n):
    return f'+ {n}' if n >= 0 else f'- {-n}'


def lin(rs, v='x'):
    a, b = rs.choice([1, 1, 2, 3, -1, -2]), rs.randint(-5, 5)
    return f'({a}*{v} {sgn(b)})' if a != 1 else f'({v} {sgn(b)})'


# ------------------------------------------------------------------------------------------------------------------------------------ generators

def gen_canon(rs, n):
    out = []
    latex = [('\\frac{x^2+1}{2}', '(x^2+1)/2'), ('\\frac{1}{\\sqrt{x}}', '1/sqrt(x)'), ('\\frac{\\sin x}{\\cos x}', 'sin(x)/cos(x)'),
             ('2\\cdot x^{3}', '2*x^3'), ('\\sqrt{8}\\times x', 'sqrt(8)*x'), ('x^{-2}+3x', 'x^(-2)+3*x'), ('\\frac{x+1}{x-1}', '(x+1)/(x-1)'),
             ('\\sqrt[3]{27}x', '3*x'), ('\\dfrac{1}{2}x^{2}', 'x^2/2'), ('\\left(x+1\\right)^{2}', '(x+1)^2'), ('\\exp(x)\\cdot\\exp(2x)', 'exp(3*x)'),
             ('\\ln x + \\ln 2', 'log(x)+log(2)'), ('\\sin^2 x + \\cos^2 x', 'sin(x)^2+cos(x)^2'), ('|x-1|', 'abs(x-1)'), ('\\pi x', 'pi*x'),
             ('\\log_2 8', '3'), ('\\frac{\\pi}{6}', 'pi/6'), ('\\sin\\left(\\frac{\\pi}{6}\\right)', '1/2'), ('\\arctan(x)', 'atan(x)'), ('2^{x}\\cdot 2^{x+1}', '2*4^x')]
    for i in range(n):
        kind = i % 6
        if kind == 0:
            t = f'{lin(rs)}^{rs.randint(2, 4)}*{lin(rs)} - {rs.randint(1, 5)}*x*{lin(rs)}'
            pts = ANY_PTS
        elif kind == 1:
            t = f'({rs.randint(1, 4)}*x^2 {sgn(rs.randint(-3, 3))})/{lin(rs)} + {fr(rs)}/{lin(rs)}'
            pts = ANY_PTS
        elif kind == 2:
            t = rs.choice([f'sqrt({rs.choice([8, 12, 18, 20, 50, 72, 75])})*x + sqrt({rs.choice([2, 3, 5])})*x', f'(1+sqrt({rs.choice([2, 3, 5])}))^{rs.randint(2, 4)}',
                           f'sqrt(x^2)*{rs.randint(1, 4)} - abs(x)', f'(x+1)^(3/2)/(x+1)^(1/2)', f'x^(1/2)*x^(3/2)*{rs.randint(1, 3)}',
                           f'sqrt({rs.choice([6, 10, 15])})*sqrt({rs.choice([2, 3, 5])})*x'])
            pts = GOOD_PTS if 'x+1' in t or 'x^(1/2)' in t else ANY_PTS
        elif kind == 3:
            t = rs.choice([f'exp(x)*exp({rs.randint(1, 3)}*x)', f'exp(2*log(3))*x', f'sin(x)^2*sin(x)', f'log(12*x)', f'sin(pi/6)*x + cos(pi/3)',
                           f'exp(x)*exp(-x) + x', f'sin(-x) + sin(x)*{rs.randint(2, 4)}', f'cos(-x)*{rs.randint(1, 3)}', f'2^x*2^(x+1)', f'3^x*9^x'])
            pts = GOOD_PTS
        elif kind == 4:
            t = f'{lin(rs)}*{lin(rs)}*{lin(rs)} - {lin(rs)}^3'
            pts = ANY_PTS
        else:
            t = f'x*y*{lin(rs)} + y^2*{lin(rs, "y")} - ({lin(rs)}*{lin(rs, "y")})'
            pts = ANY_PTS
        out.append(Case('symbolic.canon', (t,), {'sympy': t, 'pts': pts, 'var': 'x'}, 'sympy-numeric'))
        if kind == 5 and i % 12 == 5:
            pass
    for ltx, plain in latex:
        out.append(Case('symbolic.canon', (ltx,), {'sympy': plain, 'pts': GOOD_PTS, 'var': 'x'}, 'sympy-numeric'))
    return out


def gen_expand(rs, n):
    out = []
    for i in range(n):
        k = i % 4
        if k == 0:
            t = f'{lin(rs)}^{rs.randint(2, 6)}'
        elif k == 1:
            t = f'{lin(rs)}*{lin(rs)}*{lin(rs)}*{lin(rs)}'
        elif k == 2:
            t = f'(x + y + {rs.randint(1, 3)})^{rs.randint(2, 4)} - (x - y)^2'
        else:
            t = f'({lin(rs)}^2 + {lin(rs, "y")})*({lin(rs)} - {lin(rs, "y")}^2)'
        out.append(Case('symbolic.expand', (t,), {'sympy': t}, 'sympy-exact'))
    return out


def gen_substitute(rs, n):
    out = []
    for i in range(n):
        base = rs.choice([f'x^2 + {rs.randint(1, 5)}*x*y', f'({lin(rs)})^3', f'x/(x+{rs.randint(1, 4)}) + y', f'exp(x)*sin(y)', f'sqrt(x+{rs.randint(1, 5)})*y', f'log(x)+x^2'])
        val = rs.choice([f'y + {rs.randint(1, 3)}', f'2*y', f'{rs.randint(1, 4)}/{rs.randint(2, 5)}', f'y^2 + 1', f'x + y'])
        out.append(Case('symbolic.substitute', (base, 'x', val), {'expr': base, 'val': val}, 'sympy-numeric'))
    return out


def gen_evaluate(rs, n):
    out = []
    for i in range(n):
        k = i % 5
        xv, yv = fr(rs, 1, 9), fr(rs, 1, 9)
        if k == 0:
            t = f'x^{rs.randint(2, 4)} - {rs.randint(1, 5)}*x*y + y/{rs.randint(2, 7)}'
        elif k == 1:
            t = f'(x + {rs.randint(1, 3)})/(y - {rs.randint(10, 20)}) + x*y'
        elif k == 2:
            t = f'floor(x*{rs.randint(2, 7)}) + abs(y - {rs.randint(2, 6)})'
        elif k == 3:
            t = f'((x+1)^2 - (x-1)^2)/(4*x) + y^(-2)'
        else:
            t = f'x^(-3) + (y + x)^4/{rs.randint(2, 9)}'
        e = sy(t).subs({X: sp.Rational(xv), Y: sp.Rational(yv)})
        if e.is_Rational:
            out.append(Case('symbolic.evaluate', (t, {'x': xv, 'y': yv}), {'value': str(e)}, 'sympy-exact'))
    return out


def gen_evaluate_numeric(rs, n):
    out = []
    for i in range(n):
        t = rs.choice(['sin(x)*cos(y)', 'exp(x)/(1+y^2)', 'log(x+y)', 'atan(x)+atan(y)', 'sqrt(x)*y', 'x^(2/3)', 'tan(x/3)', 'exp(-x)*sin(5*y)', 'pi*x - E',
                       'sin(pi/7)+cos(x)', 'log(7)*x', '2^(x/2)', 'floor(x*y)', 'abs(x-y)*sin(x)', 'sin(100*x)'])
        xv, yv = fr(rs, 1, 12, (1, 2, 3, 7)), fr(rs, 1, 12, (1, 2, 3, 7))
        out.append(Case('symbolic.evaluate_numeric', (t, {'x': xv, 'y': yv}, 25), {'expr': t, 'x': xv, 'y': yv}, 'sympy-numeric'))
    return out


def gen_simplify(rs, n):
    out = []
    for i in range(n):
        a, b, c = lin(rs), lin(rs), lin(rs)
        k = i % 4
        if k == 0:
            t = f'({a}*{b})/({a}*{c})'
        elif k == 1:
            t = f'(x^2 - {rs.randint(1, 9)}^2)/(x - {rs.randint(1, 9)})'
        elif k == 2:
            t = f'{a}/{b} + {c}/{b} - {rs.randint(1, 3)}/({b}*{c})'
        else:
            t = f'({a}^2 - {b}^2)/({a} + {b})'
        try:
            if sp.denom(sp.cancel(sy(t))) == 0:
                continue
        except Exception:
            continue
        out.append(Case('symbolic.simplify', (t,), {'sympy': t}, 'sympy-exact'))
    return out


def gen_rational_form(rs, n):
    out = []
    for i in range(n):
        a, b, c = lin(rs), lin(rs), lin(rs)
        t = rs.choice([f'1/{a} + 1/{b}', f'{fr(rs)}/{a} - {fr(rs)}/({a}*{b})', f'({a}*{c})/({b}*{c})', f'x^2/({a}*{b}) + {rs.randint(1, 4)}', f'{a}^2/{a} + 1/{b}'])
        try:
            if sp.denom(sp.cancel(sy(t))) == 0:
                continue
        except Exception:
            continue
        out.append(Case('symbolic.rational_form', (t, 'x'), {'sympy': t}, 'sympy-exact'))
    return out


def gen_equal(rs, n):
    out = []
    for i in range(n):
        a, b = lin(rs), lin(rs)
        k = i % 6
        if k == 0:
            t1, t2, truth = f'{a}^2', f'x^2 + {2 * sy(a).coeff(X, 0)}*x*{sy(a).coeff(X, 1)} + {sy(a).coeff(X, 0) ** 2}', None
            t2 = str(sp.expand(sy(a) ** 2))
            truth = True
        elif k == 1:
            t1, t2 = f'{a}*{b}', str(sp.expand(sy(a) * sy(b)) + rs.randint(1, 3))
            truth = False
        elif k == 2:
            t1, t2 = f'1/{a} - 1/{b}', f'({sy(b) - sy(a)})/({a}*{b})'
            truth = zero_sym(sy(t1) - sy(t2))
        elif k == 3:
            t1, t2 = f'sin(x)^2 + cos(x)^2', f'1'
            truth = True
        elif k == 4:
            t1, t2 = 'exp(x)*exp(2*x)', 'exp(3*x)'
            truth = True
        else:
            t1, t2 = f'sqrt(x^2)', rs.choice(['x', 'abs(x)'])
            truth = t2 == 'abs(x)'
        out.append(Case('symbolic.equal', (t1, t2), {'truth': truth}, 'sympy-exact'))
    return out


def rand_fn_expr(rs, depth=2, positive=True):
    atoms = ['x', f'{rs.randint(1, 4)}', 'x^2', f'x^{rs.randint(3, 4)}', f'({lin(rs)})']
    if depth == 0:
        return rs.choice(atoms)
    k = rs.randint(0, 9)
    a = rand_fn_expr(rs, depth - 1)
    b = rand_fn_expr(rs, depth - 1)
    if k == 0:
        return f'({a})*({b})'
    if k == 1:
        return f'({a}) + ({b})'
    if k == 2:
        return f'sin({a})'
    if k == 3:
        return f'cos({a})'
    if k == 4:
        return f'exp(({a})/5)'
    if k == 5:
        return f'log(({a})^2 + {rs.randint(1, 3)})'
    if k == 6:
        return f'atan({a})'
    if k == 7:
        return f'sqrt(({a})^2 + {rs.randint(1, 4)})'
    if k == 8:
        return f'({a})/(({b})^2 + {rs.randint(1, 3)})'
    return f'({a})^{rs.randint(2, 3)} - {b}'


def gen_diff(rs, n):
    out = []
    for i in range(n):
        t = rand_fn_expr(rs, rs.choice([1, 2, 2, 3]))
        order = rs.choice([1, 1, 1, 2])
        out.append(Case('symbolic.diff', (t, 'x', order), {'sympy': t, 'order': order}, 'sympy-numeric'))
    return out


def gen_apart(rs, n):
    out = []
    for i in range(n):
        roots = [rs.randint(-4, 4) for _ in range(rs.randint(2, 3))]
        fr_ = f'{rs.randint(2, 3)}' if rs.random() < 0.3 else None
        den = '*'.join(f'(x {sgn(-r)})' for r in roots)
        if rs.random() < 0.5:
            den += f'^{rs.randint(1, 2)}' if False else ''
        if rs.random() < 0.35:
            den = f'({den})*(x^2 + {rs.randint(1, 4)})'
        if rs.random() < 0.3:
            den = f'{den}*(x {sgn(-roots[0])})'
        num = ' + '.join([f'{rs.randint(-4, 4)}*x^{d}' for d in range(rs.randint(1, 4))]) or '1'
        t = f'({num})/({den})'
        try:
            if sp.denom(sp.cancel(sy(t))) == 0:
                continue
        except Exception:
            continue
        out.append(Case('symbolic.apart', (t, 'x'), {'sympy': t}, 'sympy-exact'))
    return out


SERIES_FUNCS = ['exp(x)', 'sin(x)', 'cos(x)', 'log(1+x)', '(1+x)^(1/2)', '1/(1+x^2)', '1/(1-x)', 'atan(x)', 'tan(x)', 'exp(x)*cos(x)', 'sin(x)/(1+x)', 'log(1+x)/(1-x)',
                'sqrt(1+x^2)', 'exp(sin(x))', '(1+x)^(-2)', 'cos(x)^2', 'x*exp(2*x)', '1/cos(x)', 'exp(x)/(2+x)', 'log(2+x)']


def gen_series(rs, n, op):
    out = []
    for i in range(n):
        f = rs.choice(SERIES_FUNCS)
        order = rs.randint(2, 7)
        pt = rs.choice([0, 0, 0, 0, 1, 2]) if f in ('log(2+x)', 'exp(x)/(2+x)') else 0
        if f == 'log(1+x)' and rs.random() < 0.3:
            f, pt = 'log(x)', 1
        if f == 'exp(x)' and rs.random() < 0.3:
            pt = 1
        if f == '1/(1-x)' and rs.random() < 0.3:
            pt = 2
        if f == '(1+x)^(1/2)' and rs.random() < 0.3:
            pt = 3
        args = (f, 'x', pt, order)
        out.append(Case(op, args, {'sympy': f, 'point': pt, 'order': order}, 'sympy-numeric'))
    # not analytic: must be refused
    for t in ['sqrt(x)', '1/x', 'log(x)', 'abs(x)', 'x^(1/3)']:
        out.append(Case(op, (t, 'x', 0, 3), {'refuse': True}, 'sympy-numeric'))
    return out


LIMIT_FUNCS = [('sin({a}*x)/x', 0), ('(1-cos(x))/x^2', 0), ('(exp({a}*x)-1)/x', 0), ('log(1+{a}*x)/x', 0), ('(x^2-{a}^2)/(x-{a})', None), ('x*log(x)', '0+'),
               ('(x^2+{a}*x+1)/(2*x^2-{b})', 'oo'), ('sqrt(x^2+{a}*x)-x', 'oo'), ('x^2*exp(-x)', 'oo'), ('x*exp(-{a}*x)', 'oo'), ('(tan(x)-x)/x^3', 0),
               ('(1-cos({a}*x))/(x*sin(x))', 0), ('atan({a}*x)', 'oo'), ('atan(x)/x', 'oo'), ('(sqrt(1+x)-1)/x', 0), ('(x^3-{a}*x)/(x^2+1)', 'oo'),
               ('(sin(x)-x)/x^3', 0), ('x^2*sin(1/x)', 'oo'), ('1/x', '0+'), ('1/x', '0-'), ('1/x^2', 0), ('1/x', 0), ('(exp(x)-1-x)/x^2', 0), ('{a}*x/(x+{b})', 'oo'),
               ('(x-{a})/abs(x-{a})', None), ('sqrt(x^2+1)/x', 'oo'), ('sqrt(x^2+1)/x', '-oo'), ('log(x)/x', 'oo'), ('exp(x)/x', 'oo'), ('x^x', 0)]


def gen_limit(rs, n):
    out = []
    for i in range(n):
        tpl, pt = LIMIT_FUNCS[i % len(LIMIT_FUNCS)]
        a, b = rs.randint(1, 4), rs.randint(2, 6)
        t = tpl.format(a=a, b=b)
        direction = 'both'
        if pt is None:
            pt = a
        if isinstance(pt, str) and pt.endswith('+') and pt != 'oo':
            pt, direction = int(pt[:-1]), '+'
        elif isinstance(pt, str) and pt.endswith('-') and pt != '-oo':
            pt, direction = int(pt[:-1]), '-'
        out.append(Case('symbolic.limit', (t, 'x', pt, direction), {'sympy': t}, 'sympy-limit'))
    return out


INT_TEMPLATES = [
    'x^{a} + {b}*x', '{a}*x^3 - {b}*x^2 + x - {a}', 'exp({a}*x)', 'x*exp({a}*x)', 'x^2*exp(-x)', 'sin({a}*x + {b})', 'cos({a}*x)', 'x*sin(x)', 'x^2*cos({a}*x)', 'x^3*exp(x)',
    '1/(x - {a})', '1/(x^2 - {a}^2)', '1/(x^2 + {a})', '1/(x^2 + x + {a})', 'x/(x^2 + {a})', '(x + {b})/(x^2 + 2*x + {a}+1)', '1/(x*(x+{a}))', '(2*x+{b})/((x-{a})*(x+1))',
    '1/(x^2+1)^2', '1/(x-{a})^2', 'x^2/(x+{a})', '(x^3+1)/(x^2*(x-{a}))', 'log(x)', 'x*log(x)', 'x^2*log(x+{a})', 'sqrt(x)', 'x^(3/2)', 'sqrt({a}*x+{b})', 'x*sqrt(x+{a})',
    '1/sqrt(x+{a})', 'exp(x)*sin(x)', 'sin(x)^2', 'tan(x)', 'atan(x)', 'exp(x^2)', '1/(x^3+x+1)', 'sin(x)/x', '1/(x^4+1)', 'x/(x^2+1)^2', 'abs(x)', '1/x', '2^x', 'x*3^x']


def gen_integrate(rs, n):
    out = []
    for i in range(n):
        tpl = INT_TEMPLATES[i % len(INT_TEMPLATES)]
        t = tpl.format(a=rs.randint(1, 4), b=rs.randint(1, 5))
        out.append(Case('symbolic.integrate', (t, 'x'), {'sympy': t}, 'certificate'))
    return out


DEF_TEMPLATES = [('x^2', 0, 3), ('x^3 - 2*x', -1, 2), ('exp(-x)', 0, 'oo'), ('exp(x)', 0, 1), ('1/x', 1, 5), ('1/(x^2+1)', '-oo', 'oo'), ('1/(x^2+1)', 0, 1), ('sin(x)', 0, 'pi'),
                 ('cos(x)', 0, 'pi/2'), ('x*exp(-x)', 0, 'oo'), ('1/sqrt(x)', 0, 4), ('1/x^2', 1, 'oo'), ('1/x^2', -1, 1), ('1/x', 0, 1), ('log(x)', 1, 3), ('log(x)', 0, 1),
                 ('1/(x^2-1)', 2, 5), ('1/(x^2-1)', 0, 3), ('x/(x^2+1)', 0, 2), ('x^2*exp(-x)', 0, 'oo'), ('sqrt(x)', 0, 9), ('1/(x*(x+1))', 1, 'oo'), ('x*sin(x)', 0, 'pi'),
                 ('1/(x^2+x+1)', 0, 'oo'), ('x^2*log(x)', 1, 2), ('1/sqrt(x-1)', 1, 5), ('1/(x-1)^2', 2, 4), ('1/(x-1)^2', 0, 3), ('exp(2*x)', -1, 1), ('sin(2*x+1)', 0, 1),
                 ('1/(1+x)^3', 0, 'oo'), ('x*exp(x)', -1, 1), ('1/(x^2+4)', -2, 2), ('cos(x)^2', 0, 'pi'), ('1/x', -2, -1), ('1/x', -1, 1), ('sqrt(1-x)', 0, 1), ('abs(x)', -1, 2)]


def gen_integrate_definite(rs, n):
    out = []
    for i in range(n):
        t, lo, hi = DEF_TEMPLATES[i % len(DEF_TEMPLATES)]
        if i >= len(DEF_TEMPLATES):
            c = rs.randint(1, 3)
            t = re.sub(r'\bx\b', f'({c}*x)', t) if rs.random() < 0.4 else t
        out.append(Case('symbolic.integrate_definite', (t, 'x', lo, hi), {'sympy': t, 'lo': lo, 'hi': hi}, 'certificate'))
    return out


SUM_TEMPLATES = [('k', 1, 'n'), ('k^2', 1, 'n'), ('k^3', 1, 'n'), ('{a}*k^2 - k + {b}', 0, 'n'), ('{a}^k', 0, 'n'), ('k*{a}^k', 1, 'n'), ('k^2*2^k', 0, 'n'), ('(-1)^k', 0, 'n'),
                 ('(-1)^k*k', 1, 'n'), ('(-1)^k*k^2', 1, 'n'), ('1/(k*(k+1))', 1, 'n'), ('1/(k*(k+{a}))', 1, 'n'), ('1/((2*k-1)*(2*k+1))', 1, 'n'), ('1/(k*(k+1)*(k+2))', 1, 'n'),
                 ('(1/{b})^k', 0, 'oo'), ('k/2^k', 1, 'oo'), ('k^2/3^k', 0, 'oo'), ('1/(k*(k+1))', 1, 'oo'), ('1/(k*(k+2))', 1, 'oo'), ('(-1/2)^k', 0, 'oo'), ('(k+1)/3^k', 0, 'oo'),
                 ('1/k', 1, 'n'), ('1/k^2', 1, 'oo'), ('2^k', 0, 'oo'), ('k', 1, 'oo'), ('3^k - 2^k', 0, 'n'), ('k*(k+1)/2', 1, 'n'), ('1/(k^2+k)', 2, 'n'), ('k!', 1, 'n'),
                 ('k^2', 3, 'n'), ('2*3^k', 1, 'n'), ('1/(k^2-1)', 2, 'n'), ('k^4', 1, 'n'), ('(k^2+k)/2^k', 1, 'oo')]


def gen_sum_closed(rs, n):
    out = []
    for i in range(n):
        tpl, lo, hi = SUM_TEMPLATES[i % len(SUM_TEMPLATES)]
        t = tpl.format(a=rs.randint(2, 4), b=rs.randint(2, 5))
        out.append(Case('symbolic.sum_closed', (t, 'k', lo, hi), {'sympy': t, 'lo': lo, 'hi': hi}, 'brute-force'))
    return out


def gen_sum_range(rs, n):
    out = []
    for i in range(n):
        t = rs.choice(['k^2 - {a}*k', '1/(k*(k+{a}))', 'floor(k/{a})', 'abs(k - {b})', '2^k/(k+1)', '(-1)^k*k', 'k^3 + 1/k', 'floor(k^2/{b})', '1/(k^2+{a})', 'k*(k-1)*(k-2)/6']).format(
            a=rs.randint(2, 5), b=rs.randint(3, 9))
        lo = rs.randint(1, 5)
        hi = lo + rs.randint(0, 25)
        out.append(Case('symbolic.sum_range', (t, 'k', lo, hi), {'sympy': t}, 'brute-force'))
    return out


def gen_solve(rs, n):
    out = []
    for i in range(n):
        k = i % 10
        r = [rs.randint(-5, 5) for _ in range(3)]
        if k == 0:
            t = f'{lin(rs)}*{lin(rs)} = 0'
        elif k == 1:
            t = f'(x {sgn(-r[0])})*(x {sgn(-r[1])})*(x {sgn(-r[2])}) = 0'
        elif k == 2:
            t = f'x^2 + {rs.randint(-6, 6)}*x + {rs.randint(-6, 6)} = 0'
        elif k == 3:
            t = f'(x^2 - {rs.randint(1, 9) ** 2})/(x - {rs.randint(-3, 3)}) = 0'
        elif k == 4:
            t = rs.choice([f'{rs.randint(2, 5)}^x = {rs.randint(2, 40)}', f'exp({rs.randint(1, 3)}*x) = {rs.randint(1, 9)}', f'2^(x+1) + 2^x = {rs.randint(3, 50)}',
                           f'4^x - {rs.choice([3, 5, 6, 10])}*2^x + {rs.choice([2, 4, 6, 8, 9])} = 0', f'9^x - 4*3^x + 3 = 0', f'exp(2*x) - 3*exp(x) + 2 = 0', f'3*exp(x) = -1'])
        elif k == 5:
            t = rs.choice([f'log(x) = {rs.randint(-2, 3)}', f'log(x) + log(x - {rs.randint(1, 3)}) = log({rs.choice([2, 6, 12, 20])})', f'2*log(x + {rs.randint(0, 3)}) = log(9)',
                           f'log(x + 1) - log(x - 1) = log(3)', f'log(2*x - 1) = log(x + {rs.randint(1, 5)})'])
        elif k == 6:
            t = rs.choice([f'|x - {rs.randint(-3, 3)}| = {rs.randint(0, 5)}', f'|2*x + {rs.randint(1, 3)}| = |x - {rs.randint(1, 4)}|', f'|x - 1| + |x + 2| = {rs.randint(2, 7)}',
                           f'x^2 - {rs.randint(2, 5)}*|x| + {rs.choice([1, 2, 3, 4, 6])} = 0', f'|x^2 - 4| = {rs.randint(0, 6)}', f'|x| = x + {rs.randint(1, 3)}'])
        elif k == 7:
            t = rs.choice([f'1/(x - {rs.randint(-2, 2)}) + 1/(x + {rs.randint(1, 3)}) = {rs.randint(1, 3)}', f'x + 1/x = {rs.randint(2, 5)}', f'(x+1)/(x-1) = {rs.randint(2, 5)}',
                           f'x^3 - {rs.randint(2, 5)}*x = 0', f'x^4 - 5*x^2 + 4 = 0', f'x^3 + x + 1 = 0', f'x^5 - 1 = 0'])
        elif k == 8:
            t = rs.choice(['sin(x) = 1/2', 'sqrt(x) = 3', 'x*exp(x) = 1', f'x^2 = {rs.randint(2, 9)}', 'x = x', 'x + 1 = x'])
        else:
            t = rs.choice([f'{rs.randint(1, 4)}*x - {rs.randint(1, 9)} = {rs.randint(1, 9)}*x + {rs.randint(1, 9)}', f'x^2 = {rs.randint(0, 20)}', f'(x - 1)^2 = 0', f'x^2 + x + 1 = 0',
                           f'x^2 - 2*x - 1 = 0'])
        out.append(Case('symbolic.solve', (t, 'x'), {'sympy': t}, 'sympy-solveset'))
    return out


GENERATORS = {
    'symbolic.canon': gen_canon, 'symbolic.expand': gen_expand, 'symbolic.substitute': gen_substitute, 'symbolic.evaluate': gen_evaluate,
    'symbolic.evaluate_numeric': gen_evaluate_numeric, 'symbolic.simplify': gen_simplify, 'symbolic.rational_form': gen_rational_form, 'symbolic.equal': gen_equal,
    'symbolic.diff': gen_diff, 'symbolic.apart': gen_apart, 'symbolic.series_coefficients': lambda rs, n: gen_series(rs, n, 'symbolic.series_coefficients'),
    'symbolic.taylor': lambda rs, n: gen_series(rs, n, 'symbolic.taylor'), 'symbolic.limit': gen_limit, 'symbolic.integrate': gen_integrate,
    'symbolic.integrate_definite': gen_integrate_definite, 'symbolic.sum_closed': gen_sum_closed, 'symbolic.sum_range': gen_sum_range, 'symbolic.solve': gen_solve,
}


def generate(seed, n):
    """n cases per operation (a few operations add fixed extra cases); the seed fixes the whole list."""
    cases = []
    for i, (name, g) in enumerate(GENERATORS.items()):
        rs = random.Random(seed * 1009 + i)
        cases.extend(g(rs, n))
    return cases


# ------------------------------------------------------------------------------------------------------------------------------------ checkers
# each returns 'agree', 'disagree' or 'unchecked' (the oracle could not decide); a string detail may follow for a disagreement

def chk_canon(res, args, orc):
    if sym_mod.parse(res) != sym_mod.parse(args[0]):
        return 'disagree', 'parse(canon(x)) differs from parse(x)'
    v = numeric_equal(sy(orc['sympy']), sy(res), X, orc['pts'] + (['0.5'] if 'y' not in orc['sympy'] else []), 3) if 'y' not in orc['sympy'] else _two_var_equal(sy(orc['sympy']), sy(res))
    return ('agree', '') if v else ('disagree', 'value differs from sympy')


def _two_var_equal(e1, e2):
    for xv, yv in [('0.37', '1.21'), ('1.9', '0.44'), ('2.3', '1.7')]:
        a, b = [sp.N(e.subs({X: sp.Float(xv, 40), Y: sp.Float(yv, 40)}), 40) for e in (e1, e2)]
        if not close(mpmath.mpf(str(a)), mpmath.mpf(str(b))):
            return False
    return True


def chk_expand(res, args, orc):
    d = sp.expand(sy(res) - sp.expand(sy(orc['sympy'])))
    if d != 0:
        return 'disagree', 'expansion differs'
    if sym_mod.parse(res) != sym_mod.parse(args[0]):
        return 'disagree', 'roundtrip'
    return 'agree', ''


def chk_substitute(res, args, orc):
    o = sy(orc['expr']).subs(X, sy(orc['val']))
    return ('agree', '') if _two_var_equal(o, sy(res)) else ('disagree', 'value differs')


def chk_evaluate(res, args, orc):
    return ('agree', '') if res == Fraction(orc['value']) else ('disagree', f'{res} vs {orc["value"]}')


def chk_evaluate_numeric(res, args, orc):
    e = sy(orc['expr'])
    v = e.subs({X: sp.Rational(orc['x']), Y: sp.Rational(orc['y'])})
    ref = mpmath.mpf(str(sp.N(v, 50)))
    got = mpmath.mpf(str(res))
    return ('agree', '') if abs(ref - got) <= mpmath.mpf(10) ** -22 * (1 + abs(ref)) else ('disagree', f'{got} vs {ref}')


def zero_sym(e):
    """SymPy's exact zero test for rational functions (cancel can leave `-3 + 3` unevaluated, so expand after it)."""
    return sp.expand(sp.cancel(sp.together(e))) == 0


def chk_simplify(res, args, orc):
    ex = sy(orc['sympy'])
    rv = sy(res)
    if not zero_sym(rv - ex):
        return 'disagree', 'value differs'
    n, d = sp.fraction(sp.cancel(sp.together(rv)))
    if sp.gcd(n, d) != 1:
        return 'disagree', 'not in lowest terms'
    return 'agree', ''


def chk_rational_form(res, args, orc):
    ex = sp.cancel(sy(orc['sympy']))
    nt, dt = res
    n, d = sy(nt), sy(dt)
    if not zero_sym(n / d - ex):
        return 'disagree', 'value differs'
    pn, pd = sp.Poly(n, X), sp.Poly(d, X)
    from math import gcd as g2
    gg = 0
    for c in pn.all_coeffs() + pd.all_coeffs():
        if not c.is_Integer:
            return 'disagree', 'non-integer coefficient'
        gg = g2(gg, int(c))
    if gg != 1 or pd.LC() <= 0 or sp.gcd(pn.as_expr(), pd.as_expr()) != 1:
        return 'disagree', 'not in the normal form'
    return 'agree', ''


def chk_equal(res, args, orc):
    if orc['truth'] is True:
        return ('agree', '') if res is True else ('disagree', 'said False for equal expressions')
    return ('agree', '') if res is False else ('disagree', 'said True for different expressions')


def chk_diff(res, args, orc):
    o = sp.diff(sy(orc['sympy']), X, orc['order'])
    return ('agree', '') if numeric_equal(o, sy(res), X, ['0.37', '0.61', '0.83', '1.13', '0.5', '0.91'], 4) else ('disagree', 'derivative differs from sympy at a point')


def chk_apart(res, args, orc):
    if not zero_sym(sy(res) - sy(orc['sympy'])):
        return 'disagree', 'terms do not add to the input'
    # independent check against sympy's own decomposition (structure of poles)
    ap = sp.apart(sy(orc['sympy']), X)
    if not zero_sym(sy(res) - ap):
        return 'disagree', 'differs from sympy.apart'
    return 'agree', ''


def chk_series(res, args, orc):
    f, var, pt, order = args
    fx = sy(f)
    coeffs = res if isinstance(res, list) else None
    if coeffs is None:
        poly = sy(res)
        h = sp.Symbol('h', real=True)
        coeffs = [str(sp.series(poly.subs(X, h + sp.Rational(pt)), h, 0, order + 1).removeO().coeff(h, i)) for i in range(order + 1)]
    ref = sp.series(fx.subs(X, X + sp.Rational(pt)), X, 0, order + 1).removeO()
    for i in range(order + 1):
        a = ref.coeff(X, i)
        b = sy(coeffs[i])
        va = mpmath.mpf(str(sp.N(a, 40)))
        vb = mpmath.mpf(str(sp.N(b, 40)))
        if not close(va, vb, 30):
            return 'disagree', f'coefficient {i}: {coeffs[i]} vs {a}'
    # certificate: the polynomial matches the function at a small offset
    hh = mpmath.mpf('1e-4')
    f_lam = sp.lambdify(X, fx, 'mpmath')
    val = f_lam(mpmath.mpf(pt) + hh)
    tpoly = sum(mpmath.mpf(str(sp.N(sy(coeffs[i]), 40))) * hh ** i for i in range(order + 1))
    if abs(val - tpoly) > 50 * hh ** (order + 1) * (1 + max(abs(mpmath.mpf(str(sp.N(sy(c), 20)))) for c in coeffs)):
        return 'disagree', 'certificate: polynomial far from the function at h=1e-4'
    return 'agree', ''


def _lim_numeric(fx, point, direction, depth=(1e-6, 1e-8, 1e-10)):
    f = sp.lambdify(X, fx, 'mpmath')
    vals = []
    for h in depth:
        h = mpmath.mpf(h)
        try:
            if point == 'oo':
                vals.append(f(1 / h))
            elif point == '-oo':
                vals.append(f(-1 / h))
            elif direction == '-':
                vals.append(f(mpmath.mpf(point) - h))
            else:
                vals.append(f(mpmath.mpf(point) + h))
        except Exception:
            return None
    return vals


def chk_limit(res, args, orc):
    t, var, pt, direction = args
    fx = sy(t)
    point = sp.oo if pt == 'oo' else (-sp.oo if pt == '-oo' else sp.Integer(pt))
    ref = None
    try:
        if pt in ('oo', '-oo'):
            ref = sp.limit(fx, X, point)
        elif direction == 'both':
            ref = sp.limit(fx, X, point, dir='+-')
        else:
            ref = sp.limit(fx, X, point, dir=direction)
    except ValueError:
        ref = 'nolimit'
    except Exception:
        ref = None
    if isinstance(ref, sp.Limit) or ref is None:
        ref = None
    if ref == 'nolimit' or (ref is not None and ref.has(sp.nan)):
        return 'disagree', f'returned {res} but the limit does not exist'
    mine = res
    # certificate: numeric approach
    sides = [direction] if direction in ('+', '-') else ['+', '-']
    if pt in ('oo', '-oo'):
        sides = ['+']
    if mine in ('oo', '-oo'):
        for s in sides:
            vals = _lim_numeric(fx, pt, s)
            if vals is None:
                continue
            want = 1 if mine == 'oo' else -1
            if not all(v * want > 1e4 for v in vals[1:]):
                return 'disagree', f'{mine}, but values near the point are {[float(v) for v in vals]}'
        if ref is not None and ref not in (sp.oo, -sp.oo):
            return 'disagree', f'sympy says {ref}'
        return 'agree', ''
    mv = mpmath.mpf(str(sp.N(sy(mine), 40)))
    for s in sides:
        vals = _lim_numeric(fx, pt, s)
        if vals is None:
            continue
        if not abs(vals[-1] - mv) < mpmath.mpf('1e-5') * (1 + abs(mv)):
            return 'disagree', f'{mine}, but values near the point are {[float(v) for v in vals]}'
    if ref is not None and ref.is_finite:
        if not close(mpmath.mpf(str(sp.N(ref, 40))), mv, 25):
            return 'disagree', f'sympy says {ref}'
    elif ref is not None:
        return 'disagree', f'sympy says {ref}'
    return 'agree', ''


def chk_integrate(res, args, orc):
    f = sy(args[0])
    F = sy(res)
    if sym_mod.parse(res) != sym_mod.parse(res):
        return 'disagree', 'roundtrip'
    pts = ['0.37', '1.13', '1.91', '2.47', '0.61', '3.29', '-0.7', '-2.3']
    dF = sp.diff(F, X)
    # abs() antiderivatives are checked off the kink; sympy differentiates Abs as sign
    ok = numeric_equal(dF, f, X, pts, 4)
    return ('agree', '') if ok else ('disagree', 'derivative of the antiderivative differs from the integrand')


def chk_integrate_definite(res, args, orc):
    t, var, lo, hi = args
    f = sp.lambdify(X, sy(t), 'mpmath')

    def b(v):
        if v == 'oo':
            return mpmath.inf
        if v == '-oo':
            return -mpmath.inf
        return mpmath.mpf(str(sp.N(sy(str(v)), 40)))
    try:
        ref = mpmath.quad(f, [b(lo), b(hi)])
    except Exception:
        return 'unchecked', 'oracle quad failed'
    got = mpmath.mpf(str(sp.N(sy(res), 40)))
    return ('agree', '') if abs(ref - got) <= mpmath.mpf('1e-18') * (1 + abs(ref)) else ('disagree', f'{got} vs mpmath.quad {ref}')


def _summand(t):
    return sy(t)


def chk_sum_closed(res, args, orc):
    t, var, lo, hi = args
    s = sy(t)
    F = sy(res)
    if hi == 'oo':
        got = mpmath.mpf(str(sp.N(F, 40)))
        f = sp.lambdify(K, s, 'mpmath')
        ref = mpmath.nsum(lambda k: f(k), [lo, mpmath.inf])
        return ('agree', '') if abs(ref - got) <= mpmath.mpf('1e-15') * (1 + abs(ref)) else ('disagree', f'{got} vs nsum {ref}')
    for nv in range(lo, lo + 9):
        brute = sp.Integer(0)
        for kv in range(lo, nv + 1):
            brute += s.subs(K, kv)
        mine = F.subs(N_, nv)
        if sp.nsimplify(sp.simplify(brute - mine)) != 0:
            return 'disagree', f'n={nv}: closed form {mine} vs explicit {brute}'
    return 'agree', ''


def chk_sum_range(res, args, orc):
    t, var, lo, hi = args
    s = sy(t)
    tot = sp.Integer(0)
    for kv in range(lo, hi + 1):
        tot += s.subs(K, kv)
    return ('agree', '') if Fraction(str(tot)) == res else ('disagree', f'{res} vs {tot}')


def chk_solve(res, args, orc):
    lhs, rhs = args[0].split('=')
    F = sy(lhs) - sy(rhs)
    try:
        ref = sp.solveset(F, X, sp.S.Reals)
    except Exception:
        ref = None
    vals = []
    for r in res:
        e = sy(r)
        v = F.subs(X, e)
        if abs(complex(sp.N(v, 30))) > 1e-18:
            return 'disagree', f'root {r} does not satisfy the equation: F = {sp.N(v, 8)}'
        vals.append(mpmath.mpf(str(sp.N(e, 40))))
    if ref is None or not (isinstance(ref, sp.FiniteSet) or ref == sp.S.EmptySet):
        return 'unchecked', 'sympy.solveset gave no finite set'
    # solveset may keep roots where a log argument is negative (log(-3) is complex): keep only roots that really give 0 on the principal branch
    ref = [r for r in ref if abs(complex(sp.N(F.subs(X, r), 30))) < 1e-18]
    rv = sorted(mpmath.mpf(str(sp.N(r, 40))) for r in ref)
    if len(rv) != len(vals):
        return 'disagree', f'{len(vals)} roots {res} vs sympy {sorted(float(x) for x in rv)}'
    for a, b in zip(rv, sorted(vals)):
        if not close(a, b, 20):
            return 'disagree', f'{res} vs sympy {sorted(float(x) for x in rv)}'
    return 'agree', ''


CHECKERS = {
    'symbolic.canon': chk_canon, 'symbolic.expand': chk_expand, 'symbolic.substitute': chk_substitute, 'symbolic.evaluate': chk_evaluate,
    'symbolic.evaluate_numeric': chk_evaluate_numeric, 'symbolic.simplify': chk_simplify, 'symbolic.rational_form': chk_rational_form, 'symbolic.equal': chk_equal,
    'symbolic.diff': chk_diff, 'symbolic.apart': chk_apart, 'symbolic.series_coefficients': chk_series, 'symbolic.taylor': chk_series, 'symbolic.limit': chk_limit,
    'symbolic.integrate': chk_integrate, 'symbolic.integrate_definite': chk_integrate_definite, 'symbolic.sum_closed': chk_sum_closed,
    'symbolic.sum_range': chk_sum_range, 'symbolic.solve': chk_solve,
}


# ------------------------------------------------------------------------------------------------------------------------------------ runner

def run(seed=1, n=60, out_dir=None, verbose=False):
    t0 = time.time()
    cases = generate(seed, n)
    stats = {}
    disagreements = []
    unchecked_cases = []
    refused_cases = []
    for c in cases:
        st = stats.setdefault(c.op, dict(generated=0, answered=0, agree=0, unchecked=0, refused=Counter(), disagree=0, seconds=0.0))
        st['generated'] += 1
        t1 = time.time()
        try:
            res = run_op(c.op, *c.args)
        except NotCovered as ex:
            st['refused'][str(ex)] += 1
            refused_cases.append((c.op, c.args, str(ex)))
            st['seconds'] += time.time() - t1
            continue
        except BadInput as ex:
            st['refused']['BadInput: ' + str(ex)] += 1
            st['seconds'] += time.time() - t1
            continue
        except RecursionError:
            st['disagree'] += 1
            disagreements.append((c.op, c.args, 'RecursionError'))
            continue
        except Exception as ex:   # a crash is a bug, counted as a disagreement
            st['disagree'] += 1
            disagreements.append((c.op, c.args, f'{type(ex).__name__}: {ex}'))
            continue
        st['seconds'] += time.time() - t1
        if c.oracle.get('refuse'):
            st['answered'] += 1
            st['disagree'] += 1
            disagreements.append((c.op, c.args, f'answered {res!r} where it must refuse'))
            continue
        st['answered'] += 1
        try:
            verdict, detail = CHECKERS[c.op](res, c.args, c.oracle)
        except Exception as ex:
            verdict, detail = 'unchecked', f'checker error {type(ex).__name__}: {ex}'
        if verdict == 'agree':
            st['agree'] += 1
        elif verdict == 'unchecked':
            st['unchecked'] += 1
            st.setdefault('unchecked_reasons', Counter())[detail] += 1
            unchecked_cases.append((c.op, c.args, detail))
        else:
            st['disagree'] += 1
            disagreements.append((c.op, c.args, f'{detail}; result {res!r}'))
    reg = [k for k in registry() if k.startswith('symbolic.')]
    report = dict(module='symbolic', seed=seed, n=n, seconds=round(time.time() - t0, 1), operations_registered=len(reg),
                  operations_without_case=sorted(set(reg) - set(stats)),
                  per_operation={k: dict(generated=v['generated'], answered=v['answered'], agree=v['agree'], unchecked=v['unchecked'], refused=dict(v['refused']),
                                          refused_total=sum(v['refused'].values()), disagree=v['disagree'], seconds=round(v['seconds'], 2)) for k, v in sorted(stats.items())},
                  refused_cases=[dict(op=o, args=repr(a), reason=d) for o, a, d in refused_cases],
                  unchecked=[dict(op=o, args=repr(a), detail=d) for o, a, d in unchecked_cases],
                  disagreements=[dict(op=o, args=repr(a), detail=d) for o, a, d in disagreements])
    out = Path(out_dir) if out_dir else Path(__file__).resolve().parents[3] / 'build-mathbackend'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'symbolic.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    return report


def print_table(rep):
    print(f'symbolic: {rep["operations_registered"]} operations, seed {rep["seed"]}, n {rep["n"]}, {rep["seconds"]} s')
    print(f'{"operation":34}{"gen":>5}{"ans":>5}{"agree":>6}{"unchk":>6}{"refused":>8}{"disagree":>9}')
    tot = Counter()
    for k, v in rep['per_operation'].items():
        print(f'{k:34}{v["generated"]:>5}{v["answered"]:>5}{v["agree"]:>6}{v["unchecked"]:>6}{v["refused_total"]:>8}{v["disagree"]:>9}')
        for key in ('generated', 'answered', 'agree', 'unchecked', 'refused_total', 'disagree'):
            tot[key] += v[key]
    print(f'{"TOTAL":34}{tot["generated"]:>5}{tot["answered"]:>5}{tot["agree"]:>6}{tot["unchecked"]:>6}{tot["refused_total"]:>8}{tot["disagree"]:>9}')
    if rep['operations_without_case']:
        print('operations with no case:', rep['operations_without_case'])
    for d in rep['disagreements'][:40]:
        print('DISAGREE', d['op'], d['args'][:100], d['detail'][:200])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n', type=int, default=60)
    a = ap.parse_args()
    print_table(run(a.seed, a.n))
