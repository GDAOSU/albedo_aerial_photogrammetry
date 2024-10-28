from sympy.matrices import Matrix, eye, zeros, ones, diag
from symtable import Symbol
from tkinter import Y
from sympy import Symbol, simplify, diff
from sympy.interactive.printing import init_printing
from sympy.printing.cxx import CXX11CodePrinter
init_printing(use_unicode=False, wrap_line=False)


def process_expr(ex, subtable):
    return CXX11CodePrinter().doprint(simplify(ex.subs(subtable)).subs(subtable))


def vec_to_mat(v):
    return Matrix([[v[1], v[2], v[0]],
                   [v[4], v[5], v[3]],
                   [0, 0, 1]])


def mat_to_vec(m):
    return Matrix([m[0, 2], m[0, 0], m[0, 1], m[1, 2], m[1, 0], m[1, 1]])


k1 = Symbol('k1()')
k2 = Symbol('k2()')
k3 = Symbol('k3()')
k4 = Symbol('k4()')
p1 = Symbol('p1()')
p2 = Symbol('p2()')

x = Symbol('y[0]')
y = Symbol('y[1]')
mx2_u = x*x
my2_u = y*y
mxy_u = x*y

rho2_u = mx2_u + my2_u
rho4_u = rho2_u * rho2_u
rho6_u = rho4_u * rho2_u
rho8_u = rho4_u*rho4_u
rad_dist_u = k1 * rho2_u + k2*rho4_u + k3*rho6_u + k4*rho8_u

subtable = [(rad_dist_u, 'rad_dist_u'),
            (rho8_u, 'rho8_u'),
            (rho6_u, 'rho6_u'),
            (rho4_u, 'rho4_u'),
            (rho2_u, 'rho2_u'),
            (mxy_u, 'mxy_u'),
            (my2_u, 'my2_u'),
            (mx2_u, 'mx2_u')]

u = x + x * rad_dist_u + p1 * (rho2_u + 2. * mx2_u) + 2. * p2 * mxy_u
v = y+y * rad_dist_u + p2 * (rho2_u + 2. * my2_u) + 2. * p1 * mxy_u

Out = Matrix([u,v])
In = Matrix([x,y])
J = Out.jacobian(In)
for i in range(2):
  print(f'y[{i}]=',process_expr(Out[i],subtable),';')

print()
for i in range(2):
  for j in range(2):
    print(f'J({i},{j})=',process_expr(J[i,j],subtable),';')
  
# print(Out.jacobian(In).subs(subtable))


# process_expr(u)
