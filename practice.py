import numpy as np
import sympy as sp
import torch
import z3

t = sp.symbols('t', real=True)
print(sp.diff(t**2, t))

x, y, vx, vy, ux, uy = sp.symbols('x y vx vy ux uy', real=True)
m = 12
n = 0.001027
fx = (4 - 3 * sp.cos(n)) * x + 0 * y + (1 / n * sp.sin(n)) * vx + (2 / n - 2 / n * sp.cos(n)) * vy + ((1 - sp.cos(n )) / (m * (n ** 2))) * ux + (2  / (m * n) - 2 * sp.sin(n ) / (m * (n ** 2))) * uy

fy = (-6 * n + 6 * sp.sin(n)) * x + 1 * y + (-2 / n + 2 / n * sp.cos(n)) * vx + (-3 + 4 / n * sp.sin(n)) * vy + ((-2) / (m * n) + (2 * sp.sin(n)) / (m * (n ** 2))) * ux + (4 / (m * (n ** 2)) - 3 / (2 * m) - (4 * sp.cos(n)) / (m * (n ** 2))) * uy

fvx = (3 * n * sp.sin(n)) * x + 0 * y + sp.cos(n) * vx + (2 * sp.sin(n)) * vy + (sp.sin(n) / (m * n)) * ux + (2 / (m * n) - (2 * sp.cos(n)) / (m * n)) * uy

fvy = (-6 * n + 6 * n * sp.cos(n)) * x + 0 * y + (-2 * sp.sin(n)) * vx + (-3 + 4 * sp.cos(n)) * vy + ((2 * sp.cos(n) - 2) / (m * n)) * ux + (-3 / (m) + (4 * sp.sin(n)) / (m * n)) * uy


fx_x, fx_y, fx_vx, fx_vy, fx_ux, fx_uy = sp.diff(fx, x), sp.diff(fx, y), sp.diff(fx, vx), sp.diff(fx, vy), sp.diff(fx, ux), sp.diff(fx, uy)
fy_x, fy_y, fy_vx, fy_vy, fy_ux, fy_uy = sp.diff(fy, x), sp.diff(fy, y), sp.diff(fy, vx), sp.diff(fy, vy), sp.diff(fy, ux), sp.diff(fy, uy)
fvx_x, fvx_y, fvx_vx, fvx_vy, fvx_ux, fvx_uy = sp.diff(fvx, x), sp.diff(fvx, y), sp.diff(fvx, vx), sp.diff(fvx, vy), sp.diff(fvx, ux), sp.diff(fvx, uy)
fvy_x, fvy_y, fvy_vx, fvy_vy, fvy_ux, fvy_uy = sp.diff(fvy, x), sp.diff(fvy, y), sp.diff(fvy, vx), sp.diff(fvy, vy), sp.diff(fvy, ux), sp.diff(fvy, uy)

print(fx_x, fx_y, fx_vx, fx_vy, fx_ux, fx_uy)

critical_points_x = sp.solve([fx_x, fx_y, fx_vx, fx_vy, fx_ux, fx_uy], [x, y, vx, vy, ux, uy], dict=True)
critical_points_y = sp.solve([fy_x, fy_y, fy_vx, fy_vy, fy_ux, fy_uy], [x, y, vx, vy, ux, uy], dict=True)
critical_points_vx = sp.solve([fvx_x, fvx_y, fvx_vx, fvx_vy, fvx_ux, fvx_uy], [x, y, vx, vy, ux, uy], dict=True)
critical_points_vy = sp.solve([fvy_x, fvy_y, fvy_vx, fvy_vy, fvy_ux, fvy_uy], [x, y, vx, vy, ux, uy], dict=True)
print(critical_points_x)
print(critical_points_y)
print(critical_points_vx)
print(critical_points_vy)

exit()

import numpy as np
from z3 import *

# ── Physical constants ──────────────────────────────────────────────
n_val, m_val = 0.001027, 12

c = [
    float(4 - 3 * np.cos(n_val)),                              # c_x
    0.0,                                                        # c_y  (zero coefficient)
    float(np.sin(n_val) / n_val),                              # c_vx
    float(2/n_val * (1 - np.cos(n_val))),                     # c_vy
    float((1 - np.cos(n_val)) / (m_val * n_val**2)),          # c_ux
    float(2/(m_val*n_val) - 2*np.sin(n_val)/(m_val*n_val**2)),# c_uy
]

lo   = [-6.0, -6.0, -0.5, -0.5, -1.0, -1.0]
hi   = [ 6.0,  6.0,  0.5,  0.5,  1.0,  1.0]
N    = 100
step = [(hi[d] - lo[d]) / N for d in range(6)]

# ── Fixed-point scaling ─────────────────────────────────────────────
SCALE = 1 << 20   # 2^20 ≈ 1M; output fits in ~30 signed bits → use 64-bit BVs
BW    = 64

# Precomputed integer constants (all just Python ints, no Z3 needed)
A = [int(round(c[d] * step[d] * SCALE)) for d in range(6)]          # per-index coefficients
B = int(round(sum(c[d] * lo[d] for d in range(6)) * SCALE))         # constant offset

# Correction for lower/upper bound (picks worst-case endpoint per dim)
C_lower = int(round(sum(min(0.0, c[d] * step[d]) for d in range(6)) * SCALE))
C_upper = int(round(sum(max(0.0, c[d] * step[d]) for d in range(6)) * SCALE))

print("Coefficients A:", A)
print("Offset B:", B)
print("Lower correction C-:", C_lower)
print("Upper correction C+:", C_upper)

# ── Z3 index variables (7-bit unsigned, constrained to 0-99) ────────
dims = ['x', 'y', 'vx', 'vy', 'ux', 'uy']
idx  = [BitVec(f'i_{d}', 7) for d in dims]
idx64 = [ZeroExt(57, i) for i in idx]          # promote to 64-bit for arithmetic

# ── Build scaled sum: B + C + sum_d A_d * i_d ───────────────────────
def build_fx_scaled(correction: int):
    total = BitVecVal(B + correction, BW)
    for d in range(6):
        total = total + BitVecVal(A[d], BW) * idx64[d]
    return total

fx_lower_scaled = build_fx_scaled(C_lower)
fx_upper_scaled = build_fx_scaled(C_upper)

# Unscale: floor (lower) and ceil (upper) via arithmetic right shift
# ceil(a / 2^k) = (a + 2^k - 1) >> k  — correct for all signed integers
fx_lower_bv = fx_lower_scaled >> 20
fx_upper_bv  = (fx_upper_scaled + BitVecVal(SCALE - 1, BW)) >> 20

# ── Domain validity constraints ──────────────────────────────────────
domain_constraints = [
    And(UGE(idx[d], 0), ULE(idx[d], 99)) for d in range(6)
]

# ── Example: verify fx stays within [-7, 7] for all valid inputs ─────
s = Solver()
s.add(domain_constraints)

# Look for a counterexample where bounds escape [-7, 7]
SEVEN = BitVecVal(int(7 * SCALE), BW)
NEG_SEVEN = BitVecVal(int(-7 * SCALE), BW)

s.add(Or(
    fx_lower_scaled < NEG_SEVEN,
    fx_upper_scaled > SEVEN,
))

result = s.check()
if result == sat:
    m = s.model()
    print("Counterexample found:")
    for d in range(6):
        print(f"  i_{dims[d]} = {m[idx[d]]}")
elif result == unsat:
    print("Verified: fx ∈ [-7, 7] for all valid discretized inputs.")
else:
    print("Unknown")
# # Create a 2D PyTorch tensor
# tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6] , [7, 8, 9]])
# print(tensor_2d[0, :] * tensor_2d[1, :])  # Element-wise multiplication of the first and second rows

# tensor = torch.tensor([1, 2, 0])
# print(tensor.unsqueeze(0).expand_as(tensor_2d))

# probs = torch.tensor([[0,0,1],[0.2,0.2,0.6],[0.9,0.1,0]])

# print(torch.multinomial(probs,num_samples=1))
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()

    return hook
    
model = torch.nn.Sequential(
    torch.nn.Linear(1, 1, bias=False),
    torch.nn.Linear(1, 1, bias=False),
    torch.nn.Linear(1, 1, bias=False)
)

with torch.no_grad():
    for layer in model:
        torch.nn.init.constant_(layer.weight, 2.0)
model[-3].register_forward_hook(get_features("bool_vec"))
x = torch.tensor([[[1],[2],[-3],[5]]], dtype=torch.float32)
y = model(x)
print("Output of the model:")
print(y)
print("Features captured by the hook:")
print(features["bool_vec"])


