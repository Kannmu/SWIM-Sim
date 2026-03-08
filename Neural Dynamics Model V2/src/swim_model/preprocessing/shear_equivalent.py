from __future__ import annotations


def compute_dynamic_shear(tau_xy, tau_xz, tau_yz, xp):
    tau_eq = xp.sqrt(tau_xy ** 2 + tau_xz ** 2 + tau_yz ** 2)
    tau_dyn = tau_eq - xp.mean(tau_eq, axis=-1, keepdims=True)
    return tau_eq, tau_dyn
