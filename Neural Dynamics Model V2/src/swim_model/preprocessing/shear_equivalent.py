from __future__ import annotations


def compute_dynamic_shear_components(tau_xy, tau_xz, tau_yz, xp):
    tau_xy_dyn = tau_xy - xp.mean(tau_xy, axis=-1, keepdims=True)
    tau_xz_dyn = tau_xz - xp.mean(tau_xz, axis=-1, keepdims=True)
    tau_yz_dyn = tau_yz - xp.mean(tau_yz, axis=-1, keepdims=True)
    return tau_xy_dyn, tau_xz_dyn, tau_yz_dyn
