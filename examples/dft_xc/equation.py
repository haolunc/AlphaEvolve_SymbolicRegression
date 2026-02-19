def equation(rho: jnp.ndarray, s: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Computes exchange-correlation energy density (e_xc) using PBE exchange without correlation.

    Args:
        rho: Electron density at each grid point.
        s: Reduced density gradient, defined as s = |∇ρ|/(2k_F*ρ) where k_F = (3π²ρ)^(1/3).
        params: Array of optimizable numeric constants or parameters.

    Returns:
        Two arrays representing exchange energy density (e_x) and correlation energy density (e_c).

    Physical interpretation:
        - LDA exchange: e_x^LDA = -C_x * ρ^(4/3)
        - Gradient corrections via the reduced gradient s improve upon LDA using the PBE functional form.
    """
    # LDA exchange energy density: proportional to ρ^(4/3)
    e_x_lda = params[0] * rho**(4/3)

    # ========== PBE Exchange ==========
    # Extract PBE parameters
    kappa = params[1]
    mu = params[2]

    # PBE enhancement factor: F_x(s) = 1 + κ - κ/(1 + μs²/κ)
    F_x = 1.0 + kappa - kappa / (1.0 + mu * s**2 / kappa)

    # Apply PBE enhancement to LDA exchange
    e_x = e_x_lda * F_x

    # Correlation term not yet implemented
    e_c = 0

    return e_x, e_c
