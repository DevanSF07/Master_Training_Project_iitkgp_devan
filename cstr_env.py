import numpy as np
from scipy.integrate import solve_ivp
from config import Config

def cstr_ode(t, state, u_dev, F=Config.F_0, k_0=Config.k_0_t0):
    """
    CSTR governing differential equations (eq 14a & eq 14b in paper).
    state = [C_A, T] (absolute physical values)
    u_dev = [delta_C_A0, delta_Q] (deviation values)
    """
    C_A, T = state[0], state[1]
    
    # Absolute input values
    C_A0 = Config.C_A0s + u_dev[0]
    Q = Config.Q_s + u_dev[1]
    
    # Arrhenius reaction rate: k(T) = k_0 * exp(-E / (R * T))
    # Note: Safeguard temperature against division by zero or non-positive values
    T_safe = max(T, 100.0)
    k_T = k_0 * np.exp(-Config.E / (Config.R * T_safe))
    r_A = k_T * (C_A ** 2)
    
    # dC_A/dt = (F/V)*(C_A0 - C_A) - r_A
    dC_A_dt = (F / Config.V) * (C_A0 - C_A) - r_A
    
    # dT/dt = (F/V)*(T_0 - T) + (-ΔH / (ρ_L * C_p))*r_A + Q / (ρ_L * C_p * V)
    heat_coef = -Config.DELTA_H / (Config.RHO_L * Config.C_P)
    dT_dt = (F / Config.V) * (Config.T_0 - T) + heat_coef * r_A + Q / (Config.RHO_L * Config.C_P * Config.V)
    
    return [dC_A_dt, dT_dt]

class CSTREnvironment:
    """
    Simulates real-time CSTR plant dynamics and parameter disturbances.
    """
    def __init__(self, F=Config.F_0, k_0=Config.k_0_t0):
        self.F = F
        self.k_0 = k_0
        self.reset()
        
    def reset(self, initial_state_dev=None):
        if initial_state_dev is None:
            self.state_dev = np.array([0.0, 0.0])  # Equilibrium point (0, 0)
        else:
            self.state_dev = np.array(initial_state_dev, dtype=float)
        return self.state_dev.copy()
        
    def get_absolute_state(self):
        return np.array([
            Config.C_As + self.state_dev[0],
            Config.T_s + self.state_dev[1]
        ])
        
    def step(self, u_dev, dt=Config.DELTA):
        """
        Integrates CSTR dynamics forward by time dt given input deviation u_dev.
        """
        # Enforce physical saturation bounds
        u_dev_clamped = np.copy(u_dev)
        u_dev_clamped[0] = np.clip(u_dev_clamped[0], -Config.DELTA_C_A0_MAX, Config.DELTA_C_A0_MAX)
        u_dev_clamped[1] = np.clip(u_dev_clamped[1], -Config.DELTA_Q_MAX, Config.DELTA_Q_MAX)
        
        abs_state = self.get_absolute_state()
        
        sol = solve_ivp(
            fun=cstr_ode,
            t_span=(0, dt),
            y0=abs_state,
            args=(u_dev_clamped, self.F, self.k_0),
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        next_abs_state = sol.y[:, -1]
        
        # Convert back to deviation variables
        self.state_dev[0] = next_abs_state[0] - Config.C_As
        self.state_dev[1] = next_abs_state[1] - Config.T_s
        
        return self.state_dev.copy()
        
    def set_disturbance(self, F_ratio=1.0, k_0_ratio=1.0):
        """Applies parametric drift disturbance."""
        self.F = Config.F_0 * F_ratio
        self.k_0 = Config.k_0_t0 * k_0_ratio
        print(f" Applied disturbance: F = {self.F:.2f} m^3/h ({F_ratio*100:.0f}%), k_0 = {self.k_0:.2e} ({k_0_ratio*100:.0f}%)")
