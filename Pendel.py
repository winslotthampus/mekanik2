import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

def setup_equations():
    print("Härleder ekvationer med SymPy (detta tar några sekunder)...")
    t = sp.Symbol('t')
    
    # Parametrar
    M, k_rad, d, L, m, g = sp.symbols('M k d L m g', real=True)
    tau = sp.Symbol('tau', real=True)
    
    # Tidsberoende funktioner
    theta_f = sp.Function('theta')(t)
    phi_f = sp.Function('phi')(t)
    theta_f_d = sp.diff(theta_f, t)
    phi_f_d = sp.diff(phi_f, t)
    
    # KINEMATIK: 
    # phi = 0 -> Sittande vinkelrät. phi = phi_max -> Stående längs med gungan (upprätt).
    x_B = L * sp.sin(theta_f)
    y_B = -L * sp.cos(theta_f)
    x_C = x_B + d * sp.cos(theta_f + phi_f)
    y_C = y_B + d * sp.sin(theta_f + phi_f)
    
    # Kinetisk energi T
    v_B_sq = sp.diff(x_B, t)**2 + sp.diff(y_B, t)**2
    v_C_sq = sp.diff(x_C, t)**2 + sp.diff(y_C, t)**2
    I_M = M * k_rad**2
    T_m = 0.5 * m * v_B_sq
    T_M = 0.5 * M * v_C_sq + 0.5 * I_M * (theta_f_d + phi_f_d)**2
    T = T_m + T_M
    
    # Potentiell energi V
    V_m = m * g * y_B
    V_M = M * g * y_C
    V = V_m + V_M
    
    # Lagrangian
    L_lagrange = T - V
    
    # Lagranges ekvationer
    # Notera att tau adderas här (vilket ger Q_phi = -tau). Detta balanserar avsaknaden 
    # av minustecken i PD-formeln, vilket gör att momentet korrekt verkar återställande.
    eq_theta = sp.diff(sp.diff(L_lagrange, theta_f_d), t) - sp.diff(L_lagrange, theta_f)
    eq_phi = sp.diff(sp.diff(L_lagrange, phi_f_d), t) - sp.diff(L_lagrange, phi_f) + tau
    
    # Förhindra "Singular matrix"
    th, ph, th_d, ph_d, th_dd, ph_dd = sp.symbols('th ph th_d ph_d th_dd ph_dd')
    
    subs_dict = {
        sp.diff(theta_f_d, t): th_dd,
        sp.diff(phi_f_d, t): ph_dd,
        theta_f_d: th_d,
        phi_f_d: ph_d,
        theta_f: th,
        phi_f: ph
    }
    
    eq_theta_sym = sp.expand(eq_theta.subs(subs_dict))
    eq_phi_sym = sp.expand(eq_phi.subs(subs_dict))
    
    # Bryt ut massmatris M och högerled F
    eqns = [eq_theta_sym, eq_phi_sym]
    M_matrix, F_vector = sp.linear_eq_to_matrix(eqns, [th_dd, ph_dd])
    
    # Skapa numeriska funktioner för snabb beräkning
    params = (M, k_rad, d, L, m, g, tau)
    states = (th, ph, th_d, ph_d)
    
    M_func = sp.lambdify(params + states, M_matrix, "numpy")
    F_func = sp.lambdify(params + states, F_vector, "numpy")
    
    T_sym = T.subs(subs_dict)
    V_sym = V.subs(subs_dict)
    T_func = sp.lambdify(params + states, T_sym, "numpy")
    V_func = sp.lambdify(params + states, V_sym, "numpy")
    
    return M_func, F_func, T_func, V_func

# ==========================================
# Huvudprogram och Simulering
# ==========================================
p_M = 68.0
p_k = 0.5
p_L = 2.0
p_m = 15.0
p_g = 9.81
phi_max = np.pi / 2.1
tau_max = 500.0  

M_func, F_func, T_func, V_func = setup_equations()

def state_derivative(t, y, active, curr_Kp, curr_Kd, curr_d):
    theta, phi, theta_d, phi_d = y
    
    tau_val = 0.0
    if active:
        # Exakt policy enligt uppgift Steg 2
        if theta_d < 0:
            tau_ideal = curr_Kp * p_M * p_g * curr_d * (phi - 0) + curr_Kd * phi_d
        else:
            tau_ideal = curr_Kp * p_M * p_g * curr_d * (phi - phi_max) + curr_Kd * phi_d
        
        # Asymmetrisk klippning (-1.5*tau_max till tau_max)
        tau_val = np.clip(tau_ideal, -1.5 * tau_max, tau_max)
        
    M_mat = np.array(M_func(p_M, p_k, curr_d, p_L, p_m, p_g, tau_val, theta, phi, theta_d, phi_d), dtype=float)
    F_vec = np.array(F_func(p_M, p_k, curr_d, p_L, p_m, p_g, tau_val, theta, phi, theta_d, phi_d), dtype=float)
    
    q_dd = np.linalg.solve(M_mat, F_vec).flatten()
    
    return [theta_d, phi_d, q_dd[0], q_dd[1]]

def run_simulation(step):
    std_Kp = 10.0 
    std_Kd = 300.0  
    std_d = 0.5

    if step in [1, 2]:
        if step == 1:
            print("\nKör Steg 1: Passivt system (3 sekunder)...")
            # KORRIGERAT: Start i vila med theta = pi/2 enligt instruktionerna
            y0 = [np.pi/2, 0.0, 0.0, 0.0]
            t_span = (0, 3)
            active = False
        else:
            print("\nKör Steg 2: Aktivt system (20 sekunder)...")
            # KORRIGERAT: Start exakt vertikalt (theta = 0). Gravitationen sätter igång rörelsen 
            # eftersom phi_max inte är exakt 90 grader.
            y0 = [0.0, phi_max, 0.0, 0.0]
            t_span = (0, 20)
            active = True

        t_eval_points = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])*50))
        sol = solve_ivp(
            fun=lambda t, y: state_derivative(t, y, active, std_Kp, std_Kd, std_d),
            t_span=t_span,
            y0=y0,
            method='Radau',
            t_eval=t_eval_points,
            max_step=0.01
        )
        
        t = sol.t
        theta, phi, theta_d, phi_d = sol.y
        
        T_vals = np.zeros_like(t)
        V_vals = np.zeros_like(t)
        for i in range(len(t)):
            T_vals[i] = float(T_func(p_M, p_k, std_d, p_L, p_m, p_g, 0, theta[i], phi[i], theta_d[i], phi_d[i]))
            V_vals[i] = float(V_func(p_M, p_k, std_d, p_L, p_m, p_g, 0, theta[i], phi[i], theta_d[i], phi_d[i]))
        E_vals = T_vals + V_vals
        
        # Statiska grafer
        plt.figure(f"Rapportgrafer Steg {step}", figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(t, theta, label=r'$\theta$ (Gunga)')
        plt.plot(t, phi, label=r'$\phi$ (Person)')
        plt.title(f'Vinklar (Steg {step})')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(t, theta_d, label=r'$\dot{\theta}$')
        plt.plot(t, phi_d, label=r'$\dot{\phi}$')
        plt.title('Vinkelhastigheter')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(t, T_vals, label='T (Kinetisk)')
        plt.plot(t, V_vals, label='V (Potentiell)')
        plt.plot(t, E_vals, label='E (Total)', linestyle='--')
        plt.title('Energier')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.title('Stroboskop-rörelse')
        plt.axhline(0, color='k', lw=1)
        plt.axvline(0, color='k', lw=1)
        
        # KORRIGERAT: Renderar endast punkterna för t=0, 0.2, 0.4... sekunder
        interval_indices = [np.argmin(np.abs(t - t_val)) for t_val in np.arange(0, t_span[1] + 0.1, 0.2)]
        for i in interval_indices:
            if i >= len(t): continue
            th = theta[i]
            ph = phi[i]
            x_B = p_L * np.sin(th)
            y_B = -p_L * np.cos(th)
            x_C = x_B + std_d * np.cos(th + ph)
            y_C = y_B + std_d * np.sin(th + ph)
            plt.plot([0, x_B], [0, y_B], color='#8B4513', alpha=0.3)
            plt.plot([x_B, x_C], [y_B, y_C], color='blue', lw=3, alpha=0.3)
            plt.plot(x_B, y_B, 'o', color='#F4A460', markersize=4, alpha=0.3)
            
        plt.axis('equal')
        plt.ylim(-3.5, 2.5)
        plt.tight_layout()
        print(f"--> Visar grafer för Steg {step}. Stäng fönstret för att starta animationen.")
        plt.show()

        # Animation
        fig2, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(f'Steg {step}: Livesimulering av gungan')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-3.5, 2.0)

        line_swing, = ax.plot([], [], color='#8B4513', lw=3, label='Gunga (stång)')
        line_person, = ax.plot([], [], color='blue', lw=6, label='Person')
        point_A, = ax.plot([0], [0], 'ks', markersize=6, label='Ledd A')
        point_B, = ax.plot([], [], marker='o', color='#F4A460', markersize=10, linestyle='None', label='Sits B')
        
        ax.legend(loc='upper right')

        def init():
            line_swing.set_data([], [])
            line_person.set_data([], [])
            point_B.set_data([], [])
            return line_swing, line_person, point_B

        def update(i):
            th = theta[i]
            ph = phi[i]
            x_B = p_L * np.sin(th)
            y_B = -p_L * np.cos(th)
            x_C = x_B + std_d * np.cos(th + ph)
            y_C = y_B + std_d * np.sin(th + ph)
            
            line_swing.set_data([0, x_B], [0, y_B])
            line_person.set_data([x_B, x_C], [y_B, y_C])
            point_B.set_data([x_B], [y_B])
            
            return line_swing, line_person, point_B

        dt_ms = (t[1] - t[0]) * 1000 if len(t) > 1 else 20
        print(f"--> Visar animation för Steg {step}. Stäng för att gå vidare.")
        ani = animation.FuncAnimation(fig2, update, frames=len(t), init_func=init, blit=True, interval=dt_ms)
        plt.show()

    elif step == 3:
        print("\nKör Steg 3: Analyserar variationer av Kp i 20 sekunder...")
        scenarios = [
            {"name": "Halverad Kp (Kp=5)", "Kp": 5.0, "Kd": 300.0, "d": 0.5, "color": "blue"},
            {"name": "Nominell (Kp=10)", "Kp": 10.0, "Kd": 300.0, "d": 0.5, "color": "orange"},
            {"name": "Dubbel Kp (Kp=20)", "Kp": 20.0, "Kd": 300.0, "d": 0.5, "color": "green"}
        ]
        
        plt.figure("Analys och Diskussion (Steg 3)", figsize=(14, 6))
        
        t_span = (0, 20)
        t_eval_points = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])*50))

        for sc in scenarios:
            sol = solve_ivp(
                fun=lambda t, y: state_derivative(t, y, True, sc["Kp"], sc["Kd"], sc["d"]),
                t_span=t_span,
                y0=[0.0, phi_max, 0.0, 0.0],
                method='Radau',
                t_eval=t_eval_points,
                max_step=0.01
            )
            
            plt.subplot(1, 2, 1)
            plt.plot(sol.t, sol.y[0], label=sc["name"], color=sc["color"], lw=1.5)
            
            plt.subplot(1, 2, 2)
            plt.plot(sol.t, sol.y[1], label=sc["name"], color=sc["color"], lw=1.5)

        plt.subplot(1, 2, 1)
        plt.title('Gungans vinkel ($\\theta$) över tid')
        plt.xlabel('Tid (s)')
        plt.ylabel('Vinkel (rad)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.title('Kroppens vinkel ($\\varphi$) över tid')
        plt.xlabel('Tid (s)')
        plt.ylabel('Vinkel (rad)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        print("--> Klar! Visar jämförelsegraf för Steg 3.")
        plt.show()

if __name__ == "__main__":
    run_simulation(step=1)
    run_simulation(step=2)
    run_simulation(step=3)