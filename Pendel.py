import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import os

# funktionen som byggeer upp våra matematiska
# ekvationer för de fysiska problemen
def setup_equations():
    print("Härleder ekvationer med SymPy")
    # Definierar tiden (t) som en symbolisk variabel i SymPy
    t = sp.Symbol('t')

    # Skapar symboliska konstanter: M = personens massa, k_rad = tröghetsradie,
    # d = avstånd från sits till masscentrum, L = gungans längd, 
    # m = gungans massa, g = tyngdacceleration. real=True säger att vi 
    # räknar med reella tal
    M, k_rad, d, L, m, g = sp.symbols('M k d L m g', real=True)
    # Skapar symboler för vridmomenten (kraften) som appliceras på personen (tau) 
    # respektive gungan (tau_theta).
    tau, tau_theta = sp.symbols('tau tau_theta', real=True)

   # Definierar vinklarna theta och phi som funktioner som beror av tiden t.
    theta_f = sp.Function('theta')(t)
    phi_f = sp.Function('phi')(t)
    # Beräknar förstaderivatan (vinkelhastigheten) för båda vinklarna 
    # med avseende på tiden
    theta_f_d = sp.diff(theta_f, t)
    phi_f_d = sp.diff(phi_f, t)

    # Ställer upp en ekvation för gungans sits (punkt B) position
    # i x- och y-led baserat på vinkeln theta och längden L.
    x_B = L * sp.sin(theta_f)
    y_B = -L * sp.cos(theta_f)
    # Ställer upp positionen för personens masscentrum (punkt C) 
    # baserat på sitsens position plus avståndet d och vinklarna 
    # theta + phi.
    x_C = x_B + d * sp.cos(theta_f + phi_f)
    y_C = y_B + d * sp.sin(theta_f + phi_f)

    # Beräknar den kvadrerade hastigheten v^2 = prick(x)^2 + prick(y)^2 
    # för sitsen B och masscentrum C.
    v_B_sq = sp.diff(x_B, t)**2 + sp.diff(y_B, t)**2
    v_C_sq = sp.diff(x_C, t)**2 + sp.diff(y_C, t)**2
    # Beräknar personens tröghetsmoment.
    I_M = M * k_rad**2
    # Beräknar gungans rörelseenergi.
    T_m = 0.5 * m * v_B_sq
    # Beräknar personens kinetiska energi 
    # (både translations- och rotationsenergi).
    T_M = 0.5 * M * v_C_sq + 0.5 * I_M * (theta_f_d + phi_f_d)**2
    # Summerar den totala kinetiska energin.
    T = T_m + T_M

    # Beräknar den potentiella energin för både gungan och personen
    V_m = m * g * y_B
    V_M = M * g * y_C
    V = V_m + V_M

    # Ställer upp Lagrange-funktionen som skillnaden 
    # mellan kinetisk och potentiell energi.
    L_lagrange = T - V

    # Härleder Lagrange-ekvationerna med avseende på theta och phi
    eq_theta = sp.diff(sp.diff(L_lagrange, theta_f_d), t) - sp.diff(L_lagrange, theta_f) - tau_theta
    eq_phi = sp.diff(sp.diff(L_lagrange, phi_f_d), t) - sp.diff(L_lagrange, phi_f) - tau

    # Definierar symboler för de okända variablerna: 
    # theta, phi, deras första och andra derivator.
    th, ph, th_d, ph_d, th_dd, ph_dd = sp.symbols('th ph th_d ph_d th_dd ph_dd')

    # Skapar en substitutionsordbok för att ersätta de symboliska funktionerna
    subs_dict = {
        sp.diff(theta_f_d, t): th_dd,
        sp.diff(phi_f_d, t): ph_dd,
        theta_f_d: th_d,
        phi_f_d: ph_d,
        theta_f: th,
        phi_f: ph
    }
    # Använder substitutionsordboken för att ersätta de 
    # symboliska funktionerna i ekvationerna.
    eq_theta_sym = sp.expand(eq_theta.subs(subs_dict))
    eq_phi_sym = sp.expand(eq_phi.subs(subs_dict))

    eqns = [eq_theta_sym, eq_phi_sym]
    M_matrix, F_vector = sp.linear_eq_to_matrix(eqns, [th_dd, ph_dd])

    params = (M, k_rad, d, L, m, g, tau, tau_theta)
    states = (th, ph, th_d, ph_d)

    M_func = sp.lambdify(params + states, M_matrix, "numpy")
    F_func = sp.lambdify(params + states, F_vector, "numpy")

    # Omvandlar de symboliska ekvationerna till en matrisform
    eqns = [eq_theta_sym, eq_phi_sym]
    
    M_matrix, F_vector = sp.linear_eq_to_matrix(eqns, [th_dd, ph_dd])

    
    # Definierar ordningen på variablerna för de snabba Numpy-funktionerna nedan
     # Konstanter och yttre krafter
    params = (M, k_rad, d, L, m, g, tau, tau_theta) 
    # Föränderliga variabler (vinklar och hastigheter)
    states = (th, ph, th_d, ph_d)                    
    # Skapar lambdify-funktioner för att snabbt kunna beräkna
    # M-matrisen och F-vektorn med Numpy
    M_func = sp.lambdify(params + states, M_matrix, "numpy")
    F_func = sp.lambdify(params + states, F_vector, "numpy")

    # Ersätter symboliska variabler i uttrycket för den kinetiska och potentiella
    # energin med värden eller andra uttryck från subs_dict.
    T_sym = T.subs(subs_dict)
    V_sym = V.subs(subs_dict)
    # Omvandlar uttrycket för kinetisk energi till en funktion som kan beräknas numeriskt med NumPy.
    # Funktionen tar emot parametrarna och tillståndsvariablerna i ordningen params + states.
    T_func = sp.lambdify(params + states, T_sym, "numpy")
    V_func = sp.lambdify(params + states, V_sym, "numpy")

    return M_func, F_func, T_func, V_func



# Anger massan M till 68 kg.
p_M = 68.0
# Anger värdet på parametern k som används i systemets ekvationer.
p_k = 0.5
# Anger gungans längd L till 2 meter.
p_L = 2.0
# Anger massan m till 15 kg.
p_m = 15.0
# Anger tyngdaccelerationen till 9,81 m/s².
p_g = 9.81
# Anger den maximalt tillåtna kroppsvinkeln phi i radianer.
# np.pi motsvarar pi och division med 2.1 ger ungefär 85,7 grader.
phi_max = np.pi / 2.1
# Anger det nominella maximala vridmomentet till 500 Nm.
tau_max = 500.0
# Anger simuleringstiden till 15 sekunder för steg 2 och steg 3.
sim_tid_steg2 = 15
# Anropar funktionen som skapar systemets rörelseekvationer.
# De fyra funktionerna sparas i variabler så att de kan användas senare.
M_func, F_func, T_func, V_func = setup_equations()


# Definierar funktionen som beräknar systemets tillståndsderivator.
# t är tiden, y innehåller tillståndet, active väljer aktivt/passivt system.
# curr_Kp och curr_Kd är regulatorns förstärkningar och curr_d är avståndsparametern.
def state_derivative(t, y, active, curr_Kp, curr_Kd, curr_d):

    # Delar upp tillståndsvektorn y i fyra variabler.
    # theta = gungans vinkel, phi = kroppens vinkel.
    # theta_d = gungans vinkelhastighet, phi_d = kroppens vinkelhastighet.
    theta, phi, theta_d, phi_d = y

    # Sätter det styrande vridmomentet på kroppen till noll från början.
    tau_val = 0.0

    # Sätter vridmomentet som påverkar gungans rörelse direkt till noll.
    tau_theta_val = 0.0

    # Kontrollerar om det aktiva styrsystemet ska användas.
    # Om active är False används inget av styrmomenten nedan.
    if active:

        # Beräknar ett mjukt styrvärde mellan 0 och 1 med tanh.
        # När gungan rör sig i positiv riktning blir värdet nära 1.
        # När gungan rör sig i negativ riktning blir värdet nära 0.
        # Talet 0.15 bestämmer hur snabbt växlingen sker.
        pump_switch = 0.5 * (1.0 + np.tanh(theta_d / 0.15))

        # Samlar rörelseekvationerna och skriver om dem på matrisformen M*q_dd = F
        # Beräknar önskad kroppsvinkel utifrån rörelsens riktning.
        # Referensvinkeln ligger mellan 0 och phi_max.
        phi_ref = phi_max * pump_switch

        # Beräknar önskat vridmoment med en PD-regulator.
        # Kp-delen korrigerar skillnaden mellan aktuell och önskad kroppsvinkel.
        # Kd-delen bromsar kroppens vinkelhastighet för att dämpa rörelsen.
        tau_ideal = -curr_Kp * p_M * p_g * curr_d * (phi - phi_ref) - curr_Kd * phi_d

        # Begränsar regulatorns vridmoment till ett bestämt intervall.
        # Lägsta värdet är -500 Nm och högsta värdet är 750 Nm.
        # Observera att stoppmomenten nedan kan öka momentet ytterligare.
        tau_val = np.clip(tau_ideal, -tau_max, 1.5 * tau_max)

        # Anger styvheten hos de virtuella stopp som begränsar kroppsvinkeln.
        k_stop_phi = 3000.0

        # Anger dämpningen hos de virtuella stoppen.
        d_stop_phi = 400.0

        # Kontrollerar om kroppsvinkeln har blivit mindre än noll.
        if phi < 0.0:

            # Lägger till ett återförande moment som motverkar rörelsen
            # utanför den undre vinkelgränsen och dämpar vinkelhastigheten.
            tau_val += -k_stop_phi * phi - d_stop_phi * phi_d

        # Kontrollerar om kroppsvinkeln har överskridit maxvinkeln.
        elif phi > phi_max:

            # Lägger till ett återförande moment som försöker föra tillbaka
            # kroppen inom tillåtet vinkelområde och samtidigt dämpa rörelsen.
            tau_val += -k_stop_phi * (phi - phi_max) - d_stop_phi * phi_d

        # Anger koefficienten för linjär dämpning av gungans rörelse.
        c_lin = 6.0

        # Anger koefficienten för kvadratisk dämpning av gungans rörelse.
        c_quad = 18.0

        # Beräknar ett motverkande moment på gungan.
        # Den linjära delen ökar med hastigheten och den kvadratiska
        # delen blir särskilt betydelsefull vid högre hastigheter.
        # abs() används så att dämpningen alltid motverkar rörelseriktningen.
        tau_theta_val = -c_lin * theta_d - c_quad * theta_d * abs(theta_d)

    # Beräknar systemets massmatris genom att anropa M_func.
    # Matrisen används i rörelseekvationen M*q_dd = F.
    M_mat = np.array(

        # Skickar in systemets parametrar, moment, vinklar och vinkelhastigheter.
        M_func(p_M, p_k, curr_d, p_L, p_m, p_g, tau_val, tau_theta_val, theta, phi, theta_d, phi_d),

        # Ser till att resultatet representeras som flyttal.
        dtype=float
    )

    # Beräknar högerledet F i systemets rörelseekvationer.
    F_vec = np.array(

        # Anropar den tidigare skapade funktionen för högerledet
        # med samma parametrar och aktuella tillstånd.
        F_func(p_M, p_k, curr_d, p_L, p_m, p_g, tau_val, tau_theta_val, theta, phi, theta_d, phi_d),

        # Omvandlar resultatets datatyp till flyttal.
        dtype=float
    )

    # Löser ekvationssystemet M*q_dd = F för att få vinkelaccelerationerna.
    # flatten() gör om resultatet till en endimensionell array.
    q_dd = np.linalg.solve(M_mat, F_vec).flatten()

    # Returnerar de fyra tillståndsderivatorna i rätt ordning:
    # gungans vinkelhastighet, kroppens vinkelhastighet,
    # gungans vinkelacceleration och kroppens vinkelacceleration.
    return [theta_d, phi_d, q_dd[0], q_dd[1]]

def plot_parameter_variation(scenarios, param_title, filename):
    """
    Hjälpfunktion för att köra och rita 2 grafer per bild (theta och phi över 20 sekunder)
    samt visa och spara slutvärdena.
    """
    plt.figure(f"Analys - {param_title}", figsize=(14, 6))

    t_span = (0, 20)
    t_eval_points = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) * 50))

    for sc in scenarios:
        sol = solve_ivp(
            fun=lambda t, y: state_derivative(
                t, y, True, 
                curr_Kp=sc.get("Kp", 10.0), 
                curr_Kd=sc.get("Kd", 300.0), 
                curr_d=sc.get("d", 0.5),
                curr_tau_max=sc.get("tau_max", tau_max_nom),
                curr_M=sc.get("M", p_M_nom)
            ),
            t_span=t_span,
            y0=[0.0, phi_max, 0.0, 0.0],
            method='RK45',
            t_eval=t_eval_points,
            rtol=1e-6,
            atol=1e-8,
            max_step=0.02
        )

        t_end = sol.t[-1]
        theta_end = sol.y[0][-1]
        phi_end = sol.y[1][-1]

        # Graf 1: Gungans vinkel (theta)
        plt.subplot(1, 2, 1)
        line, = plt.plot(sol.t, sol.y[0], label=sc["name"], color=sc["color"], lw=1.5)
        plt.plot(t_end, theta_end, 'o', color=sc["color"])
        plt.annotate(
            f"{theta_end:.2f} rad", 
            xy=(t_end, theta_end), 
            xytext=(-45, 5), 
            textcoords='offset points',
            color=sc["color"],
            fontsize=9,
            fontweight='bold'
        )

        # Graf 2: Kroppens vinkel (phi)
        plt.subplot(1, 2, 2)
        plt.plot(sol.t, sol.y[1], label=sc["name"], color=sc["color"], lw=1.5)
        plt.plot(t_end, phi_end, 'o', color=sc["color"])
        plt.annotate(
            f"{phi_end:.2f} rad", 
            xy=(t_end, phi_end), 
            xytext=(-45, 5), 
            textcoords='offset points',
            color=sc["color"],
            fontsize=9,
            fontweight='bold'
        )

    plt.subplot(1, 2, 1)
    plt.title(f'Gungans vinkel ($\\theta$) över tid – Variation av {param_title}')
    plt.xlabel('Tid (s)')
    plt.ylabel('Vinkel (rad)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title(f'Kroppens vinkel ($\\varphi$) över tid – Variation av {param_title}')
    plt.xlabel('Tid (s)')
    plt.ylabel('Vinkel (rad)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"--> Graf sparad som: {os.path.abspath(filename)}")
    plt.show()

# Huvudfunktion som kör simuleringen för olika steg.
def run_simulation(step):
    # Anger standardvärden för regulatorns förstärkningar och 
    # avståndsparametern.
    std_Kp = 10.0
    std_Kd = 300.0
    std_d = 0.5
    # Anger nominella värden för tau_max och M för senare användning.
    # tau_max_nom = 500.0
    if step in [1, 2]:
        # Anger initialt tillstånd och simuleringstid beroende på 
        # vilket steg som körs.
        if step == 1:
            print("\nKör Steg 1: Passivt system (3 sekunder)...")
            y0 = [np.pi / 2, 0.0, 0.0, 0.0]
            t_span = (0, 3)
            active = False
            solver_kwargs = {'rtol': 1e-8, 'atol': 1e-10, 'max_step': 0.01}
        # Anger initialt tillstånd och simuleringstid beroende på 
        # vilket steg som körs.
        else:
            print(f"\nKör Steg 2: Aktivt system ({sim_tid_steg2} sekunder)...")
            y0 = [0.0, phi_max, 0.0, 0.0]
            t_span = (0, sim_tid_steg2)
            active = True
            solver_kwargs = {'rtol': 1e-6, 'atol': 1e-8, 'max_step': 0.02}
        # Anger antalet utvärderingspunkter för simuleringen.
        t_eval_points = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) * 50))
        sol = solve_ivp(
            fun=lambda t, y: state_derivative(t, y, active, std_Kp, std_Kd, std_d),
            t_span=t_span,
            y0=y0,
            method='RK45',
            t_eval=t_eval_points,
            **solver_kwargs
        )
        # Anger att simuleringen är klar och visar hur lång tid som beräknats.
        print(f"Simulering klar: {sol.t[-1]:.1f} av {t_span[1]} s beräknade ({len(sol.t)} bildrutor).")

        t = sol.t
        theta, phi, theta_d, phi_d = sol.y
        # Beräknar kinetisk och potentiell energi för varje tidpunkt i simuleringen.
        T_vals = np.zeros_like(t)
        V_vals = np.zeros_like(t)
        # Loopar igenom varje tidpunkt och beräknar energierna med hjälp av
        # de tidigare definierade funktionerna.
        for i in range(len(t)):
            T_vals[i] = float(T_func(p_M, p_k, std_d, p_L, p_m, p_g, 0, 0, theta[i], phi[i], theta_d[i], phi_d[i]))
            V_vals[i] = float(V_func(p_M, p_k, std_d, p_L, p_m, p_g, 0, 0, theta[i], phi[i], theta_d[i], phi_d[i]))
        E_vals = T_vals + V_vals

        fig1 = plt.figure(f"Rapportgrafer Steg {step}", figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(t, theta, label=r'$\theta$ (Gunga)')
        plt.plot(t, phi, label=r'$\phi$ (Person)')
        plt.title(f'Vinklar (Steg {step})')
        plt.xlabel('Tid [s]')
        plt.ylabel('Vinkel [rad]')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(t, theta_d, label=r'$\dot{\theta}$')
        plt.plot(t, phi_d, label=r'$\dot{\phi}$')
        plt.title('Vinkelhastigheter')
        plt.xlabel('Tid [s]')
        plt.ylabel('Vinkelhastighet [rad/s]')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(t, T_vals, label='T (Kinetisk)')
        plt.plot(t, V_vals, label='V (Potentiell)')
        plt.plot(t, E_vals, label='E (Total)', linestyle='--')
        plt.title('Energier')
        plt.xlabel('Tid [s]')
        plt.ylabel('Energi [J]')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.title('Rörelse över tid')
        plt.axhline(0, color='k', lw=1)
        plt.axvline(0, color='k', lw=1)
        
        strobe_step = 0.2 if t_span[1] <= 20 else 0.5
        interval_indices = [np.argmin(np.abs(t - t_val)) for t_val in np.arange(0, t_span[1] + 0.1, strobe_step)]
        
        for i in interval_indices:
            if i >= len(t):
                continue
            th = theta[i]
            ph = phi[i]
            x_B = p_L * np.sin(th)
            y_B = -p_L * np.cos(th)
            x_C = x_B + std_d * np.cos(th + ph)
            y_C = y_B + std_d * np.sin(th + ph)
            plt.plot([0, x_B], [0, y_B], color='#8B4513', alpha=0.25)
            plt.plot([x_B, x_C], [y_B, y_C], color='blue', lw=3, alpha=0.25)
            plt.plot(x_B, y_B, 'o', color='#F4A460', markersize=4, alpha=0.25)

        plt.axis('equal')
        plt.ylim(-3.5, 2.5)
        plt.tight_layout()
        
        filename = f"rapport_steg_{step}.png"
        fig1.savefig(filename, dpi=300)
        print(f"--> Graf sparad som: {os.path.abspath(filename)}")
        print(f"--> Visar grafer för Steg {step}. Stäng fönstret för att starta animationen.")
        plt.show()

        # Animation
        fig2, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(f'Steg {step}: Livesimulering av gungan ({t_span[1]} s)')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-3.5, 2.0)

        line_swing, = ax.plot([], [], color='#8B4513', lw=3, label='Gunga (stång)')
        line_person, = ax.plot([], [], color='blue', lw=6, label='Person')
        point_A, = ax.plot([0], [0], 'ks', markersize=6, label='Led A')
        point_B, = ax.plot([], [], marker='o', color='#F4A460', markersize=10, linestyle='None', label='Sits B')
        time_text = ax.text(
            0.03, 0.95, '', transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax.legend(loc='upper right')

        def init():
            line_swing.set_data([], [])
            line_person.set_data([], [])
            point_B.set_data([], [])
            time_text.set_text('')
            return line_swing, line_person, point_B, time_text

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
            time_text.set_text(f"Tid: {t[i]:.1f} / {t_span[1]} s\nGungvinkel: {np.rad2deg(th):.1f}°")

            return line_swing, line_person, point_B, time_text

        dt_ms = (t[1] - t[0]) * 1000 if len(t) > 1 else 20
        print(f"--> Visar animation för Steg {step}. Stäng för att gå vidare.")
        ani = animation.FuncAnimation(fig2, update, frames=len(t), init_func=init, blit=True, interval=dt_ms)
        plt.show()

    elif step == 3:
        print(f"\nKör Steg 3: Analyserar variationer av Kp i {sim_tid_steg2} sekunder...")
        scenarios = [
            {"name": "Halverad Kp (Kp=5)", "Kp": 5.0, "Kd": 300.0, "d": 0.5, "color": "blue"},
            {"name": "Nominell (Kp=10)", "Kp": 10.0, "Kd": 300.0, "d": 0.5, "color": "orange"},
            {"name": "Dubbel Kp (Kp=20)", "Kp": 20.0, "Kd": 300.0, "d": 0.5, "color": "green"}
        ]

        plt.figure("Analys och Diskussion (Steg 3)", figsize=(14, 6))

        t_span = (0, sim_tid_steg2)
        t_eval_points = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) * 50))

        for sc in scenarios:
            sol = solve_ivp(
                fun=lambda t, y: state_derivative(t, y, True, sc["Kp"], sc["Kd"], sc["d"]),
                t_span=t_span,
                y0=[0.0, phi_max, 0.0, 0.0],
                method='RK45',
                t_eval=t_eval_points,
                rtol=1e-6,
                atol=1e-8,
                max_step=0.02
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
    # run_simulation(step=3)  # Kd
    # run_simulation(step=4)  # Kp
    # run_simulation(step=5)  # tau-max
    # run_simulation(step=6)  # M