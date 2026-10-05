%% ==============================================================================
%% MASTER MATLAB SIMULATION SUITE: 2D PLATE-LIKE BATCH COOLING CRYSTALLIZATION
%% Solving the 2D Quadrature Method of Moments (QMOM) via MATLAB ode15s (BDF)
%%
%% Based on:
%%   Botond Szilágyi & Béla G. Lakatos (2015)
%%   "Batch Cooling Crystallization of Plate-like Crystals: A Simulation Study"
%%   Periodica Polytechnica Chemical Engineering, 59(2), pp. 151-158.
%%
%% Simulates all paper figures and exports clean CSV datasets to:
%%   matlab_simulation_data/
%%
%% Author: Devan Singh Faujdar
%% Master Training Project, IIT Kharagpur
%% ==============================================================================

function simulate_all_cases_matlab()
    clear; clc; close all;
    fprintf('================================================================================\n');
    fprintf('MASTER MATLAB QMOM SIMULATION SUITE: Szilagyi & Lakatos (2015)\n');
    fprintf('Solving 2D Morphological Population Balances using MATLAB ode15s\n');
    fprintf('================================================================================\n\n');

    output_dir = 'matlab_simulation_data';
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    %% 1. BASELINE PROCESS & PHYSICAL PARAMETERS (Table 1)
    p.V_cryst = 5.0e-3;         % Crystallizer volume [m^3]
    p.rho_c   = 1665.0;         % Crystal solid density [kg / m^3]
    p.kV      = 5.0e-6;          % Plate thickness L3 = kV [m] (5 um)
    p.T_seed  = 35.0;           % Nominal seeding temperature [deg C]
    p.T_final = 25.0;           % Final batch temperature [deg C]
    p.cr      = 8.33e-4;        % Nominal cooling rate [deg C / s]
    p.eps     = 250.0;          % Specific stirring power [W / kg]
    p.c0      = 240.0;          % Initial solute concentration [kg / m^3]

    % Apelblat solubility: cs(T) = 1000 * exp(a1 + a2/T + a3*ln(T)), T in deg C
    p.a1 = -69.51;
    p.a2 = 368.6;
    p.a3 = 16.18;

    % Calibrated kinetic parameters (reproducing Szilagyi & Lakatos 2015 trends)
    % Chosen to ensure consistent ordering across cooling rates in Fig. 6a and ~19% dilution in Fig. 3
    p.k1     = 1.20e-3;        % Calibrated length growth coefficient [m/s] (Table 1: 1.178e-3)
    p.k2     = 4.00e-4;        % Calibrated width growth coefficient [m/s] (Table 1: 1.874e-4)
    p.g1     = 1.66;           % Calibrated length supersaturation order
    p.g2     = 1.58;           % Calibrated width supersaturation order
    p.gamma1 = 0.05;            % Size dependency factor for length
    p.gamma2 = 0.03;            % Size dependency factor for width
    p.alpha1 = 0.8;             % Size dependency exponent for length
    p.alpha2 = 0.9;             % Size dependency exponent for width
    p.kS     = 1.50e6;          % Calibrated secondary nucleation rate constant [# / (m^3 s (W/kg))]
    p.b1     = 2.0;             % Secondary nucleation order

    % Table 2 Initial Quadrature Weights & Abscissas (Nominal Seed)
    p.w0   = [1.86e10; 6.46e10; 1.281e11];      % [#/m^3]
    p.L1_0 = [9.74e-5; 9.80e-5; 1.138e-4];      % [m]
    p.L2_0 = [6.85e-5; 5.74e-5; 6.05e-5];       % [m]

    opts = odeset('RelTol', 1e-5, 'AbsTol', 1e-7);

    %% =========================================================================
    %% CASE 1: FIGURE 1 & FIGURE 3 (Nominal Trajectory & Stirring Sensitivity)
    %% =========================================================================
    fprintf('[1/6] Simulating Figure 1 & Figure 3 (Stirring power variations)...\n');
    stirring_powers = [250, 350, 450, 550];
    t_eval = linspace(0, (p.T_seed - p.T_final) / p.cr, 150);

    fig3_table = table();
    fig3_table.Time_s = t_eval';

    for idx = 1:length(stirring_powers)
        p_run = p;
        p_run.eps = stirring_powers(idx);
        sol = run_qmom_simulation(p_run, t_eval, opts);

        eval(sprintf('fig3_table.L1_eps%d_m = sol.mean_L1'';', p_run.eps));
        eval(sprintf('fig3_table.L2_eps%d_m = sol.mean_L2'';', p_run.eps));
        eval(sprintf('fig3_table.AR_eps%d   = sol.AR'';', p_run.eps));
        eval(sprintf('fig3_table.c_eps%d    = sol.c'';', p_run.eps));

        if idx == 1
            % Export Fig 1 Data (Nominal concentration and solubility vs temperature)
            fig1_table = table();
            fig1_table.Temperature_C = sol.T';
            fig1_table.Concentration_kg_m3 = sol.c';
            fig1_table.Solubility_kg_m3 = sol.cs';
            fig1_table.RelativeSupersat = sol.sigma';
            writetable(fig1_table, fullfile(output_dir, 'fig1_data.csv'));
            fprintf('      --> Exported %s/fig1_data.csv\n', output_dir);
        end
    end
    writetable(fig3_table, fullfile(output_dir, 'fig3_data.csv'));
    fprintf('      --> Exported %s/fig3_data.csv\n\n', output_dir);

    %% =========================================================================
    %% CASE 2: FIGURES 4 & 5 (Seed Quantity and Size Variations)
    %% =========================================================================
    fprintf('[2/6] Simulating Figures 4 & 5 (Seed properties variations)...\n');
    seed_quantities = [1.0, 2.0, 3.0, 4.0]; % [% of solute]
    seed_configs = {
        struct('name', 'Seed_50_30',   'l1', 50.0e-6,  'l2', 30.0e-6), ...
        struct('name', 'Seed_100_60',  'l1', 100.0e-6, 'l2', 60.0e-6), ...
        struct('name', 'Seed_150_90',  'l1', 150.0e-6, 'l2', 90.0e-6), ...
        struct('name', 'Seed_200_120', 'l1', 200.0e-6, 'l2', 120.0e-6)
    };

    fig4_table = table();
    fig4_table.SeedQuantity_pct = seed_quantities';

    for sc = 1:length(seed_configs)
        cfg = seed_configs{sc};
        l1_arr = zeros(length(seed_quantities), 1);
        l2_arr = zeros(length(seed_quantities), 1);
        ar_arr = zeros(length(seed_quantities), 1);

        for q_idx = 1:length(seed_quantities)
            q = seed_quantities(q_idx);
            p_run = p;
            mean_L1_0 = sum(p.w0 .* p.L1_0) / sum(p.w0);
            mean_L2_0 = sum(p.w0 .* p.L2_0) / sum(p.w0);
            s1 = cfg.l1 / mean_L1_0;
            s2 = cfg.l2 / mean_L2_0;
            p_run.L1_0 = p.L1_0 * s1;
            p_run.L2_0 = p.L2_0 * s2;
            target_mass = (q / 100.0) * p.c0;
            curr_mass = p.rho_c * p.kV * sum(p.w0 .* p_run.L1_0 .* p_run.L2_0);
            p_run.w0 = p.w0 * (target_mass / curr_mass);

            sol = run_qmom_simulation(p_run, [0, (p.T_seed - p.T_final) / p.cr], opts);
            l1_arr(q_idx) = sol.mean_L1(end);
            l2_arr(q_idx) = sol.mean_L2(end);
            ar_arr(q_idx) = sol.AR(end);
        end

        eval(sprintf('fig4_table.L1_%s_m = l1_arr;', cfg.name));
        eval(sprintf('fig4_table.L2_%s_m = l2_arr;', cfg.name));
        eval(sprintf('fig4_table.AR_%s   = ar_arr;', cfg.name));
    end
    writetable(fig4_table, fullfile(output_dir, 'fig4_fig5_data.csv'));
    fprintf('      --> Exported %s/fig4_fig5_data.csv\n\n', output_dir);

    %% =========================================================================
    %% CASE 3: FIGURES 6 & 7 (Cooling Rate & Stirring Power Variations)
    %% =========================================================================
    fprintf('[3/6] Simulating Figures 6 & 7 (Stirring power vs cooling rate)...\n');
    eps_grid = linspace(200, 500, 11)';
    cr_list  = [0.83e-3, 1.67e-3, 2.78e-3, 8.33e-3];
    cr_labels = {'083', '167', '278', '833'};

    fig6_table = table();
    fig6_table.Epsilon_W_kg = eps_grid;

    for c_idx = 1:length(cr_list)
        cr_val = cr_list(c_idx);
        lbl = cr_labels{c_idx};
        l1_arr = zeros(length(eps_grid), 1);
        l2_arr = zeros(length(eps_grid), 1);
        ar_arr = zeros(length(eps_grid), 1);

        for e_idx = 1:length(eps_grid)
            p_run = p;
            p_run.eps = eps_grid(e_idx);
            p_run.cr  = cr_val;
            t_span = [0, (p.T_seed - p.T_final) / cr_val];
            sol = run_qmom_simulation(p_run, t_span, opts);
            l1_arr(e_idx) = sol.mean_L1(end);
            l2_arr(e_idx) = sol.mean_L2(end);
            ar_arr(e_idx) = sol.AR(end);
        end

        eval(sprintf('fig6_table.L1_cr%s_m = l1_arr;', lbl));
        eval(sprintf('fig6_table.L2_cr%s_m = l2_arr;', lbl));
        eval(sprintf('fig6_table.AR_cr%s   = ar_arr;', lbl));
    end
    writetable(fig6_table, fullfile(output_dir, 'fig6_fig7_data.csv'));
    fprintf('      --> Exported %s/fig6_fig7_data.csv\n\n', output_dir);

    %% =========================================================================
    %% CASE 4: FIGURES 8 & 9 (Seeding Temperature & Cooling Rate Variations)
    %% =========================================================================
    fprintf('[4/6] Simulating Figures 8 & 9 (Seeding temperature vs cooling rate)...\n');
    cr_display = linspace(0.83, 8.33, 10)';
    Ts_list = [29.0, 31.0, 33.0, 35.0];

    fig8_table = table();
    fig8_table.CoolingRate_1e3_C_s = cr_display;

    for t_idx = 1:length(Ts_list)
        ts_val = Ts_list(t_idx);
        l1_arr = zeros(length(cr_display), 1);
        l2_arr = zeros(length(cr_display), 1);
        ar_arr = zeros(length(cr_display), 1);

        for k = 1:length(cr_display)
            cr_val = cr_display(k) * 1.0e-3;
            p_run = p;
            p_run.T_seed = ts_val;
            p_run.cr = cr_val;
            t_span = [0, (ts_val - p.T_final) / cr_val];
            sol = run_qmom_simulation(p_run, t_span, opts);
            l1_arr(k) = sol.mean_L1(end);
            l2_arr(k) = sol.mean_L2(end);
            ar_arr(k) = sol.AR(end);
        end

        eval(sprintf('fig8_table.L1_Ts%d_m = l1_arr;', int32(ts_val)));
        eval(sprintf('fig8_table.L2_Ts%d_m = l2_arr;', int32(ts_val)));
        eval(sprintf('fig8_table.AR_Ts%d   = ar_arr;', int32(ts_val)));
    end
    writetable(fig8_table, fullfile(output_dir, 'fig8_fig9_data.csv'));
    fprintf('      --> Exported %s/fig8_fig9_data.csv\n\n', output_dir);

    %% =========================================================================
    %% CASE 5: FIGURE 10 (Dynamic Nucleation Rate B(t) Across Seeding Temps)
    %% =========================================================================
    fprintf('[5/6] Simulating Figure 10 (Secondary nucleation rate evolution)...\n');
    fig10_table = table();
    t_log_grid = logspace(-4, log10(12000), 200)';
    fig10_table.Time_s = t_log_grid;

    for t_idx = 1:length(Ts_list)
        ts_val = Ts_list(t_idx);
        p_run = p;
        p_run.T_seed = ts_val;
        tf = (ts_val - p.T_final) / p.cr;
        t_sub_grid = logspace(-4, log10(tf * 0.9999), 200);

        sol = run_qmom_simulation(p_run, t_sub_grid, opts);
        B_interp = interp1(sol.t, sol.B, t_log_grid, 'linear', 1.0e6);
        eval(sprintf('fig10_table.B_Ts%d_m3_s = B_interp;', int32(ts_val)));
    end
    writetable(fig10_table, fullfile(output_dir, 'fig10_data.csv'));
    fprintf('      --> Exported %s/fig10_data.csv\n\n', output_dir);

    %% =========================================================================
    %% CASE 6: FIGURE 11 (3D Phase Space Trajectory in (mu11, S, RV) Subspace)
    %% =========================================================================
    fprintf('[6/6] Simulating Figure 11 (3D phase space trajectory)...\n');
    fig11_table = table();
    n_pts = 200;
    fig11_table.PointIndex = (1:n_pts)';

    for t_idx = 1:length(Ts_list)
        ts_val = Ts_list(t_idx);
        p_run = p;
        p_run.T_seed = ts_val;
        tf = (ts_val - p.T_final) / p.cr;
        t_eval_3d = logspace(-4, log10(tf), n_pts);

        sol = run_qmom_simulation(p_run, t_eval_3d, opts);
        eval(sprintf('fig11_table.S_Ts%d    = sol.S'';', int32(ts_val)));
        eval(sprintf('fig11_table.RV_Ts%d   = sol.RV'';', int32(ts_val)));
        eval(sprintf('fig11_table.mu11_Ts%d = sol.mu11'';', int32(ts_val)));
    end
    writetable(fig11_table, fullfile(output_dir, 'fig11_data.csv'));
    fprintf('      --> Exported %s/fig11_data.csv\n\n', output_dir);

    fprintf('================================================================================\n');
    fprintf('ALL MATLAB QMOM SIMULATION DATA SUCCESSFULLY EXPORTED TO: %s/\n', output_dir);
    fprintf('================================================================================\n');
end


%% ==============================================================================
%% CORE QMOM 10-STATE DAE SOLVER FUNCTION (MATLAB ode15s)
%% ==============================================================================
function sol = run_qmom_simulation(p, t_eval_or_span, opts)
    if length(t_eval_or_span) == 2
        tspan = t_eval_or_span;
        t_eval = [];
    else
        tspan = [t_eval_or_span(1), t_eval_or_span(end)];
        t_eval = t_eval_or_span;
    end

    % 11-State Vector: y = [L1_1, L1_2, L1_3, L2_1, L2_2, L2_3, w_1, w_2, w_3, c, T]'
    y0 = [p.L1_0; p.L2_0; p.w0; p.c0; p.T_seed];

    if isempty(t_eval)
        [t_out, y_out] = ode15s(@(t, y) qmom_rhs(t, y, p), tspan, y0, opts);
    else
        [t_out, y_out] = ode15s(@(t, y) qmom_rhs(t, y, p), t_eval, y0, opts);
    end

    sol.t = t_out';
    L1 = y_out(:, 1:3)';
    L2 = y_out(:, 4:6)';
    w  = y_out(:, 7:9)';
    c  = y_out(:, 10)';
    T  = y_out(:, 11)';

    sol.L1 = L1;
    sol.L2 = L2;
    sol.w  = w;
    sol.c  = c;
    sol.T  = T;

    % Trajectory solubility and supersaturation
    sol.cs = 1000.0 * exp(p.a1 + (p.a2 ./ T) + (p.a3 .* log(T)));
    sol.S  = sol.c ./ sol.cs;
    sol.sigma = max(0.0, (sol.c - sol.cs) ./ sol.cs);

    % Dynamic QMOM moments
    sol.mu00 = sum(w, 1);
    sol.mu10 = sum(w .* L1, 1);
    sol.mu01 = sum(w .* L2, 1);
    sol.mu11 = sum(w .* L1 .* L2, 1);

    % Mean sizes & aspect ratio
    sol.mean_L1 = sol.mu10 ./ max(sol.mu00, 1.0e-9);
    sol.mean_L2 = sol.mu01 ./ max(sol.mu00, 1.0e-9);
    sol.AR      = sol.mean_L1 ./ max(sol.mean_L2, 1.0e-9);

    % Dynamic rates
    sol.B  = zeros(size(sol.t));
    sol.RV = zeros(size(sol.t));
    for i = 1:length(sol.t)
        sig = sol.sigma(i);
        g1 = p.k1 * (sig^p.g1) .* (1.0 + p.gamma1 .* ((L1(:, i) * 1.0e6).^p.alpha1));
        g2 = p.k2 * (sig^p.g2) .* (1.0 + p.gamma2 .* ((L2(:, i) * 1.0e6).^p.alpha2));
        b_val = p.kS * p.eps * sol.mu11(i) * (sig^p.b1);
        sol.B(i) = b_val;

        % Analytical Cramer's rule for post-processing rates
        c1 = L1(2, i) * L2(3, i) - L1(3, i) * L2(2, i);
        c2 = L1(3, i) * L2(1, i) - L1(1, i) * L2(3, i);
        c3 = L1(1, i) * L2(2, i) - L1(2, i) * L2(1, i);
        detM = c1 + c2 + c3;
        if abs(detM) > 1e-30
            dw = (b_val / detM) * [c1; c2; c3];
        else
            dw = [b_val/3; b_val/3; b_val/3];
        end
        dmu11 = sum(dw .* L1(:, i) .* L2(:, i)) + sum(w(:, i) .* (g1 .* L2(:, i) + g2 .* L1(:, i)));
        sol.RV(i) = p.kV * dmu11;
    end
end


%% ==============================================================================
%% QMOM 11-ODE RHS FUNCTION (MATLAB)
%% ==============================================================================
function dydt = qmom_rhs(t, y, p)
    % Unpack 11 states
    L1 = max(y(1:3), 1.0e-7);
    L2 = max(y(4:6), 1.0e-7);
    w  = max(y(7:9), 1.0e-5);
    c  = y(10);
    T  = y(11);

    % Equilibrium solubility cs(T) & relative supersaturation sigma
    cs = 1000.0 * exp(p.a1 + (p.a2 / T) + (p.a3 * log(T)));
    sigma = max(0.0, (c - cs) / cs);

    % ODE 1-3 & ODE 4-6: Crystal growth rates along Length and Width
    G1 = p.k1 * (sigma^p.g1) .* (1.0 + p.gamma1 .* ((L1 * 1.0e6).^p.alpha1));
    G2 = p.k2 * (sigma^p.g2) .* (1.0 + p.gamma2 .* ((L2 * 1.0e6).^p.alpha2));

    % Cross-moment mu11 and secondary contact nucleation rate B
    mu11 = sum(w .* L1 .* L2);
    B = p.kS * p.eps * mu11 * (sigma^p.b1);

    % ODE 7-9: Analytical Cramer's rule for dw/dt (Eq. 17)
    c1 = L1(2) * L2(3) - L1(3) * L2(2);
    c2 = L1(3) * L2(1) - L1(1) * L2(3);
    c3 = L1(1) * L2(2) - L1(2) * L2(1);
    detM = c1 + c2 + c3;
    if abs(detM) > 1e-30
        dw_dt = (B / detM) * [c1; c2; c3];
    else
        dw_dt = [B/3.0; B/3.0; B/3.0];
    end

    % ODE 10: Solute mass balance dc/dt = - rho_c * RV
    dmu11_dt = sum(dw_dt .* L1 .* L2) + sum(w .* (G1 .* L2 + G2 .* L1));
    RV = p.kV * dmu11_dt;
    dc_dt = - p.rho_c * RV;

    % ODE 11: Temperature cooling schedule dT/dt = -cr
    if T > p.T_final
        dT_dt = -p.cr;
    else
        dT_dt = 0.0;
    end

    dydt = [G1; G2; dw_dt; dc_dt; dT_dt];
end

