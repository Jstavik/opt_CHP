        model = pulp.LpProblem("Dispatcher_PRO", pulp.LpMaximize)
        
        # Proměnné
        q_kgj   = pulp.LpVariable.dicts("q_KGJ",   range(T), 0)
        q_boil  = pulp.LpVariable.dicts("q_Boil",  range(T), 0, p['b_max'])
        q_ek    = pulp.LpVariable.dicts("q_EK",    range(T), 0, p['ek_max'])
        q_imp   = pulp.LpVariable.dicts("q_Imp",   range(T), 0, p['imp_max'] if use_ext_heat else 0)
        on      = pulp.LpVariable.dicts("on",      range(T), 0, 1, cat="Binary")
        start   = pulp.LpVariable.dicts("start",   range(T), 0, 1, cat="Binary")
        
        tes_soc = pulp.LpVariable.dicts("TES_SOC", range(T+1), 0, p['tes_cap'])
        tes_in  = pulp.LpVariable.dicts("TES_In",  range(T),   0)
        tes_out = pulp.LpVariable.dicts("TES_Out", range(T),   0)
        
        bess_soc = pulp.LpVariable.dicts("BESS_SOC", range(T+1), 0, p['bess_cap'])
        bess_cha = pulp.LpVariable.dicts("BESS_Cha", range(T),   0, p['bess_p'])
        bess_dis = pulp.LpVariable.dicts("BESS_Dis", range(T),   0, p['bess_p'])
        
        ee_export = pulp.LpVariable.dicts("ee_export", range(T), 0)
        ee_import = pulp.LpVariable.dicts("ee_import", range(T), 0)
        
        heat_shortfall = pulp.LpVariable.dicts("heat_shortfall", range(T), 0)
        heat_delivered = pulp.LpVariable.dicts("heat_delivered", range(T), 0)

        # Počáteční stavy
        model += tes_soc[0] == p['tes_cap'] * 0.5
        model += bess_soc[0] == p['bess_cap'] * 0.2

        # KGJ start/stop + min runtime
        for t in range(T):
            model += on[t] <= 1 if use_kgj else 0

        for t in range(1, T):
            model += on[t] - on[t-1] == start[t]
            # Pozn: stop není potřeba explicitně, start implikuje změnu 0→1

        # Jednoduchá aproximace min. runtime (lepší než předchozí verze)
        for t in range(T):
            for dt in range(1, int(p['k_min_runtime'])):
                if t + dt < T:
                    model += on[t + dt] >= start[t]

        obj_terms = []

        for t in range(T):
            p_ee  = float(df['ee_price'].iloc[t])
            p_gas = float(df['gas_price'].iloc[t])
            h_dem = float(df['Poptávka po teple (MW)'].iloc[t])
            fve   = float(df['FVE (MW)'].iloc[t]) if use_fve else 0.0

            # Bilance tepla
            heat_prod = q_kgj[t] + q_boil[t] + q_ek[t] + q_imp[t]
            model += tes_soc[t+1] == tes_soc[t] * (1 - p['tes_loss']) + tes_in[t] - tes_out[t]
            
            # Dodané teplo = výroba + vybití TES - nabití TES
            model += heat_delivered[t] == heat_prod + tes_out[t] - tes_in[t]
            
            # Omezení: nelze dodat víc než poptávka × pokrytí
            model += heat_delivered[t] <= h_dem * p['h_cover']
            
            # Deficit jen pokud nelze pokrýt (shortfall ≥ 0)
            model += heat_delivered[t] + heat_shortfall[t] >= h_dem * p['h_cover']

            # KGJ
            if use_kgj:
                model += q_kgj[t] <= p['k_th'] * on[t]
                model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]

            # EE bilance
            ee_kgj = q_kgj[t] * (p['k_el'] / p['k_th']) if use_kgj else 0
            model += ee_kgj + fve + ee_import[t] + bess_dis[t] == (q_ek[t] / 0.95) + bess_cha[t] + ee_export[t]
            model += bess_soc[t+1] == bess_soc[t] + bess_cha[t] * 0.90 - bess_dis[t] / 0.90

            # Objektiv
            revenue = p['h_price'] * heat_delivered[t] + (p_ee - p['dist_ee_sell']) * ee_export[t]
            costs = (p_gas + p['gas_dist']) * (q_kgj[t]/p['k_eff_th'] + q_boil[t]/0.95) + \
                    (p_ee + p['dist_ee_buy']) * ee_import[t] + \
                    p['k_start_cost'] * start[t] + \
                    p['imp_price'] * q_imp[t]
            
            obj_terms.append(revenue - costs - p['h_price'] * heat_shortfall[t])

        model += pulp.lpSum(obj_terms)
        status = model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))

        st.success(f"Optimalizace dokončena — status: {pulp.LpStatus[status]} ({model.objective.value():+.0f} €)")
