import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go

# ... (předchozí importy a nastavení stejné) ...

# --- 5. KROK: OPTIMALIZACE ---
if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    if st.button("🏁 SPUSTIT KOMPLETNÍ OPTIMALIZACI"):
        df = pd.merge(st.session_state.fwd_data, st.session_state.loc_data, on='mdh', how='inner')
        T = len(df)
        model = pulp.LpProblem("Dispatcher", pulp.LpMaximize)
        
        # PROMĚNNÉ [cite: 13]
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0, params.get('k_th', 0))
        q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, params.get('b_max', 0))
        q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, params.get('ek_max', 0))
        q_ext = pulp.LpVariable.dicts("q_Ext", range(T), 0)
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        
        # Slack proměnná pro nepokryté teplo (deficit)
        q_deficit = pulp.LpVariable.dicts("q_Deficit", range(T), 0)

        kgj_gas_ratio = (params.get('k_th', 1) / params.get('k_eff', 1)) / params.get('k_th', 1)
        kgj_el_ratio = params.get('k_el', 0) / params.get('k_th', 1)

        profit_total = []
        for t in range(T):
            ee = df.loc[t, 'ee_price']
            gas = df.loc[t, 'gas_price']
            hp = df.loc[t, 'heat_price'] if 'heat_price' in df.columns else params['fixed_heat_price']
            dem = df.loc[t, 'demand']
            h_req = dem * params['h_cover']

            # BILANCE TEPLA: Zdroje + Deficit se musí rovnat požadavku [cite: 14, 15]
            model += q_kgj[t] + q_boil[t] + q_ek[t] + q_ext[t] + q_deficit[t] >= h_req
            
            # OMEZENÍ TECHNOLOGIÍ
            if not use_kgj: model += q_kgj[t] == 0
            else:
                model += q_kgj[t] <= params['k_th'] * on[t]
                model += q_kgj[t] >= params['k_min'] * params['k_th'] * on[t]
            
            if not use_boil: model += q_boil[t] == 0
            if not use_ek: model += q_ek[t] == 0
            if not (use_ext_heat and 'ext_price' in df.columns): model += q_ext[t] == 0

            # EKONOMIKA [cite: 16, 17]
            income = (hp * (h_req - q_deficit[t])) + (ee * q_kgj[t] * kgj_el_ratio)
            costs = (gas * (q_kgj[t] * kgj_gas_ratio)) + \
                    (gas * (q_boil[t] / params.get('b_eff', 0.95))) + \
                    ((ee + params.get('dist_ee', 33)) * (q_ek[t] / params.get('ek_eff', 0.98))) + \
                    (params.get('k_serv', 12) * on[t])
            
            if use_ext_heat and 'ext_price' in df.columns:
                costs += df.loc[t, 'ext_price'] * q_ext[t]
            
            # PENALIZACE DEFICITU: Obrovská pokuta za nedodání tepla (např. 5000 EUR/MWh)
            # To zajistí, že model použije všechny dostupné zdroje, než to vzdá.
            penalty = q_deficit[t] * 5000 

            profit_total.append(income - costs - penalty)

        model += pulp.lpSum(profit_total)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # --- VÝSLEDKY ---
        st.success(f"Optimalizace hotova. Celkový hrubý zisk (po započtení penalizací): {pulp.value(model.objective):,.0f} EUR")
        
        t_col = 'datetime_x' if 'datetime_x' in df.columns else ('datetime' if 'datetime' in df.columns else df.columns[0])
        res = pd.DataFrame({
            'T': df[t_col],
            'KGJ': [q_kgj[t].value() for t in range(T)],
            'Kotel': [q_boil[t].value() for t in range(T)],
            'EK': [q_ek[t].value() for t in range(T)],
            'Nákup': [q_ext[t].value() for t in range(T)],
            'Deficit': [q_deficit[t].value() for t in range(T)],
            'Poptávka': df['demand'] * params['h_cover']
        })

        # GRAF 1: DISPATCH ZDROJŮ
        fig1 = go.Figure()
        colors = {'KGJ': '#FF9900', 'Kotel': '#1f77b4', 'EK': '#2ca02c', 'Nákup': '#d62728'}
        for c in ['KGJ', 'Kotel', 'EK', 'Nákup']:
            if res[c].sum() > 0.001:
                fig1.add_trace(go.Bar(x=res['T'], y=res[c], name=c, marker_color=colors[c]))
        fig1.update_layout(barmode='stack', title="Hodinový Dispatch zdrojů [MW]")
        st.plotly_chart(fig1, use_container_width=True)

        # GRAF 2: NEPOKRYTÉ TEPLO (DEFICIT)
        if res['Deficit'].sum() > 0.1:
            st.warning("⚠️ POZOR: Systém v některých hodinách nedokáže pokrýt poptávku!")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=res['T'], y=res['Deficit'], fill='tozeroy', name="Nedostatek tepla", line=dict(color='black')))
            fig2.update_layout(title="Graf nepokrytého tepla (Deficit) [MW]", yaxis_title="Chybějící výkon [MW]")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("✅ Poptávka je plně pokryta dostupnými zdroji.")
