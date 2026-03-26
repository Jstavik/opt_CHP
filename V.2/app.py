import io
import streamlit as st
import pandas as pd
import numpy as np
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

# ── Session state ────────────────────────────────
for key, default in [
    ('fwd_data', None), ('avg_ee_raw', 100.0), ('avg_gas_raw', 50.0),
    ('ee_new', 100.0), ('gas_new', 50.0), ('results', None), ('df_main', None),
    ('scenario_results', None), ('selected_profile', 'free'),
    ('monthly_profile_results', None), ('sensitivity_results', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

Expand 26 hidden lines
    elif profile_type == 'extpeak':
        # 6-22 (Extended Peak)
        constraints = [h in range(6, 23) for h in df_work['hour']]
    

    elif profile_type == 'offpeak':
        # Noční hodiny 0-8 a 22-23 (off-peak v českém trhu)
        constraints = [h <= 8 or h >= 22 for h in df_work['hour']]

    elif profile_type == 'custom' and custom_hours:
        # Custom hours zadané uživatelem
        constraints = [h in custom_hours for h in df_work['hour']]

Expand 153 hidden lines
    
    profiles_to_run = st.multiselect(
        "Které profily testovat?",
        options=['free', 'base', 'peak', 'extpeak', 'custom'],
        default=['free', 'base', 'peak', 'extpeak'],
        options=['free', 'base', 'peak', 'extpeak', 'offpeak', 'custom'],
        default=['free', 'base', 'peak', 'extpeak', 'offpeak'],
        help="Spusť optimalizaci pro vybrané profily a porovnej je"
    )
    

    profile_definitions = {
        'free': {'name': 'Volná Opt.', 'hours': None, 'desc': 'Bez omezení'},
        'base': {'name': 'Base (0-24h)', 'hours': list(range(24)), 'desc': 'Celý den'},
        'peak': {'name': 'Peak (9-21h)', 'hours': list(range(9, 22)), 'desc': '12 hodin'},
        'extpeak': {'name': 'ExtPeak (6-22h)', 'hours': list(range(6, 23)), 'desc': '16 hodin'},
        'free':    {'name': 'Volná Opt.',          'hours': None,                          'desc': 'Bez omezení'},
        'base':    {'name': 'Base (0-24h)',         'hours': list(range(24)),               'desc': 'Celý den'},
        'peak':    {'name': 'Peak (9-21h)',         'hours': list(range(9, 22)),            'desc': '12 hodin'},
        'extpeak': {'name': 'ExtPeak (6-22h)',      'hours': list(range(6, 23)),            'desc': '16 hodin'},
        'offpeak': {'name': 'Offpeak (0-8,22-23h)', 'hours': list(range(0, 9))+[22, 23],   'desc': '11 hodin'},
    }
    
    custom_hours = None

Expand 23 hidden lines
    st.subheader("4️⃣ Režim Porovnání")
    scenario_mode = st.radio(
        "Spusť optimalizaci:",
        ["Pouze aktuální nastavení", "Všechny vybrané profily"],
        help="'Všechny profily' = porovnání scénářů"
        ["Pouze aktuální nastavení", "Všechny vybrané profily", "Měsíční analýza profilů"],
        help="'Všechny profily' = porovnání scénářů | 'Měsíční analýza' = nejlepší profil per měsíc"
    )

# ────────────────────────────────────────────────

Expand 90 hidden lines
        if p['kgj_gas_fix']:
            p['kgj_gas_fix_price'] = st.number_input("Fixní cena plynu – KGJ [€/MWh]",
                value=float(st.session_state.avg_gas_raw))
        # Proměnná účinnost dle výkonu
        p['kgj_var_eff'] = st.checkbox("Proměnná účinnost dle výkonu", value=False,
            help="Linearizovaná 2-bodová křivka: účinnost při min. zátěži vs. jmenovitém výkonu")
        if p['kgj_var_eff']:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                p['eta_th_min'] = st.number_input("η_th při min. zátěži [-]",
                    value=round(p['k_eff_th'] * 0.9, 3), min_value=0.1, max_value=1.0, step=0.01,
                    help="Tepelná účinnost při minimálním zatížení KGJ")
            with col_v2:
                p['eta_el_min'] = st.number_input("η_el při min. zátěži [-]",
                    value=round(p['k_eff_el'] * 0.9, 3), min_value=0.05, max_value=1.0, step=0.01,
                    help="Elektrická účinnost při minimálním zatížení KGJ")
            st.caption(f"ℹ️ Při jmenovitém výkonu: η_th={p['k_eff_th']}, η_el={p['k_eff_el']}")

    # ── Kotel ─────────────────────────────────────
    if use_boil:

Expand 59 hidden lines
        if p['imp_hour_limit_on']:
            p['imp_hour_limit'] = st.number_input("Max. hodin importu tepla / rok", value=2000, min_value=1)

# ────────────────────────────────────────────────
# POMOCNÉ FUNKCE PRO SOLVER
# ────────────────────────────────────────────────

def compute_linear_fuel_params(k_th, k_min, eta_th_rated, eta_th_min, eta_el_rated, eta_el_min):
    """
    Linearizace fuel/output funkce přes 2 provozní body: min zátěž a jmenovitý výkon.

    Vrátí (c0_th, c1_th, c0_el, c1_el) kde:
      gas_consumed[t] = c0_th * on[t] + c1_th * q_kgj[t]   [MWh_gas/h]
      ee_kgj[t]       = c0_el * on[t] + c1_el * q_kgj[t]   [MWh_el/h]

    Optimalizace zůstává lineární (LP) – žádné nové binary proměnné.
    """
    q_min = k_min * k_th
    q_max = k_th
    fuel_min = q_min / eta_th_min
    fuel_max = q_max / eta_th_rated
    c1_th = (fuel_max - fuel_min) / (q_max - q_min)
    c0_th = fuel_min - c1_th * q_min   # offset platí jen když on=1

    ee_min = q_min * (eta_el_min / eta_th_min)
    ee_max = q_max * (eta_el_rated / eta_th_rated)
    c1_el = (ee_max - ee_min) / (q_max - q_min)
    c0_el = ee_min - c1_el * q_min

    return c0_th, c1_th, c0_el, c1_el


# ────────────────────────────────────────────────
# ENHANCED SOLVER S PROFILE SUPPORT
# ────────────────────────────────────────────────

Expand 18 hidden lines
    boil_eff = p.get('boil_eff', 0.95)
    ek_eff   = p.get('ek_eff',   0.98)

    # Koeficienty pro linearizovanou účinnostní křivku KGJ
    if p.get('kgj_var_eff') and u.get('kgj'):
        c0_th, c1_th, c0_el, c1_el = compute_linear_fuel_params(
            p['k_th'], p['k_min'],
            p['k_eff_th'], p.get('eta_th_min', p['k_eff_th']),
            p['k_eff_el'], p.get('eta_el_min', p['k_eff_el'])
        )
    else:
        c0_th, c1_th = 0.0, 1.0 / p.get('k_eff_th', 0.46)
        c0_el, c1_el = 0.0, p.get('k_eff_el', 0.40) / p.get('k_eff_th', 0.46)

    # Filtruj podle období
    if period_mask is not None:
        df = df[period_mask].reset_index(drop=True)

Expand 121 hidden lines
        model += heat_delivered + heat_shortfall[t] >= h_dem * p['h_cover']
        model += heat_delivered <= h_dem + 1e-3

        ee_kgj_out = q_kgj[t] * (p['k_eff_el'] / p['k_eff_th']) if u['kgj'] else 0
        ee_kgj_out = (c0_el * on[t] + c1_el * q_kgj[t]) if u['kgj'] else 0
        ee_ek_in   = q_ek[t] / ek_eff                            if u['ek']  else 0
        model += ee_kgj_out + fve_p + ee_import[t] + bess_dis[t] == ee_ek_in + bess_cha[t] + ee_export[t]


Expand 6 hidden lines
        revenue = (h_price * heat_delivered
                   + (p_ee_m - dist_sell_net - fve_dist_sell_cost) * ee_export[t])
        costs = (
            ((p_gas_kgj  + p['gas_dist']) * (q_kgj[t]  / p['k_eff_th']) if u['kgj']      else 0) +
            ((p_gas_kgj  + p['gas_dist']) * (c0_th * on[t] + c1_th * q_kgj[t]) if u['kgj'] else 0) +
            ((p_gas_boil + p['gas_dist']) * (q_boil[t] / boil_eff)       if u['boil']     else 0) +
            (p_ee_m + dist_buy_net) * ee_import[t] +
            ((p_ee_ek + dist_buy_net) * ee_ek_in                          if u['ek']       else 0) +

Expand 30 hidden lines
        'Shortfall [MW]':       [vv(heat_shortfall, t) for t in range(T)],
        'EE export [MW]':       [vv(ee_export, t) for t in range(T)],
        'EE import [MW]':       [vv(ee_import, t) for t in range(T)],
        'EE z KGJ [MW]':        [vv(q_kgj, t)*(p['k_eff_el']/p['k_eff_th']) if u['kgj'] else 0.0 for t in range(T)],
        'EE z KGJ [MW]':        [(c0_el * vv(on, t) + c1_el * vv(q_kgj, t)) if u['kgj'] else 0.0 for t in range(T)],
        'EE z FVE [MW]':        [float(df['FVE (MW)'].iloc[t]) if (u['fve'] and 'FVE (MW)' in df.columns) else 0.0 for t in range(T)],
        'EE do EK [MW]':        [vv(q_ek, t)/ek_eff if u['ek'] else 0.0 for t in range(T)],
        'Cena EE [€/MWh]':     (df['ee_price'] + ee_delta).values,

Expand 29 hidden lines

        rt  = h_price * res['Dodáno tepla [MW]'].iloc[t]
        re  = (p_ee_m - dist_s - fve_ds) * res['EE export [MW]'].iloc[t]
        cg1 = (p_gas_kj + p['gas_dist']) * (res['KGJ [MW_th]'].iloc[t]  / p['k_eff_th']) if u['kgj']  else 0
        cg1 = (p_gas_kj + p['gas_dist']) * (c0_th * res['KGJ on'].iloc[t] + c1_th * res['KGJ [MW_th]'].iloc[t]) if u['kgj'] else 0
        cg2 = (p_gas_bh + p['gas_dist']) * (res['Kotel [MW_th]'].iloc[t] / boil_eff)      if u['boil'] else 0
        ce1 = (p_ee_m + dist_b) * res['EE import [MW]'].iloc[t]
        ce2 = (p_ee_ekh + dist_b) * res['EE do EK [MW]'].iloc[t] if u['ek'] else 0

Expand 81 hidden lines
    
    return scenarios


def run_monthly_profile_analysis(df, params, uses, profiles_to_run,
                                  custom_hours=None, max_starts_per_month=None):
    """
    Pro každý měsíc v datech × každý profil spustí optimalizaci.
    Každý měsíc je nezávislý (TES/BESS startuje od 50 % kapacity).
    Vrátí: {month_int: {profile_str: {profit, profit_per_h, smoothness}}}
    """
    results = {}
    months = sorted(pd.to_datetime(df['datetime']).dt.month.unique())
    total_runs = len(months) * len(profiles_to_run)

    progress = st.progress(0)
    status = st.empty()
    run_idx = 0

    for month in months:
        mask = pd.to_datetime(df['datetime']).dt.month == month
        results[month] = {}
        for profile in profiles_to_run:
            status.write(f"⏳ Měsíc **{MONTH_NAMES.get(month, month)}**, profil **{profile.upper()}**...")
            r = run_optimization_with_profile(
                df=df, params=params, uses=uses,
                profile_type=profile,
                custom_hours=custom_hours if profile == 'custom' else None,
                max_starts_per_month=max_starts_per_month,
                period_mask=mask
            )
            if r is not None:
                n_hours = int(mask.sum())
                results[month][profile] = {
                    'profit':       r['total_profit'],
                    'profit_per_h': r['total_profit'] / n_hours if n_hours > 0 else 0,
                    'smoothness':   calculate_smoothness_metrics(r['res']),
                }
            run_idx += 1
            progress.progress(run_idx / total_runs)

    progress.empty()
    status.empty()
    return results


# ────────────────────────────────────────────────
# CITLIVOSTNÍ ANALÝZA
# ────────────────────────────────────────────────

def run_sensitivity_analysis(df, params, uses, profile_type, gas_range, ee_range, steps,
                              custom_hours=None):
    """
    Variuje gas_delta a ee_delta symetricky okolo základní varianty.
    Vrátí DataFrame: typ | delta | profit | delta_pct
    """
    base = run_optimization_with_profile(df, params, uses, profile_type,
                                          custom_hours=custom_hours)
    if base is None:
        return None
    base_profit = base['total_profit']
    if abs(base_profit) < 1e-6:
        base_profit = 1.0  # ochrana před dělením nulou

    rows = []
    gas_vals = np.linspace(-gas_range, gas_range, steps)
    ee_vals  = np.linspace(-ee_range,  ee_range,  steps)

    for gd in gas_vals:
        r = run_optimization_with_profile(df, params, uses, profile_type,
                                          gas_delta=gd, custom_hours=custom_hours)
        if r:
            rows.append({'typ': 'Cena plynu', 'delta': round(gd, 2),
                         'profit': r['total_profit'],
                         'delta_pct': (r['total_profit'] - base_profit) / abs(base_profit) * 100})

    for ed in ee_vals:
        r = run_optimization_with_profile(df, params, uses, profile_type,
                                          ee_delta=ed, custom_hours=custom_hours)
        if r:
            rows.append({'typ': 'Cena EE', 'delta': round(ed, 2),
                         'profit': r['total_profit'],
                         'delta_pct': (r['total_profit'] - base_profit) / abs(base_profit) * 100})

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────
# LOKÁLNÍ DATA + SPUŠTĚNÍ OPTIMALIZACE
# ────────────────────────────────────────────────

Expand 51 hidden lines
                st.session_state.scenario_results = None  # Vymaž scenario results
                st.success("✅ Optimalizace dokončena!")
        
        else:  # Scenario Mode
        elif scenario_mode == "Všechny vybrané profily":
            st.markdown("### 📊 Režim: Porovnání Profilů")
            st.markdown(
                f"Spustit se budou profily: **{', '.join([p.upper() for p in profiles_to_run])}**. "
                f"Spustit se budou profily: **{', '.join([pr.upper() for pr in profiles_to_run])}**. "
                f"Poté se zobrazí přehledné porovnání jejich ekonomiky a kvality provozu."
            )
            

            if st.button("🔬 Spustit analýzu scénářů", type="primary", key="btn_scenarios"):
                with st.spinner("⏳ Probíhá scenáristická analýza …"):
                    scenarios = run_scenario_analysis(

Expand 6 hidden lines
                        period_end=period_end,
                        max_starts_per_month=max_starts_per_month if use_month_start_limit else None
                    )
                

                if not scenarios:
                    st.error("❌ Žádný scenář nebyl úspěšně vypočten!")
                    st.stop()
                

                st.session_state.scenario_results = scenarios
                st.session_state.results = None  # Vymaž single result
                st.session_state.results = None
                st.session_state.monthly_profile_results = None
                st.success("✅ Scenáristická analýza dokončena!")
    

        else:  # Měsíční analýza profilů
            st.markdown("### 🗓️ Režim: Měsíční Analýza Profilů")
            n_runs = len(profiles_to_run) * pd.to_datetime(df['datetime']).dt.month.nunique()
            st.markdown(
                f"Systém spustí optimalizaci pro každý **měsíc × profil** "
                f"({n_runs} kombinací) a ukáže, který profil je optimální pro který měsíc."
            )

            if st.button("🗓️ Spustit měsíční analýzu", type="primary", key="btn_monthly"):
                with st.spinner("⏳ Probíhá měsíční analýza …"):
                    monthly_res = run_monthly_profile_analysis(
                        df=df, params=p, uses=uses,
                        profiles_to_run=profiles_to_run,
                        custom_hours=custom_hours if 'custom' in profiles_to_run else None,
                        max_starts_per_month=max_starts_per_month if use_month_start_limit else None
                    )

                st.session_state.monthly_profile_results = monthly_res
                st.session_state.results = None
                st.session_state.scenario_results = None
                st.success("✅ Měsíční analýza dokončena!")

    with col_mode_2:
        st.markdown("")
        st.markdown("")

Expand 6 hidden lines
            )
            st.session_state.selected_profile = selected_profile

# ────────────────────────────────────────────────
# MONTHLY PROFILE ANALYSIS VIEW
# ────────────────────────────────────────────────
if st.session_state.monthly_profile_results is not None:
    monthly_pr = st.session_state.monthly_profile_results
    st.divider()
    st.subheader("🗓️ Měsíční Analýza Profilů")

    # Sestavit tabulku a heatmapu
    all_profiles = sorted({pr for m_data in monthly_pr.values() for pr in m_data.keys()})
    months_sorted = sorted(monthly_pr.keys())

    # Tabulka: nejlepší profil per měsíc
    best_rows = []
    for month in months_sorted:
        m_data = monthly_pr[month]
        if not m_data:
            continue
        best_pr = max(m_data, key=lambda pr: m_data[pr]['profit_per_h'])
        best_rows.append({
            'Měsíc':           MONTH_NAMES.get(month, month),
            'Nejlepší profil': best_pr.upper(),
            'Zisk/hod [€]':    f"{m_data[best_pr]['profit_per_h']:,.1f}",
            'Zisk celkem [€]': f"{m_data[best_pr]['profit']:,.0f}",
            'Stabilita [%]':   f"{m_data[best_pr]['smoothness']['stability_score']:.1f}",
        })

    if best_rows:
        df_best = pd.DataFrame(best_rows)
        st.dataframe(df_best, use_container_width=True, hide_index=True)
        total_opt = sum(
            monthly_pr[m][max(monthly_pr[m], key=lambda pr: monthly_pr[m][pr]['profit'])]['profit']
            for m in months_sorted if monthly_pr[m]
        )
        st.success(f"💡 Celkový potenciál při optimálním výběru profilu per měsíc: **{total_opt:,.0f} €**")

    # Heatmapa: profily × měsíce, hodnota = profit/hod
    heat_z, heat_x, heat_y = [], all_profiles, [MONTH_NAMES.get(m, m) for m in months_sorted]
    for pr in all_profiles:
        row = [monthly_pr[m].get(pr, {}).get('profit_per_h', None) for m in months_sorted]
        heat_z.append(row)

    fig_heat = go.Figure(go.Heatmap(
        z=heat_z, x=heat_y, y=[pr.upper() for pr in heat_x],
        colorscale='RdYlGn', zmid=0,
        colorbar=dict(title='€/hod'),
        hovertemplate='Měsíc: %{x}<br>Profil: %{y}<br>Zisk/hod: %{z:.1f} €<extra></extra>'
    ))
    fig_heat.update_layout(
        height=320, title="Zisk/hod [€] dle profilu a měsíce",
        xaxis_title="Měsíc", yaxis_title="Profil"
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ────────────────────────────────────────────────
# SCENARIO COMPARISON VIEW
# ────────────────────────────────────────────────

Expand 387 hidden lines
    # Excel Export
    st.divider()
    st.subheader("⬇️ Export výsledků")
    
    def to_excel(df_res, df_input, params, monthly_df):

    def to_excel(df_res, df_input, params, monthly_df, scenario_comparison_df=None):
        buf = io.BytesIO()
        skip_cols = {'Měsíc','Hodina dne','KGJ on','Kotel on','Import tepla on'}
        df_exp    = df_res[[c for c in df_res.columns if c not in skip_cols]].copy()
        
        skip_cols = {'Měsíc', 'Hodina dne', 'KGJ on', 'Kotel on', 'Import tepla on'}
        df_exp = df_res[[c for c in df_res.columns if c not in skip_cols]].copy()

        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            df_exp.to_excel(writer, index=False, sheet_name='Hodinová data')
            monthly_df.to_excel(writer, index=False, sheet_name='Měsíční souhrn')
            df_input.to_excel(writer, index=False, sheet_name='Vstupní data')
        

            # List s parametry konfigurace
            params_df = pd.DataFrame([
                {'Parametr': k, 'Hodnota': str(v)}
                for k, v in params.items()
                if not isinstance(v, (dict, list))
            ])
            params_df.to_excel(writer, index=False, sheet_name='Parametry')

            # List se scénáři (pokud existuje)
            if scenario_comparison_df is not None:
                scenario_comparison_df.to_excel(writer, index=False, sheet_name='Scénáře')

            # xlsxwriter formátování – tučné hlavičky a šířky sloupců
            workbook  = writer.book
            hdr_fmt   = workbook.add_format({
                'bold': True, 'bg_color': '#D9E1F2',
                'border': 1, 'text_wrap': False
            })
            num_fmt   = workbook.add_format({'num_format': '#,##0.00'})
            for sheet_name, df_sheet in [
                ('Hodinová data',  df_exp),
                ('Měsíční souhrn', monthly_df),
                ('Vstupní data',   df_input),
                ('Parametry',      params_df),
            ] + ([('Scénáře', scenario_comparison_df)] if scenario_comparison_df is not None else []):
                ws = writer.sheets[sheet_name]
                ws.set_column(0, max(len(df_sheet.columns) - 1, 0), 18)
                for col_idx, col_name in enumerate(df_sheet.columns):
                    ws.write(0, col_idx, col_name, hdr_fmt)

        return buf.getvalue()
    
    xlsx = to_excel(res.round(4), df, p, monthly)

    scenario_comp_df = create_scenario_comparison_df(st.session_state.scenario_results) \
        if st.session_state.scenario_results else None
    xlsx = to_excel(res.round(4), df, p, monthly, scenario_comp_df)
    st.download_button(
        label="📥 Stáhnout výsledky (Excel .xlsx)",
        data=xlsx,
        file_name="kgj_optimalizace.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ─────────────────────────────────────────────
    # CITLIVOSTNÍ ANALÝZA
    # ─────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Citlivostní analýza")
    st.markdown("Zobrazí, jak se změní zisk při změně tržní ceny plynu nebo elektřiny.")

    with st.expander("⚙️ Nastavení citlivostní analýzy", expanded=False):
        col_sa1, col_sa2, col_sa3 = st.columns(3)
        with col_sa1:
            sa_gas_range = st.slider("Rozsah ceny plynu [±€/MWh]", 5, 50, 20, step=5,
                                     key="sa_gas_range")
        with col_sa2:
            sa_ee_range  = st.slider("Rozsah ceny EE [±€/MWh]",    5, 50, 20, step=5,
                                     key="sa_ee_range")
        with col_sa3:
            sa_steps     = st.selectbox("Počet kroků:", [3, 5, 7, 9], index=1, key="sa_steps")

        # Výběr profilu pro citlivostní analýzu
        available_profiles = ['free']
        if st.session_state.scenario_results:
            available_profiles = list(st.session_state.scenario_results.keys())
        sa_profile = st.selectbox(
            "Profil pro analýzu:",
            options=available_profiles,
            format_func=lambda x: x.upper(),
            key="sa_profile"
        )

    if st.button("📊 Spustit citlivostní analýzu", key="btn_sensitivity"):
        with st.spinner("⏳ Probíhá citlivostní analýza …"):
            sa_df = run_sensitivity_analysis(
                df=df, params=p, uses=uses,
                profile_type=sa_profile,
                gas_range=sa_gas_range, ee_range=sa_ee_range, steps=sa_steps,
                custom_hours=custom_hours if sa_profile == 'custom' else None
            )
        if sa_df is None:
            st.error("❌ Citlivostní analýza selhala.")
        else:
            st.session_state.sensitivity_results = sa_df.to_dict('records')
            st.success("✅ Hotovo!")

    if st.session_state.sensitivity_results:
        sa_df = pd.DataFrame(st.session_state.sensitivity_results)

        # Tornado chart – min/max per parametr
        tornado_rows = []
        for typ, grp in sa_df.groupby('typ'):
            tornado_rows.append({
                'Parametr': typ,
                'Min zisk [k€]': grp['profit'].min() / 1000,
                'Max zisk [k€]': grp['profit'].max() / 1000,
            })
        df_tornado = pd.DataFrame(tornado_rows)

        fig_t = go.Figure()
        for _, row in df_tornado.iterrows():
            fig_t.add_trace(go.Bar(
                y=[row['Parametr']],
                x=[row['Max zisk [k€]'] - row['Min zisk [k€]']],
                base=row['Min zisk [k€]'],
                orientation='h',
                name=row['Parametr'],
                text=f"{row['Min zisk [k€]']:.1f} → {row['Max zisk [k€]']:.1f} k€",
                textposition='inside'
            ))
        fig_t.update_layout(
            height=250, title="Tornádo chart – rozsah zisku dle cenové změny",
            xaxis_title="Zisk [k€]", showlegend=False, bargap=0.4
        )
        st.plotly_chart(fig_t, use_container_width=True)

        # Detailní tabulka
        st.dataframe(
            sa_df[['typ', 'delta', 'profit', 'delta_pct']].rename(columns={
                'typ': 'Parametr', 'delta': 'Δ cena [€/MWh]',
                'profit': 'Zisk [€]', 'delta_pct': 'Změna [%]'
            }).round(2),
            use_container_width=True, hide_index=True
        )

.gitignore
+10
__pycache__/
*.pyc
*.pyo
.streamlit/
.env
*.egg-info/
dist/
build/
.venv/
venv/
