import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

# ────────────────────────────────────────────────
# Inicializace session state
# ────────────────────────────────────────────────
if 'fwd_data' not in st.session_state:
    st.session_state.fwd_data = None
if 'avg_ee_raw' not in st.session_state:
    st.session_state.avg_ee_raw = 100.0
if 'avg_gas_raw' not in st.session_state:
    st.session_state.avg_gas_raw = 50.0

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

# ────────────────────────────────────────────────
# SIDEBAR – technologie + FWD
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Technologie na lokalitě")
    use_kgj      = st.checkbox("Kogenerace (KGJ)",         value=True)
    use_boil     = st.checkbox("Plynový kotel",             value=True)
    use_ek       = st.checkbox("Elektrokotel",              value=True)
    use_tes      = st.checkbox("Nádrž (TES)",               value=True)
    use_bess     = st.checkbox("Baterie (BESS)",            value=True)
    use_fve      = st.checkbox("Fotovoltaika (FVE)",        value=True)
    use_ext_heat = st.checkbox("Nákup tepla (Import)",      value=True)

    st.divider()
    st.header("📈 Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])

    if fwd_file is not None:
        try:
            df_raw = pd.read_excel(fwd_file)
            df_raw.columns = [str(c).strip() for c in df_raw.columns]
            date_col = df_raw.columns[0]
            df_raw[date_col] = pd.to_datetime(df_raw[date_col], dayfirst=True)

            years = sorted(df_raw[date_col].dt.year.unique())
            sel_year = st.selectbox("Rok pro analýzu", years)
            df_year = df_raw[df_raw[date_col].dt.year == sel_year].copy()

            avg_ee  = float(df_year.iloc[:, 1].mean())
            avg_gas = float(df_year.iloc[:, 2].mean())
            st.session_state.avg_ee_raw  = avg_ee
            st.session_state.avg_gas_raw = avg_gas

            st.info(f"Průměr EE: **{avg_ee:.1f} €/MWh** | Plyn: **{avg_gas:.1f} €/MWh**")

            ee_new  = st.number_input("Cílová base cena EE [€/MWh]",  value=round(avg_ee, 1),  step=1.0)
            gas_new = st.number_input("Cílová base cena Plyn [€/MWh]", value=round(avg_gas, 1), step=1.0)

            df_fwd = df_year.copy()
            df_fwd.columns = ['datetime', 'ee_original', 'gas_original']
            # Posun celé křivky o delta od původního průměru
            df_fwd['ee_price']  = df_fwd['ee_original']  + (ee_new  - avg_ee)
            df_fwd['gas_price'] = df_fwd['gas_original'] + (gas_new - avg_gas)
            st.session_state.fwd_data   = df_fwd
            st.session_state.ee_new     = ee_new
            st.session_state.gas_new    = gas_new

            st.success("FWD načteno ✔")
        except Exception as e:
            st.error(f"Chyba při načítání FWD: {e}")

# ────────────────────────────────────────────────
# Graf FWD křivky (originál vs. upravená)
# ────────────────────────────────────────────────
if st.session_state.fwd_data is not None:
    df_fwd = st.session_state.fwd_data
    with st.expander("📈 FWD křivka – originál vs. upravená", expanded=True):
        tab_ee, tab_gas = st.tabs(["Elektřina [€/MWh]", "Plyn [€/MWh]"])

        with tab_ee:
            fig_fwd_ee = go.Figure()
            fig_fwd_ee.add_trace(go.Scatter(
                x=df_fwd['datetime'], y=df_fwd['ee_original'],
                name='EE – originál', line=dict(color='#95a5a6', width=1, dash='dot')
            ))
            fig_fwd_ee.add_trace(go.Scatter(
                x=df_fwd['datetime'], y=df_fwd['ee_price'],
                name='EE – upravená', line=dict(color='#2ecc71', width=2)
            ))
            fig_fwd_ee.add_hline(
                y=st.session_state.avg_ee_raw,
                line_dash="dash", line_color="#95a5a6",
                annotation_text=f"Orig. průměr {st.session_state.avg_ee_raw:.1f}"
            )
            fig_fwd_ee.add_hline(
                y=st.session_state.get('ee_new', st.session_state.avg_ee_raw),
                line_dash="dash", line_color="#27ae60",
                annotation_text=f"Nový průměr {st.session_state.get('ee_new', st.session_state.avg_ee_raw):.1f}"
            )
            fig_fwd_ee.update_layout(height=350, hovermode='x unified', margin=dict(t=30))
            st.plotly_chart(fig_fwd_ee, use_container_width=True)

        with tab_gas:
            fig_fwd_gas = go.Figure()
            fig_fwd_gas.add_trace(go.Scatter(
                x=df_fwd['datetime'], y=df_fwd['gas_original'],
                name='Plyn – originál', line=dict(color='#95a5a6', width=1, dash='dot')
            ))
            fig_fwd_gas.add_trace(go.Scatter(
                x=df_fwd['datetime'], y=df_fwd['gas_price'],
                name='Plyn – upravená', line=dict(color='#e67e22', width=2)
            ))
            fig_fwd_gas.add_hline(
                y=st.session_state.avg_gas_raw,
                line_dash="dash", line_color="#95a5a6",
                annotation_text=f"Orig. průměr {st.session_state.avg_gas_raw:.1f}"
            )
            fig_fwd_gas.add_hline(
                y=st.session_state.get('gas_new', st.session_state.avg_gas_raw),
                line_dash="dash", line_color="#e67e22",
                annotation_text=f"Nový průměr {st.session_state.get('gas_new', st.session_state.avg_gas_raw):.1f}"
            )
            fig_fwd_gas.update_layout(height=350, hovermode='x unified', margin=dict(t=30))
            st.plotly_chart(fig_fwd_gas, use_container_width=True)

# ────────────────────────────────────────────────
# Parametry – dynamicky dle zaškrtnutých technologií
# ────────────────────────────────────────────────
p = {}

t_gen, t_tech = st.tabs(["Obecné", "Technika"])

with t_gen:
    col1, col2 = st.columns(2)
    with col1:
        p['dist_ee_buy']  = st.number_input("Distribuce nákup EE [€/MWh]",  value=33.0)
        p['dist_ee_sell'] = st.number_input("Distribuce prodej EE [€/MWh]",  value=2.0)
        p['gas_dist']     = st.number_input("Distribuce plyn [€/MWh]",       value=5.0)
    with col2:
        p['internal_ee_use'] = st.checkbox("Ušetřit distribuci při interní spotřebě EE", value=True)
        p['h_price']  = st.number_input("Prodejní cena tepla [€/MWh]", value=120.0)
        p['h_cover']  = st.slider("Minimální pokrytí poptávky", 0.0, 1.0, 0.99, step=0.01)
        p['shortfall_penalty'] = st.number_input(
            "Penalizace za nedodání tepla [€/MWh]",
            value=500.0,
            help="Vysoká hodnota = optimizer prioritizuje pokrytí tepla nad ziskem"
        )

with t_tech:
    if use_kgj:
        st.subheader("Kogenerace (KGJ)")
        c1, c2 = st.columns(2)
        with c1:
            p['k_th']          = st.number_input("Tepelný výkon [MW]",          value=1.09)
            p['k_el']          = st.number_input("Elektrický výkon [MW]",        value=1.0)
            p['k_eff_th']      = st.number_input("Tepelná účinnost [-]",         value=0.46)
            p['k_min']         = st.slider("Min. zatížení [%]", 0, 100, 55) / 100
        with c2:
            p['k_start_cost']  = st.number_input("Náklady na start [€/start]",  value=1200.0)
            p['k_min_runtime'] = st.number_input("Min. doba běhu [hod]",        value=4, min_value=1)
        p['kgj_gas_fix'] = st.checkbox("Fixní cena plynu pro KGJ")
        if p['kgj_gas_fix']:
            p['kgj_gas_fix_price'] = st.number_input(
                "Fixní cena plynu – KGJ [€/MWh]",
                value=float(st.session_state.avg_gas_raw)
            )

    if use_boil:
        st.subheader("Plynový kotel")
        p['b_max']      = st.number_input("Max. výkon [MW]", value=3.91)
        p['boil_eff']   = st.number_input("Účinnost kotle [-]", value=0.95)
        p['boil_gas_fix'] = st.checkbox("Fixní cena plynu pro kotel")
        if p['boil_gas_fix']:
            p['boil_gas_fix_price'] = st.number_input(
                "Fixní cena plynu – kotel [€/MWh]",
                value=float(st.session_state.avg_gas_raw)
            )

    if use_ek:
        st.subheader("Elektrokotel")
        p['ek_max'] = st.number_input("Max. výkon [MW]", value=0.61)
        p['ek_eff'] = st.number_input("Účinnost EK [-]",  value=0.98)
        p['ek_ee_fix'] = st.checkbox("Fixní cena EE pro elektrokotel")
        if p['ek_ee_fix']:
            p['ek_ee_fix_price'] = st.number_input(
                "Fixní cena EE – EK [€/MWh]",
                value=float(st.session_state.avg_ee_raw)
            )

    if use_tes:
        st.subheader("Nádrž TES")
        p['tes_cap']  = st.number_input("Kapacita [MWh]", value=10.0)
        p['tes_loss'] = st.number_input("Ztráta [%/h]",   value=0.5) / 100

    if use_bess:
        st.subheader("Baterie BESS")
        p['bess_cap']        = st.number_input("Kapacita [MWh]",         value=1.0)
        p['bess_p']          = st.number_input("Max. výkon [MW]",         value=0.5)
        p['bess_eff']        = st.number_input("Účinnost nabíjení/vybíjení [-]", value=0.90)
        p['bess_cycle_cost'] = st.number_input("Náklady na cyklus [€/MWh]",      value=5.0,
                                               help="Náklady na opotřebení za každou MWh proteklou baterií")
        p['bess_ee_fix'] = st.checkbox("Fixní cena EE pro BESS")
        if p['bess_ee_fix']:
            p['bess_ee_fix_price'] = st.number_input(
                "Fixní cena EE – BESS [€/MWh]",
                value=float(st.session_state.avg_ee_raw)
            )

    if use_fve:
        st.subheader("Fotovoltaika FVE")
        p['fve_installed_p'] = st.number_input("Instalovaný výkon [MW]", value=1.0)

    if use_ext_heat:
        st.subheader("Nákup tepla (Import)")
        p['imp_max']   = st.number_input("Max. výkon [MW]",      value=2.0)
        p['imp_price'] = st.number_input("Cena importu [€/MWh]", value=150.0)

# ────────────────────────────────────────────────
# Lokální data + spuštění optimalizace
# ────────────────────────────────────────────────
st.divider()
loc_file = st.file_uploader("📂 Lokální data (poptávka tepla, FVE profil, ...)", type=["xlsx"])

if st.session_state.fwd_data is not None and loc_file is not None:
    df_loc = pd.read_excel(loc_file)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    df_loc.rename(columns={df_loc.columns[0]: 'datetime'}, inplace=True)
    df_loc['datetime'] = pd.to_datetime(df_loc['datetime'], dayfirst=True)

    df = pd.merge(st.session_state.fwd_data, df_loc, on='datetime', how='inner').fillna(0)
    T  = len(df)

    # Škálování FVE
    if use_fve and 'fve_installed_p' in p and 'FVE (MW)' in df.columns:
        df['FVE (MW)'] = df['FVE (MW)'] * p['fve_installed_p']

    st.info(f"Načteno **{T}** hodin ({df['datetime'].min().date()} → {df['datetime'].max().date()})")

    if st.button("🏁 Spustit optimalizaci", type="primary"):
        with st.spinner("Probíhá optimalizace (CBC solver)..."):

            model = pulp.LpProblem("KGJ_Dispatch", pulp.LpMaximize)

            # ── Rozhodovací proměnné ──────────────────────────
            # KGJ
            if use_kgj:
                q_kgj  = pulp.LpVariable.dicts("q_KGJ",  range(T), 0, p['k_th'])
                on     = pulp.LpVariable.dicts("on",      range(T), 0, 1, "Binary")
                start  = pulp.LpVariable.dicts("start",   range(T), 0, 1, "Binary")
            else:
                q_kgj = {t: 0 for t in range(T)}
                on    = {t: 0 for t in range(T)}
                start = {t: 0 for t in range(T)}

            # Kotel
            if use_boil:
                q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, p['b_max'])
            else:
                q_boil = {t: 0 for t in range(T)}

            # Elektrokotel
            if use_ek:
                q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, p['ek_max'])
            else:
                q_ek = {t: 0 for t in range(T)}

            # Import tepla
            if use_ext_heat:
                q_imp = pulp.LpVariable.dicts("q_Imp", range(T), 0, p['imp_max'])
            else:
                q_imp = {t: 0 for t in range(T)}

            # TES
            if use_tes:
                tes_soc = pulp.LpVariable.dicts("TES_SOC", range(T + 1), 0, p['tes_cap'])
                tes_in  = pulp.LpVariable.dicts("TES_In",  range(T), 0)
                tes_out = pulp.LpVariable.dicts("TES_Out", range(T), 0)
                model  += tes_soc[0] == p['tes_cap'] * 0.5
            else:
                tes_soc = {t: 0 for t in range(T + 1)}
                tes_in  = {t: 0 for t in range(T)}
                tes_out = {t: 0 for t in range(T)}

            # BESS
            if use_bess:
                bess_soc = pulp.LpVariable.dicts("BESS_SOC", range(T + 1), 0, p['bess_cap'])
                bess_cha = pulp.LpVariable.dicts("BESS_Cha", range(T), 0, p['bess_p'])
                bess_dis = pulp.LpVariable.dicts("BESS_Dis", range(T), 0, p['bess_p'])
                model   += bess_soc[0] == p['bess_cap'] * 0.2
            else:
                bess_soc = {t: 0 for t in range(T + 1)}
                bess_cha = {t: 0 for t in range(T)}
                bess_dis = {t: 0 for t in range(T)}

            # EE síť
            ee_export     = pulp.LpVariable.dicts("ee_export",  range(T), 0)
            ee_import     = pulp.LpVariable.dicts("ee_import",  range(T), 0)
            # Tepelný shortfall (penalizovaný)
            heat_shortfall = pulp.LpVariable.dicts("shortfall",  range(T), 0)

            # ── KGJ omezení ───────────────────────────────────
            if use_kgj:
                for t in range(T):
                    model += q_kgj[t] <= p['k_th'] * on[t]
                    model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]

                # start = 1 pouze pokud on přešel z 0 → 1
                model += start[0] == on[0]   # první hodina
                for t in range(1, T):
                    model += start[t] >= on[t] - on[t - 1]
                    model += start[t] <= on[t]
                    model += start[t] <= 1 - on[t - 1]

                # Min. doba běhu
                min_rt = int(p['k_min_runtime'])
                for t in range(T):
                    for dt in range(1, min_rt):
                        if t + dt < T:
                            model += on[t + dt] >= start[t]

            # ── Časové omezení pro každou hodinu ─────────────
            obj = []

            for t in range(T):
                # Tržní ceny z FWD (upravené)
                p_ee_market  = df['ee_price'].iloc[t]
                p_gas_market = df['gas_price'].iloc[t]

                # Cena plynu pro každou technologii zvlášť
                p_gas_kgj  = p.get('kgj_gas_fix_price',  p_gas_market) if (use_kgj  and p.get('kgj_gas_fix'))  else p_gas_market
                p_gas_boil = p.get('boil_gas_fix_price', p_gas_market) if (use_boil and p.get('boil_gas_fix')) else p_gas_market

                # Cena EE pro každou technologii zvlášť
                p_ee_ek   = p.get('ek_ee_fix_price',   p_ee_market) if (use_ek   and p.get('ek_ee_fix'))   else p_ee_market
                p_ee_bess = p.get('bess_ee_fix_price', p_ee_market) if (use_bess and p.get('bess_ee_fix')) else p_ee_market

                h_dem = df['Poptávka po teple (MW)'].iloc[t]
                fve_p = float(df['FVE (MW)'].iloc[t]) if (use_fve and 'FVE (MW)' in df.columns) else 0.0

                # ── TES dynamika ──────────────────────────────
                if use_tes:
                    model += tes_soc[t + 1] == tes_soc[t] * (1 - p['tes_loss']) + tes_in[t] - tes_out[t]

                # ── BESS dynamika ─────────────────────────────
                if use_bess:
                    model += bess_soc[t + 1] == bess_soc[t] \
                             + bess_cha[t] * p['bess_eff'] \
                             - bess_dis[t] / p['bess_eff']

                # ── Tepelná bilance ───────────────────────────
                # Celková tepelná produkce (přímá)
                heat_direct = q_kgj[t] + q_boil[t] + q_ek[t] + q_imp[t]
                # Teplo dodané spotřebiteli (přímé + z TES - do TES)
                heat_delivered = heat_direct + tes_out[t] - tes_in[t]

                # Pokrytí: dodáno + shortfall >= požadavek
                target = h_dem * p['h_cover']
                model += heat_delivered + heat_shortfall[t] >= target
                # Nedodáváme víc než je poptávka (volitelné – lze uvolnit)
                model += heat_delivered <= h_dem + 1e-3  # malá tolerance

                # ── EE bilance ────────────────────────────────
                ee_kgj_out = q_kgj[t] * (p['k_el'] / p['k_th']) if use_kgj else 0
                ee_ek_in   = q_ek[t] / p.get('ek_eff', 0.98)    if use_ek   else 0

                model += (
                    ee_kgj_out + fve_p + ee_import[t] + bess_dis[t]
                    ==
                    ee_ek_in + bess_cha[t] + ee_export[t]
                )

                # ── Příjmy a náklady ──────────────────────────
                dist_sell = p['dist_ee_sell'] if not p['internal_ee_use'] else 0.0
                dist_buy  = p['dist_ee_buy']  if not p['internal_ee_use'] else 0.0

                revenue = (
                    p['h_price'] * heat_delivered
                    + (p_ee_market - dist_sell) * ee_export[t]
                )

                boil_eff = p.get('boil_eff', 0.95)
                cost_gas_kgj  = (p_gas_kgj  + p['gas_dist']) * (q_kgj[t]  / p['k_eff_th']) if use_kgj  else 0
                cost_gas_boil = (p_gas_boil + p['gas_dist']) * (q_boil[t] / boil_eff)       if use_boil else 0
                cost_ee_import = (p_ee_market + dist_buy) * ee_import[t]
                cost_ee_ek     = (p_ee_ek     + dist_buy) * ee_ek_in          if use_ek   else 0
                cost_imp_heat  = p['imp_price'] * q_imp[t]                    if use_ext_heat else 0
                cost_start     = p['k_start_cost'] * start[t]                 if use_kgj  else 0

                # BESS opotřebení: €/MWh × proteklá energie
                cost_bess = 0
                if use_bess:
                    cost_bess = p['bess_cycle_cost'] * (bess_cha[t] + bess_dis[t])

                # Penalizace za nedodání (vysoká, aby optimizer prioritizoval pokrytí)
                penalty = p['shortfall_penalty'] * heat_shortfall[t]

                obj.append(
                    revenue
                    - cost_gas_kgj - cost_gas_boil
                    - cost_ee_import - cost_ee_ek
                    - cost_imp_heat - cost_start - cost_bess
                    - penalty
                )

            model += pulp.lpSum(obj)

            # ── Řešení ────────────────────────────────────────
            status = model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300))

        st.subheader("📋 Výsledky optimalizace")
        status_str = pulp.LpStatus[status]
        obj_val    = pulp.value(model.objective)
        st.write(f"**Solver status:** {status_str} | **Účelová funkce:** {obj_val:,.0f} €")

        if status not in (1, 2):
            st.error("Optimalizace nenašla přijatelné řešení. Zkontroluj parametry.")
            st.stop()

        # ── Extrakce výsledků ─────────────────────────────
        def val(v, t):
            """Bezpečné získání hodnoty LP proměnné nebo konstanty."""
            x = v[t]
            if isinstance(x, (int, float)):
                return float(x)
            return float(pulp.value(x) or 0)

        boil_eff = p.get('boil_eff', 0.95)

        res = pd.DataFrame({
            'Čas':                      df['datetime'],
            'Poptávka tepla [MW]':      df['Poptávka po teple (MW)'],
            'KGJ [MW_th]':              [val(q_kgj,  t) for t in range(T)],
            'Kotel [MW_th]':            [val(q_boil, t) for t in range(T)],
            'Elektrokotel [MW_th]':     [val(q_ek,   t) for t in range(T)],
            'Import tepla [MW_th]':     [val(q_imp,  t) for t in range(T)],
            'TES příjem [MW_th]':       [val(tes_in,  t) for t in range(T)],
            'TES výdej [MW_th]':        [val(tes_out, t) for t in range(T)],
            'TES SOC [MWh]':            [val(tes_soc, t + 1) for t in range(T)],
            'BESS SOC [MWh]':           [val(bess_soc, t + 1) for t in range(T)],
            'Shortfall [MW]':           [val(heat_shortfall, t) for t in range(T)],
            'EE export [MW]':           [val(ee_export, t) for t in range(T)],
            'EE import [MW]':           [val(ee_import, t) for t in range(T)],
            'EE z KGJ [MW]':            [val(q_kgj, t) * (p['k_el'] / p['k_th']) if use_kgj else 0.0 for t in range(T)],
            'EE z FVE [MW]':            [float(df['FVE (MW)'].iloc[t]) if (use_fve and 'FVE (MW)' in df.columns) else 0.0 for t in range(T)],
            'EE do BESS [MW]':          [val(bess_cha, t) for t in range(T)],
            'EE z BESS [MW]':           [val(bess_dis, t) for t in range(T)],
            'EE do EK [MW]':            [val(q_ek, t) / p.get('ek_eff', 0.98) if use_ek else 0.0 for t in range(T)],
            'Cena EE [€/MWh]':          df['ee_price'].values,
            'Cena plyn [€/MWh]':        df['gas_price'].values,
        })

        res['TES netto [MW_th]'] = res['TES výdej [MW_th]'] - res['TES příjem [MW_th]']
        res['Dodáno tepla [MW]'] = (
            res['KGJ [MW_th]'] + res['Kotel [MW_th]'] + res['Elektrokotel [MW_th]']
            + res['Import tepla [MW_th]'] + res['TES netto [MW_th]']
        )

        # Hodinový zisk (přepočet bez fixní ceny override pro výpis – používáme tržní)
        hourly_profit = []
        for t in range(T):
            p_ee_m  = df['ee_price'].iloc[t]
            p_gas_m = df['gas_price'].iloc[t]

            p_gas_kgj_h  = p.get('kgj_gas_fix_price',  p_gas_m) if (use_kgj  and p.get('kgj_gas_fix'))  else p_gas_m
            p_gas_boil_h = p.get('boil_gas_fix_price', p_gas_m) if (use_boil and p.get('boil_gas_fix')) else p_gas_m
            p_ee_ek_h    = p.get('ek_ee_fix_price',   p_ee_m)  if (use_ek   and p.get('ek_ee_fix'))   else p_ee_m

            rev = (p['h_price'] * res['Dodáno tepla [MW]'].iloc[t]
                   + (p_ee_m - p['dist_ee_sell']) * res['EE export [MW]'].iloc[t])

            c_gas  = ((p_gas_kgj_h  + p['gas_dist']) * (res['KGJ [MW_th]'].iloc[t]  / p['k_eff_th']) if use_kgj  else 0)
            c_gas += ((p_gas_boil_h + p['gas_dist']) * (res['Kotel [MW_th]'].iloc[t] / boil_eff)       if use_boil else 0)
            c_ee   = (p_ee_m  + p['dist_ee_buy'])  * res['EE import [MW]'].iloc[t]
            c_ek   = (p_ee_ek_h + p['dist_ee_buy']) * res['EE do EK [MW]'].iloc[t] if use_ek else 0
            c_imp  = p['imp_price'] * res['Import tepla [MW_th]'].iloc[t] if use_ext_heat else 0
            c_start = p['k_start_cost'] * val(start, t) if use_kgj else 0
            c_bess  = p['bess_cycle_cost'] * (val(bess_cha, t) + val(bess_dis, t)) if use_bess else 0
            pen     = p['shortfall_penalty'] * res['Shortfall [MW]'].iloc[t]

            hourly_profit.append(rev - c_gas - c_ee - c_ek - c_imp - c_start - c_bess - pen)

        res['Hodinový zisk [€]'] = hourly_profit

        # ── Metriky ───────────────────────────────────────
        total_profit    = res['Hodinový zisk [€]'].sum()
        total_shortfall = res['Shortfall [MW]'].sum()
        target_heat     = (res['Poptávka tepla [MW]'] * p['h_cover']).sum()
        coverage        = 100 * (1 - total_shortfall / target_heat) if target_heat > 0 else 100.0

        st.subheader("📊 Klíčové metriky")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Celkový zisk",         f"{total_profit:,.0f} €")
        m2.metric("Shortfall celkem",     f"{total_shortfall:,.1f} MWh")
        m3.metric("Pokrytí poptávky",     f"{coverage:.1f} %")
        m4.metric("Export EE",            f"{res['EE export [MW]'].sum():,.1f} MWh")
        m5.metric("Import EE",            f"{res['EE import [MW]'].sum():,.1f} MWh")

        if total_shortfall > 0.5:
            st.warning(
                f"⚠️ Celkový shortfall {total_shortfall:.1f} MWh – "
                "zvyš penalizaci nebo zkontroluj kapacity zdrojů."
            )

        # ── Graf 1 – Pokrytí tepla ────────────────────────
        st.subheader("🔥 Pokrytí tepelné poptávky")
        fig_heat = go.Figure()

        heat_sources = [
            ('KGJ [MW_th]',           'KGJ',         '#27ae60'),
            ('Kotel [MW_th]',         'Kotel',        '#3498db'),
            ('Elektrokotel [MW_th]',  'Elektrokotel', '#9b59b6'),
            ('Import tepla [MW_th]',  'Import tepla', '#e74c3c'),
            ('TES netto [MW_th]',     'TES netto',    '#f39c12'),
        ]
        for col, name, color in heat_sources:
            fig_heat.add_trace(go.Scatter(
                x=res['Čas'], y=res[col].clip(lower=0),
                name=name, stackgroup='teplo',
                fillcolor=color, line_width=0
            ))

        fig_heat.add_trace(go.Scatter(
            x=res['Čas'], y=res['Shortfall [MW]'],
            name='Nedodáno ⚠️', stackgroup='teplo',
            fillcolor='rgba(200,0,0,0.5)', line_width=0
        ))
        fig_heat.add_trace(go.Scatter(
            x=res['Čas'],
            y=res['Poptávka tepla [MW]'] * p['h_cover'],
            name='Cílová poptávka',
            mode='lines',
            line=dict(color='black', width=2, dash='dot')
        ))
        fig_heat.update_layout(height=500, hovermode='x unified',
                               title="Složení tepelné dodávky v čase")
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Graf 2 – EE bilance ───────────────────────────
        st.subheader("⚡ Bilance elektřiny")
        fig_ee = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.08, row_heights=[0.5, 0.5],
            subplot_titles=("Zdroje EE [MW]", "Spotřeba / export EE [MW]")
        )
        for col, name, color in [
            ('EE z KGJ [MW]',  'KGJ',       '#2ecc71'),
            ('EE z FVE [MW]',  'FVE',       '#f1c40f'),
            ('EE import [MW]', 'Import EE', '#2980b9'),
            ('EE z BESS [MW]', 'BESS výdej','#8e44ad'),
        ]:
            fig_ee.add_trace(go.Scatter(
                x=res['Čas'], y=res[col], name=name,
                stackgroup='vyroba', fillcolor=color
            ), row=1, col=1)

        for col, name, color in [
            ('EE do EK [MW]',   'EK',             '#e74c3c'),
            ('EE do BESS [MW]', 'BESS nabíjení',  '#34495e'),
            ('EE export [MW]',  'Export EE',      '#16a085'),
        ]:
            fig_ee.add_trace(go.Scatter(
                x=res['Čas'], y=-res[col], name=name,
                stackgroup='spotreba', fillcolor=color
            ), row=2, col=1)

        # Cena EE jako linie přes oba grafy
        fig_ee.add_trace(go.Scatter(
            x=res['Čas'], y=res['Cena EE [€/MWh]'],
            name='Cena EE', mode='lines',
            line=dict(color='orange', width=1.5, dash='dot'),
            yaxis='y3'
        ), row=1, col=1)

        fig_ee.update_layout(height=680, hovermode='x unified')
        st.plotly_chart(fig_ee, use_container_width=True)

        # ── Graf 3 – Stavy akumulace ─────────────────────
        st.subheader("🔋 Stavy akumulátorů")
        fig_soc = make_subplots(rows=1, cols=2, subplot_titles=("TES SOC [MWh]", "BESS SOC [MWh]"))
        fig_soc.add_trace(go.Scatter(
            x=res['Čas'], y=res['TES SOC [MWh]'],
            name='TES SOC', line_color='#e67e22'
        ), row=1, col=1)
        if use_tes:
            fig_soc.add_hline(y=p['tes_cap'], line_dash="dot", line_color='#e67e22',
                              annotation_text="Max", row=1, col=1)
        fig_soc.add_trace(go.Scatter(
            x=res['Čas'], y=res['BESS SOC [MWh]'],
            name='BESS SOC', line_color='#3498db'
        ), row=1, col=2)
        if use_bess:
            fig_soc.add_hline(y=p['bess_cap'], line_dash="dot", line_color='#3498db',
                              annotation_text="Max", row=1, col=2)
        fig_soc.update_layout(height=400)
        st.plotly_chart(fig_soc, use_container_width=True)

        # ── Graf 4 – Kumulativní zisk ─────────────────────
        st.subheader("💰 Kumulativní zisk")
        res['Kumulativní zisk [€]'] = res['Hodinový zisk [€]'].cumsum()
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=res['Čas'], y=res['Kumulativní zisk [€]'],
            fill='tozeroy', fillcolor='rgba(39,174,96,0.25)',
            line_color='#27ae60', name='Kum. zisk'
        ))
        fig_cum.update_layout(height=400, title="Průběh kumulativního zisku")
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── Detailní tabulka ─────────────────────────────
        st.subheader("📄 Detail (prvních 48 hodin)")
        st.dataframe(
            res.head(48).round(3),
            use_container_width=True
        )

        # ── Download ─────────────────────────────────────
        csv = res.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
        st.download_button(
            "⬇️ Stáhnout výsledky (CSV)",
            data=csv,
            file_name="kgj_optimalizace.csv",
            mime="text/csv"
        )
