import io
import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────
for key, default in [
    ('fwd_data', None), ('avg_ee_raw', 100.0), ('avg_gas_raw', 50.0),
    ('ee_new', 100.0), ('gas_new', 50.0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Technologie na lokalitě")
    use_kgj      = st.checkbox("Kogenerace (KGJ)",     value=True)
    use_boil     = st.checkbox("Plynový kotel",         value=True)
    use_ek       = st.checkbox("Elektrokotel",          value=True)
    use_tes      = st.checkbox("Nádrž (TES)",           value=True)
    use_bess     = st.checkbox("Baterie (BESS)",        value=True)
    use_fve      = st.checkbox("Fotovoltaika (FVE)",    value=True)
    use_ext_heat = st.checkbox("Nákup tepla (Import)",  value=True)

    st.divider()
    st.header("📈 Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])

    if fwd_file is not None:
        try:
            df_raw = pd.read_excel(fwd_file)
            df_raw.columns = [str(c).strip() for c in df_raw.columns]
            date_col = df_raw.columns[0]
            df_raw[date_col] = pd.to_datetime(df_raw[date_col], dayfirst=True)

            years    = sorted(df_raw[date_col].dt.year.unique())
            sel_year = st.selectbox("Rok pro analýzu", years)
            df_year  = df_raw[df_raw[date_col].dt.year == sel_year].copy()

            avg_ee  = float(df_year.iloc[:, 1].mean())
            avg_gas = float(df_year.iloc[:, 2].mean())
            st.session_state.avg_ee_raw  = avg_ee
            st.session_state.avg_gas_raw = avg_gas

            st.info(f"Průměr EE: **{avg_ee:.1f} €/MWh** | Plyn: **{avg_gas:.1f} €/MWh**")

            ee_new  = st.number_input("Cílová base cena EE [€/MWh]",   value=round(avg_ee,  1), step=1.0)
            gas_new = st.number_input("Cílová base cena Plyn [€/MWh]", value=round(avg_gas, 1), step=1.0)

            df_fwd = df_year.copy()
            df_fwd.columns = ['datetime', 'ee_original', 'gas_original']
            df_fwd['ee_price']  = df_fwd['ee_original']  + (ee_new  - avg_ee)
            df_fwd['gas_price'] = df_fwd['gas_original'] + (gas_new - avg_gas)

            st.session_state.fwd_data = df_fwd
            st.session_state.ee_new   = ee_new
            st.session_state.gas_new  = gas_new
            st.success("FWD načteno ✔")
        except Exception as e:
            st.error(f"Chyba při načítání FWD: {e}")

# ────────────────────────────────────────────────
# FWD GRAFY
# ────────────────────────────────────────────────
if st.session_state.fwd_data is not None:
    df_fwd = st.session_state.fwd_data
    with st.expander("📈 FWD křivka – originál vs. upravená", expanded=True):
        tab_ee, tab_gas, tab_dur = st.tabs(["Elektřina [€/MWh]", "Plyn [€/MWh]", "Trvání cen"])

        with tab_ee:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_fwd['datetime'], y=df_fwd['ee_original'],
                name='EE – originál', line=dict(color='#95a5a6', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df_fwd['datetime'], y=df_fwd['ee_price'],
                name='EE – upravená', line=dict(color='#2ecc71', width=2)))
            fig.add_hline(y=st.session_state.avg_ee_raw, line_dash="dash", line_color="#95a5a6",
                annotation_text=f"Orig. průměr {st.session_state.avg_ee_raw:.1f}")
            fig.add_hline(y=st.session_state.ee_new, line_dash="dash", line_color="#27ae60",
                annotation_text=f"Nový průměr {st.session_state.ee_new:.1f}")
            fig.update_layout(height=350, hovermode='x unified', margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

        with tab_gas:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_fwd['datetime'], y=df_fwd['gas_original'],
                name='Plyn – originál', line=dict(color='#95a5a6', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df_fwd['datetime'], y=df_fwd['gas_price'],
                name='Plyn – upravená', line=dict(color='#e67e22', width=2)))
            fig.add_hline(y=st.session_state.avg_gas_raw, line_dash="dash", line_color="#95a5a6",
                annotation_text=f"Orig. průměr {st.session_state.avg_gas_raw:.1f}")
            fig.add_hline(y=st.session_state.gas_new, line_dash="dash", line_color="#e67e22",
                annotation_text=f"Nový průměr {st.session_state.gas_new:.1f}")
            fig.update_layout(height=350, hovermode='x unified', margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

        with tab_dur:
            # Křivky trvání cen EE a plynu
            ee_sorted  = df_fwd['ee_price'].sort_values(ascending=False).values
            gas_sorted = df_fwd['gas_price'].sort_values(ascending=False).values
            hours      = list(range(1, len(ee_sorted) + 1))
            fig = make_subplots(rows=1, cols=2,
                subplot_titles=("Křivka trvání – EE", "Křivka trvání – Plyn"))
            fig.add_trace(go.Scatter(x=hours, y=ee_sorted, name='EE',
                line=dict(color='#2ecc71', width=2), fill='tozeroy',
                fillcolor='rgba(46,204,113,0.15)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hours, y=gas_sorted, name='Plyn',
                line=dict(color='#e67e22', width=2), fill='tozeroy',
                fillcolor='rgba(230,126,34,0.15)'), row=1, col=2)
            fig.update_xaxes(title_text="Hodiny [h]")
            fig.update_yaxes(title_text="€/MWh")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────
# PARAMETRY
# ────────────────────────────────────────────────
p = {}
t_gen, t_tech = st.tabs(["Obecné", "Technika"])

with t_gen:
    col1, col2 = st.columns(2)
    with col1:
        p['dist_ee_buy']       = st.number_input("Distribuce nákup EE [€/MWh]",   value=33.0)
        p['dist_ee_sell']      = st.number_input("Distribuce prodej EE [€/MWh]",  value=2.0)
        p['gas_dist']          = st.number_input("Distribuce plyn [€/MWh]",        value=5.0)
    with col2:
        p['internal_ee_use']   = st.checkbox("Ušetřit distribuci při interní spotřebě EE", value=True,
            help="Pokud spotřebu EE (EK, BESS) pokrývá lokální výroba (KGJ, FVE), distribuci neplatíme.")
        p['h_price']           = st.number_input("Prodejní cena tepla [€/MWh]",   value=120.0)
        p['h_cover']           = st.slider("Minimální pokrytí poptávky tepla", 0.0, 1.0, 0.99, step=0.01)
        p['shortfall_penalty'] = st.number_input("Penalizace za nedodání tepla [€/MWh]", value=500.0,
            help="Doporučeno 3–5× cena tepla. Vyšší hodnota = silnější priorita pokrytí poptávky.")

with t_tech:
    if use_kgj:
        st.subheader("Kogenerace (KGJ)")
        c1, c2 = st.columns(2)
        with c1:
            p['k_th']          = st.number_input("Jmenovitý tepelný výkon [MW]",  value=1.09)
            p['k_eff_th']      = st.number_input("Tepelná účinnost η_th [-]",      value=0.46,
                help="η_th = Q_th / Q_fuel")
            p['k_eff_el']      = st.number_input("Elektrická účinnost η_el [-]",   value=0.40,
                help="η_el = P_el / Q_fuel. El. výkon = k_th × (η_el / η_th)")
            p['k_min']         = st.slider("Min. zatížení [%]", 0, 100, 55) / 100
        with c2:
            p['k_start_cost']  = st.number_input("Náklady na start [€/start]",    value=1200.0)
            p['k_min_runtime'] = st.number_input("Min. doba běhu [hod]",          value=4, min_value=1)
        k_el_derived = p['k_th'] * (p['k_eff_el'] / p['k_eff_th'])
        p['k_el']    = k_el_derived
        st.caption(f"ℹ️ Odvozený el. výkon KGJ: **{k_el_derived:.3f} MW** | "
                   f"Celková účinnost: **{(p['k_eff_th'] + p['k_eff_el']):.2f}**")
        p['kgj_gas_fix'] = st.checkbox("Fixní cena plynu pro KGJ")
        if p['kgj_gas_fix']:
            p['kgj_gas_fix_price'] = st.number_input("Fixní cena plynu – KGJ [€/MWh]",
                value=float(st.session_state.avg_gas_raw))

    if use_boil:
        st.subheader("Plynový kotel")
        p['b_max']    = st.number_input("Max. výkon [MW]",    value=3.91)
        p['boil_eff'] = st.number_input("Účinnost kotle [-]", value=0.95)
        p['boil_gas_fix'] = st.checkbox("Fixní cena plynu pro kotel")
        if p['boil_gas_fix']:
            p['boil_gas_fix_price'] = st.number_input("Fixní cena plynu – kotel [€/MWh]",
                value=float(st.session_state.avg_gas_raw))

    if use_ek:
        st.subheader("Elektrokotel")
        p['ek_max'] = st.number_input("Max. výkon [MW]",  value=0.61)
        p['ek_eff'] = st.number_input("Účinnost EK [-]",  value=0.98)
        p['ek_ee_fix'] = st.checkbox("Fixní cena EE pro elektrokotel")
        if p['ek_ee_fix']:
            p['ek_ee_fix_price'] = st.number_input("Fixní cena EE – EK [€/MWh]",
                value=float(st.session_state.avg_ee_raw))

    if use_tes:
        st.subheader("Nádrž TES")
        p['tes_cap']  = st.number_input("Kapacita [MWh]", value=10.0)
        p['tes_loss'] = st.number_input("Ztráta [%/h]",   value=0.5) / 100

    if use_bess:
        st.subheader("Baterie BESS")
        c1, c2 = st.columns(2)
        with c1:
            p['bess_cap']        = st.number_input("Kapacita [MWh]",                   value=1.0)
            p['bess_p']          = st.number_input("Max. výkon [MW]",                   value=0.5)
            p['bess_eff']        = st.number_input("Účinnost nabíjení/vybíjení [-]",    value=0.90)
            p['bess_cycle_cost'] = st.number_input("Náklady na opotřebení [€/MWh]",     value=5.0,
                help="Náklad za každou MWh proteklou baterií (nabití + vybití).")
        with c2:
            st.markdown("**Distribuce pro arbitráž**")
            p['bess_dist_buy']  = st.checkbox("Účtovat distribuci NÁKUP do BESS",  value=False,
                help="Zapni pokud BESS nabíjí ze sítě a platíš distribuci za nákup EE.")
            p['bess_dist_sell'] = st.checkbox("Účtovat distribuci PRODEJ z BESS",  value=False,
                help="Zapni pokud BESS prodává do sítě a platíš distribuci za prodej/export EE.")
            st.caption("💡 Interní arbitráž (KGJ/FVE → BESS → EK) distribuci neplatí, "
                       "pokud je zapnuta volba 'Ušetřit distribuci při interní spotřebě'.")
        p['bess_ee_fix'] = st.checkbox("Fixní cena EE pro BESS")
        if p['bess_ee_fix']:
            p['bess_ee_fix_price'] = st.number_input("Fixní cena EE – BESS [€/MWh]",
                value=float(st.session_state.avg_ee_raw))

    if use_fve:
        st.subheader("Fotovoltaika FVE")
        p['fve_installed_p'] = st.number_input("Instalovaný výkon [MW]", value=1.0,
            help="Profil FVE v lokálních datech = capacity factor 0–1. Výsledek = CF × instalovaný výkon.")

    if use_ext_heat:
        st.subheader("Nákup tepla (Import)")
        p['imp_max']   = st.number_input("Max. výkon [MW]",      value=2.0)
        p['imp_price'] = st.number_input("Cena importu [€/MWh]", value=150.0)

# ────────────────────────────────────────────────
# LOKÁLNÍ DATA + OPTIMALIZACE
# ────────────────────────────────────────────────
st.divider()
st.markdown(
    "**Formát lokálních dat:** 1. sloupec = datetime | `Poptávka po teple (MW)` "
    "| `FVE (MW)` jako capacity factor **0–1** (pokud FVE zapnuta)."
)
loc_file = st.file_uploader("📂 Lokální data (poptávka tepla, FVE profil, ...)", type=["xlsx"])

if st.session_state.fwd_data is not None and loc_file is not None:
    df_loc = pd.read_excel(loc_file)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    df_loc.rename(columns={df_loc.columns[0]: 'datetime'}, inplace=True)
    df_loc['datetime'] = pd.to_datetime(df_loc['datetime'], dayfirst=True)

    df = pd.merge(st.session_state.fwd_data, df_loc, on='datetime', how='inner').fillna(0)
    T  = len(df)

    if use_fve and 'fve_installed_p' in p and 'FVE (MW)' in df.columns:
        df['FVE (MW)'] = df['FVE (MW)'].clip(0, 1) * p['fve_installed_p']

    st.info(f"Načteno **{T}** hodin ({df['datetime'].min().date()} → {df['datetime'].max().date()})")

    if st.button("🏁 Spustit optimalizaci", type="primary"):
        with st.spinner("Probíhá optimalizace (CBC solver) …"):

            model = pulp.LpProblem("KGJ_Dispatch", pulp.LpMaximize)

            # ── Proměnné ─────────────────────────────────────
            if use_kgj:
                q_kgj = pulp.LpVariable.dicts("q_KGJ",  range(T), 0, p['k_th'])
                on    = pulp.LpVariable.dicts("on",      range(T), 0, 1, "Binary")
                start = pulp.LpVariable.dicts("start",   range(T), 0, 1, "Binary")
            else:
                q_kgj = on = start = {t: 0 for t in range(T)}

            q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, p['b_max']) \
                     if use_boil else {t: 0 for t in range(T)}
            q_ek   = pulp.LpVariable.dicts("q_EK",   range(T), 0, p['ek_max']) \
                     if use_ek   else {t: 0 for t in range(T)}
            q_imp  = pulp.LpVariable.dicts("q_Imp",  range(T), 0, p['imp_max']) \
                     if use_ext_heat else {t: 0 for t in range(T)}

            if use_tes:
                tes_soc = pulp.LpVariable.dicts("TES_SOC", range(T + 1), 0, p['tes_cap'])
                tes_in  = pulp.LpVariable.dicts("TES_In",  range(T), 0)
                tes_out = pulp.LpVariable.dicts("TES_Out", range(T), 0)
                model  += tes_soc[0] == p['tes_cap'] * 0.5
            else:
                tes_soc = {t: 0 for t in range(T + 1)}
                tes_in = tes_out = {t: 0 for t in range(T)}

            if use_bess:
                bess_soc = pulp.LpVariable.dicts("BESS_SOC", range(T + 1), 0, p['bess_cap'])
                bess_cha = pulp.LpVariable.dicts("BESS_Cha", range(T), 0, p['bess_p'])
                bess_dis = pulp.LpVariable.dicts("BESS_Dis", range(T), 0, p['bess_p'])
                model   += bess_soc[0] == p['bess_cap'] * 0.2
            else:
                bess_soc = {t: 0 for t in range(T + 1)}
                bess_cha = bess_dis = {t: 0 for t in range(T)}

            ee_export      = pulp.LpVariable.dicts("ee_export",  range(T), 0)
            ee_import      = pulp.LpVariable.dicts("ee_import",  range(T), 0)
            heat_shortfall = pulp.LpVariable.dicts("shortfall",  range(T), 0)

            # ── KGJ omezení ───────────────────────────────────
            if use_kgj:
                for t in range(T):
                    model += q_kgj[t] <= p['k_th'] * on[t]
                    model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]
                model += start[0] == on[0]
                for t in range(1, T):
                    model += start[t] >= on[t] - on[t - 1]
                    model += start[t] <= on[t]
                    model += start[t] <= 1 - on[t - 1]
                min_rt = int(p['k_min_runtime'])
                for t in range(T):
                    for dt in range(1, min_rt):
                        if t + dt < T:
                            model += on[t + dt] >= start[t]

            # ── Hlavní smyčka ─────────────────────────────────
            obj      = []
            boil_eff = p.get('boil_eff', 0.95)
            ek_eff   = p.get('ek_eff',   0.98)

            for t in range(T):
                p_ee_m  = df['ee_price'].iloc[t]
                p_gas_m = df['gas_price'].iloc[t]

                p_gas_kgj  = p.get('kgj_gas_fix_price',  p_gas_m) if (use_kgj  and p.get('kgj_gas_fix'))  else p_gas_m
                p_gas_boil = p.get('boil_gas_fix_price', p_gas_m) if (use_boil and p.get('boil_gas_fix')) else p_gas_m
                p_ee_ek    = p.get('ek_ee_fix_price',    p_ee_m)  if (use_ek   and p.get('ek_ee_fix'))   else p_ee_m

                h_dem = df['Poptávka po teple (MW)'].iloc[t]
                fve_p = float(df['FVE (MW)'].iloc[t]) if (use_fve and 'FVE (MW)' in df.columns) else 0.0

                if use_tes:
                    model += tes_soc[t + 1] == tes_soc[t] * (1 - p['tes_loss']) + tes_in[t] - tes_out[t]
                if use_bess:
                    model += bess_soc[t + 1] == (
                        bess_soc[t] + bess_cha[t] * p['bess_eff'] - bess_dis[t] / p['bess_eff']
                    )

                heat_delivered = q_kgj[t] + q_boil[t] + q_ek[t] + q_imp[t] + tes_out[t] - tes_in[t]
                model += heat_delivered + heat_shortfall[t] >= h_dem * p['h_cover']
                model += heat_delivered <= h_dem + 1e-3

                ee_kgj_out = q_kgj[t] * (p['k_eff_el'] / p['k_eff_th']) if use_kgj else 0
                ee_ek_in   = q_ek[t] / ek_eff                            if use_ek  else 0
                model += ee_kgj_out + fve_p + ee_import[t] + bess_dis[t] == ee_ek_in + bess_cha[t] + ee_export[t]

                dist_sell_net = p['dist_ee_sell'] if not p['internal_ee_use'] else 0.0
                dist_buy_net  = p['dist_ee_buy']  if not p['internal_ee_use'] else 0.0

                bess_dist_buy_cost  = p['dist_ee_buy']  * bess_cha[t] if (use_bess and p.get('bess_dist_buy'))  else 0
                bess_dist_sell_cost = p['dist_ee_sell'] * bess_dis[t] if (use_bess and p.get('bess_dist_sell')) else 0

                revenue = (
                    p['h_price'] * heat_delivered
                    + (p_ee_m - dist_sell_net) * ee_export[t]
                )
                costs = (
                    ((p_gas_kgj  + p['gas_dist']) * (q_kgj[t]  / p['k_eff_th']) if use_kgj      else 0) +
                    ((p_gas_boil + p['gas_dist']) * (q_boil[t] / boil_eff)       if use_boil     else 0) +
                    (p_ee_m + dist_buy_net)  * ee_import[t] +
                    ((p_ee_ek + dist_buy_net) * ee_ek_in                          if use_ek       else 0) +
                    (p['imp_price'] * q_imp[t]                                    if use_ext_heat else 0) +
                    (p['k_start_cost'] * start[t]                                 if use_kgj      else 0) +
                    (p['bess_cycle_cost'] * (bess_cha[t] + bess_dis[t])           if use_bess     else 0) +
                    bess_dist_buy_cost + bess_dist_sell_cost +
                    p['shortfall_penalty'] * heat_shortfall[t]
                )
                obj.append(revenue - costs)

            model += pulp.lpSum(obj)
            status = model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300))

        status_str = pulp.LpStatus[status]
        obj_val    = pulp.value(model.objective)
        st.subheader("📋 Výsledky optimalizace")
        st.write(f"**Solver status:** {status_str} | **Účelová funkce:** {obj_val:,.0f} €")

        if status not in (1, 2):
            st.error("Optimalizace nenašla přijatelné řešení. Zkontroluj parametry.")
            st.stop()

        # ── Extrakce výsledků ─────────────────────────────
        def val(v, t):
            x = v[t]
            return float(x) if isinstance(x, (int, float)) else float(pulp.value(x) or 0)

        boil_eff = p.get('boil_eff', 0.95)
        ek_eff   = p.get('ek_eff',   0.98)

        res = pd.DataFrame({
            'Čas':                    df['datetime'],
            'Poptávka tepla [MW]':    df['Poptávka po teple (MW)'],
            'KGJ [MW_th]':            [val(q_kgj,  t) for t in range(T)],
            'Kotel [MW_th]':          [val(q_boil, t) for t in range(T)],
            'Elektrokotel [MW_th]':   [val(q_ek,   t) for t in range(T)],
            'Import tepla [MW_th]':   [val(q_imp,  t) for t in range(T)],
            'TES příjem [MW_th]':     [val(tes_in,  t) for t in range(T)],
            'TES výdej [MW_th]':      [val(tes_out, t) for t in range(T)],
            'TES SOC [MWh]':          [val(tes_soc, t + 1) for t in range(T)],
            'BESS nabíjení [MW]':     [val(bess_cha, t) for t in range(T)],
            'BESS vybíjení [MW]':     [val(bess_dis, t) for t in range(T)],
            'BESS SOC [MWh]':         [val(bess_soc, t + 1) for t in range(T)],
            'Shortfall [MW]':         [val(heat_shortfall, t) for t in range(T)],
            'EE export [MW]':         [val(ee_export, t) for t in range(T)],
            'EE import [MW]':         [val(ee_import, t) for t in range(T)],
            'EE z KGJ [MW]':          [val(q_kgj, t) * (p['k_eff_el'] / p['k_eff_th']) if use_kgj else 0.0 for t in range(T)],
            'EE z FVE [MW]':          [float(df['FVE (MW)'].iloc[t]) if (use_fve and 'FVE (MW)' in df.columns) else 0.0 for t in range(T)],
            'EE do EK [MW]':          [val(q_ek, t) / ek_eff if use_ek else 0.0 for t in range(T)],
            'Cena EE [€/MWh]':       df['ee_price'].values,
            'Cena plyn [€/MWh]':     df['gas_price'].values,
        })
        res['TES netto [MW_th]'] = res['TES výdej [MW_th]'] - res['TES příjem [MW_th]']
        res['Dodáno tepla [MW]'] = (
            res['KGJ [MW_th]'] + res['Kotel [MW_th]'] + res['Elektrokotel [MW_th]']
            + res['Import tepla [MW_th]'] + res['TES netto [MW_th]']
        )
        res['Měsíc'] = pd.to_datetime(res['Čas']).dt.month
        res['Hodina dne'] = pd.to_datetime(res['Čas']).dt.hour

        # ── Hodinový zisk ─────────────────────────────────
        hourly_profit = []
        for t in range(T):
            p_ee_m   = df['ee_price'].iloc[t]
            p_gas_m  = df['gas_price'].iloc[t]
            p_gas_kj = p.get('kgj_gas_fix_price',  p_gas_m) if (use_kgj  and p.get('kgj_gas_fix'))  else p_gas_m
            p_gas_bh = p.get('boil_gas_fix_price', p_gas_m) if (use_boil and p.get('boil_gas_fix')) else p_gas_m
            p_ee_ekh = p.get('ek_ee_fix_price',    p_ee_m)  if (use_ek   and p.get('ek_ee_fix'))   else p_ee_m

            rev  = (p['h_price'] * res['Dodáno tepla [MW]'].iloc[t]
                    + (p_ee_m - p['dist_ee_sell']) * res['EE export [MW]'].iloc[t])
            c_gas  = ((p_gas_kj + p['gas_dist']) * (res['KGJ [MW_th]'].iloc[t]  / p['k_eff_th']) if use_kgj  else 0)
            c_gas += ((p_gas_bh + p['gas_dist']) * (res['Kotel [MW_th]'].iloc[t] / boil_eff)      if use_boil else 0)
            c_ee   = (p_ee_m  + p['dist_ee_buy'])  * res['EE import [MW]'].iloc[t]
            c_ek   = (p_ee_ekh + p['dist_ee_buy']) * res['EE do EK [MW]'].iloc[t] if use_ek else 0
            c_imp  = p['imp_price'] * res['Import tepla [MW_th]'].iloc[t]           if use_ext_heat else 0
            c_st   = p['k_start_cost'] * val(start, t)                              if use_kgj  else 0
            c_bw   = p['bess_cycle_cost'] * (res['BESS nabíjení [MW]'].iloc[t] + res['BESS vybíjení [MW]'].iloc[t]) if use_bess else 0
            c_bd   = (p['dist_ee_buy']  * res['BESS nabíjení [MW]'].iloc[t] if (use_bess and p.get('bess_dist_buy'))  else 0) \
                   + (p['dist_ee_sell'] * res['BESS vybíjení [MW]'].iloc[t] if (use_bess and p.get('bess_dist_sell')) else 0)
            pen    = p['shortfall_penalty'] * res['Shortfall [MW]'].iloc[t]

            hourly_profit.append(rev - c_gas - c_ee - c_ek - c_imp - c_st - c_bw - c_bd - pen)

        res['Hodinový zisk [€]']    = hourly_profit
        res['Kumulativní zisk [€]'] = res['Hodinový zisk [€]'].cumsum()

        # ── Metriky ───────────────────────────────────────
        total_profit    = res['Hodinový zisk [€]'].sum()
        total_shortfall = res['Shortfall [MW]'].sum()
        target_heat     = (res['Poptávka tepla [MW]'] * p['h_cover']).sum()
        coverage        = 100 * (1 - total_shortfall / target_heat) if target_heat > 0 else 100.0
        total_ee_gen    = res['EE z KGJ [MW]'].sum() + res['EE z FVE [MW]'].sum()
        kgj_hours       = sum(1 for t in range(T) if val(on, t) > 0.5) if use_kgj else 0

        st.subheader("📊 Klíčové metriky")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Celkový zisk",         f"{total_profit:,.0f} €")
        m2.metric("Shortfall celkem",     f"{total_shortfall:,.1f} MWh")
        m3.metric("Pokrytí poptávky",     f"{coverage:.1f} %")
        m4.metric("Export EE",            f"{res['EE export [MW]'].sum():,.1f} MWh")
        m5.metric("Výroba EE (KGJ+FVE)", f"{total_ee_gen:,.1f} MWh")
        m6.metric("Provozní hodiny KGJ",  f"{kgj_hours:,} h")

        if total_shortfall > 0.5:
            st.warning(f"⚠️ Celkový shortfall {total_shortfall:.1f} MWh – zvyš penalizaci nebo kapacity zdrojů.")

        # ════════════════════════════════════════════════
        # GRAFY
        # ════════════════════════════════════════════════

        # ── Graf 1 – Pokrytí tepla ────────────────────────
        st.subheader("🔥 Pokrytí tepelné poptávky")
        fig = go.Figure()
        for col, name, color in [
            ('KGJ [MW_th]',          'KGJ',          '#27ae60'),
            ('Kotel [MW_th]',        'Kotel',         '#3498db'),
            ('Elektrokotel [MW_th]', 'Elektrokotel',  '#9b59b6'),
            ('Import tepla [MW_th]', 'Import tepla',  '#e74c3c'),
            ('TES netto [MW_th]',    'TES netto',     '#f39c12'),
        ]:
            fig.add_trace(go.Scatter(x=res['Čas'], y=res[col].clip(lower=0),
                name=name, stackgroup='teplo', fillcolor=color, line_width=0))
        fig.add_trace(go.Scatter(x=res['Čas'], y=res['Shortfall [MW]'],
            name='Nedodáno ⚠️', stackgroup='teplo', fillcolor='rgba(200,0,0,0.45)', line_width=0))
        fig.add_trace(go.Scatter(x=res['Čas'], y=res['Poptávka tepla [MW]'] * p['h_cover'],
            name='Cílová poptávka', mode='lines', line=dict(color='black', width=2, dash='dot')))
        fig.update_layout(height=480, hovermode='x unified', title="Složení tepelné dodávky v čase")
        st.plotly_chart(fig, use_container_width=True)

        # ── Graf 2 – EE bilance ───────────────────────────
        st.subheader("⚡ Bilance elektřiny")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.08, row_heights=[0.5, 0.5],
            subplot_titles=("Zdroje EE [MW]", "Spotřeba / export EE [MW]"))
        for col, name, color in [
            ('EE z KGJ [MW]',      'KGJ',         '#2ecc71'),
            ('EE z FVE [MW]',      'FVE',          '#f1c40f'),
            ('EE import [MW]',     'Import EE',    '#2980b9'),
            ('BESS vybíjení [MW]', 'BESS výdej',   '#8e44ad'),
        ]:
            fig.add_trace(go.Scatter(x=res['Čas'], y=res[col], name=name,
                stackgroup='vyroba', fillcolor=color), row=1, col=1)
        for col, name, color in [
            ('EE do EK [MW]',       'EK',             '#e74c3c'),
            ('BESS nabíjení [MW]',  'BESS nabíjení',  '#34495e'),
            ('EE export [MW]',      'Export EE',      '#16a085'),
        ]:
            fig.add_trace(go.Scatter(x=res['Čas'], y=-res[col], name=name,
                stackgroup='spotreba', fillcolor=color), row=2, col=1)
        fig.update_layout(height=650, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # ── Graf 3 – Stavy akumulace ─────────────────────
        st.subheader("🔋 Stavy akumulátorů")
        fig = make_subplots(rows=1, cols=2, subplot_titles=("TES SOC [MWh]", "BESS SOC [MWh]"))
        fig.add_trace(go.Scatter(x=res['Čas'], y=res['TES SOC [MWh]'],
            name='TES', line_color='#e67e22'), row=1, col=1)
        if use_tes:
            fig.add_hline(y=p['tes_cap'], line_dash="dot", line_color='#e67e22',
                annotation_text="Max", row=1, col=1)
        fig.add_trace(go.Scatter(x=res['Čas'], y=res['BESS SOC [MWh]'],
            name='BESS', line_color='#3498db'), row=1, col=2)
        if use_bess:
            fig.add_hline(y=p['bess_cap'], line_dash="dot", line_color='#3498db',
                annotation_text="Max", row=1, col=2)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        # ── Graf 4 – Kumulativní zisk ─────────────────────
        st.subheader("💰 Kumulativní zisk")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['Čas'], y=res['Kumulativní zisk [€]'],
            fill='tozeroy', fillcolor='rgba(39,174,96,0.2)',
            line_color='#27ae60', name='Kum. zisk'))
        fig.update_layout(height=380, title="Průběh kumulativního zisku v čase")
        st.plotly_chart(fig, use_container_width=True)

        # ── Graf 5 – Měsíční analýza ──────────────────────
        st.subheader("📅 Měsíční analýza")
        month_names = {1:'Led',2:'Úno',3:'Bře',4:'Dub',5:'Kvě',6:'Čvn',
                       7:'Čvc',8:'Srp',9:'Zář',10:'Říj',11:'Lis',12:'Pro'}

        monthly = res.groupby('Měsíc').agg(
            zisk=('Hodinový zisk [€]', 'sum'),
            teplo_kgj=('KGJ [MW_th]', 'sum'),
            teplo_kotel=('Kotel [MW_th]', 'sum'),
            teplo_ek=('Elektrokotel [MW_th]', 'sum'),
            ee_export=('EE export [MW]', 'sum'),
            ee_import=('EE import [MW]', 'sum'),
            shortfall=('Shortfall [MW]', 'sum'),
        ).reset_index()
        monthly['Měsíc_str'] = monthly['Měsíc'].map(month_names)

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=("Měsíční zisk [€]", "Měsíční mix tepelných zdrojů [MWh]"))
        bar_colors = ['#e74c3c' if z < 0 else '#27ae60' for z in monthly['zisk']]
        fig.add_trace(go.Bar(x=monthly['Měsíc_str'], y=monthly['zisk'],
            marker_color=bar_colors, name='Zisk'), row=1, col=1)
        for col, name, color in [
            ('teplo_kgj',   'KGJ',         '#27ae60'),
            ('teplo_kotel', 'Kotel',        '#3498db'),
            ('teplo_ek',    'Elektrokotel', '#9b59b6'),
        ]:
            fig.add_trace(go.Bar(x=monthly['Měsíc_str'], y=monthly[col],
                name=name, marker_color=color), row=1, col=2)
        fig.update_layout(height=400, barmode='stack', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # ── Graf 6 – Průměrný denní profil ───────────────
        st.subheader("🕐 Průměrný denní profil (všechny dny)")
        hourly_avg = res.groupby('Hodina dne').agg(
            teplo_popt=('Poptávka tepla [MW]', 'mean'),
            teplo_kgj=('KGJ [MW_th]', 'mean'),
            teplo_kotel=('Kotel [MW_th]', 'mean'),
            teplo_ek=('Elektrokotel [MW_th]', 'mean'),
            ee_kgj=('EE z KGJ [MW]', 'mean'),
            ee_fve=('EE z FVE [MW]', 'mean'),
            ee_export=('EE export [MW]', 'mean'),
            ee_import=('EE import [MW]', 'mean'),
            cena_ee=('Cena EE [€/MWh]', 'mean'),
        ).reset_index()
        hours_x = hourly_avg['Hodina dne']

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.5, 0.5], vertical_spacing=0.08,
            subplot_titles=("Průměrná tepelná produkce [MW]", "Průměrná EE bilance [MW]"))

        for col, name, color in [
            ('teplo_kgj',   'KGJ',         '#27ae60'),
            ('teplo_kotel', 'Kotel',        '#3498db'),
            ('teplo_ek',    'Elektrokotel', '#9b59b6'),
        ]:
            fig.add_trace(go.Bar(x=hours_x, y=hourly_avg[col], name=name, marker_color=color), row=1, col=1)
        fig.add_trace(go.Scatter(x=hours_x, y=hourly_avg['teplo_popt'],
            name='Poptávka', mode='lines', line=dict(color='black', width=2, dash='dot')), row=1, col=1)

        for col, name, color in [
            ('ee_kgj',   'KGJ',    '#2ecc71'),
            ('ee_fve',   'FVE',    '#f1c40f'),
            ('ee_import','Import', '#2980b9'),
        ]:
            fig.add_trace(go.Bar(x=hours_x, y=hourly_avg[col], name=name, marker_color=color), row=2, col=1)
        fig.add_trace(go.Scatter(x=hours_x, y=hourly_avg['cena_ee'],
            name='Cena EE', mode='lines', line=dict(color='orange', width=2, dash='dot'),
            yaxis='y4'), row=2, col=1)

        fig.update_layout(height=600, barmode='stack', hovermode='x unified',
            xaxis2=dict(title='Hodina dne'))
        st.plotly_chart(fig, use_container_width=True)

        # ── Graf 7 – Heatmapa zisku ───────────────────────
        st.subheader("🗓️ Heatmapa hodinového zisku")
        res_hm = res.copy()
        res_hm['Den']       = pd.to_datetime(res_hm['Čas']).dt.dayofyear
        res_hm['Hodina']    = pd.to_datetime(res_hm['Čas']).dt.hour
        pivot_profit = res_hm.pivot_table(index='Hodina', columns='Den',
            values='Hodinový zisk [€]', aggfunc='sum')
        fig = go.Figure(go.Heatmap(
            z=pivot_profit.values,
            x=pivot_profit.columns,
            y=pivot_profit.index,
            colorscale='RdYlGn',
            colorbar=dict(title='€/hod'),
            zmid=0,
        ))
        fig.update_layout(
            height=420,
            title="Hodinový zisk – den vs. hodina (zelená = zisk, červená = ztráta)",
            xaxis_title="Den v roce",
            yaxis_title="Hodina dne",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Graf 8 – Scatter EE cena vs. provoz KGJ ──────
        if use_kgj:
            st.subheader("🔍 Citlivost KGJ na cenu EE a plynu")
            res['KGJ_on'] = [val(on, t) for t in range(T)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=res['Cena EE [€/MWh]'],
                y=res['Cena plyn [€/MWh]'],
                mode='markers',
                marker=dict(
                    color=res['KGJ_on'],
                    colorscale=[[0, '#e74c3c'], [1, '#27ae60']],
                    size=4, opacity=0.6,
                    colorbar=dict(title='KGJ on/off', tickvals=[0, 1], ticktext=['Off', 'On']),
                ),
                text=[f"EE: {e:.1f} | Plyn: {g:.1f} | {'ON' if o > 0.5 else 'OFF'}"
                      for e, g, o in zip(res['Cena EE [€/MWh]'], res['Cena plyn [€/MWh]'], res['KGJ_on'])],
                hovertemplate='%{text}<extra></extra>',
                name='Hodiny',
            ))
            fig.update_layout(
                height=450,
                xaxis_title='Cena EE [€/MWh]',
                yaxis_title='Cena plynu [€/MWh]',
                title='Provoz KGJ v závislosti na cenách EE a plynu (zelená = KGJ běží)',
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Graf 9 – Složení příjmů a nákladů (waterfall) ─
        st.subheader("💵 Rozpad zisku – příjmy a náklady")
        rev_teplo = p['h_price'] * res['Dodáno tepla [MW]'].sum()
        rev_ee    = (res['Cena EE [€/MWh]'] * res['EE export [MW]']).sum()
        c_gas_kgj = sum(
            (p.get('kgj_gas_fix_price', df['gas_price'].iloc[t]) + p['gas_dist'])
            * res['KGJ [MW_th]'].iloc[t] / p['k_eff_th']
            for t in range(T)
        ) if use_kgj else 0
        c_gas_boil = sum(
            (p.get('boil_gas_fix_price', df['gas_price'].iloc[t]) + p['gas_dist'])
            * res['Kotel [MW_th]'].iloc[t] / boil_eff
            for t in range(T)
        ) if use_boil else 0
        c_ee_imp   = ((res['Cena EE [€/MWh]'] + p['dist_ee_buy']) * res['EE import [MW]']).sum()
        c_imp_heat = p['imp_price'] * res['Import tepla [MW_th]'].sum() if use_ext_heat else 0
        c_starts   = p['k_start_cost'] * sum(val(start, t) for t in range(T)) if use_kgj else 0
        c_penalty  = p['shortfall_penalty'] * res['Shortfall [MW]'].sum()

        wf_labels  = ['Příjmy: teplo', 'Příjmy: EE export',
                      'Náklady: plyn KGJ', 'Náklady: plyn kotel', 'Náklady: import EE',
                      'Náklady: import tepla', 'Náklady: starty KGJ', 'Penalizace shortfall',
                      'Celkový zisk']
        wf_values  = [rev_teplo, rev_ee,
                      -c_gas_kgj, -c_gas_boil, -c_ee_imp,
                      -c_imp_heat, -c_starts, -c_penalty,
                      total_profit]
        wf_measure = ['relative'] * (len(wf_values) - 1) + ['total']
        wf_colors  = ['#27ae60' if v >= 0 else '#e74c3c' for v in wf_values[:-1]] + ['#2980b9']

        fig = go.Figure(go.Waterfall(
            orientation='v',
            measure=wf_measure,
            x=wf_labels,
            y=wf_values,
            connector=dict(line=dict(color='#bdc3c7', width=1)),
            decreasing=dict(marker_color='#e74c3c'),
            increasing=dict(marker_color='#27ae60'),
            totals=dict(marker_color='#2980b9'),
            text=[f"{v:,.0f} €" for v in wf_values],
            textposition='outside',
        ))
        fig.update_layout(height=480, title="Waterfall – rozpad příjmů a nákladů za celé období")
        st.plotly_chart(fig, use_container_width=True)

        # ════════════════════════════════════════════════
        # EXCEL EXPORT
        # ════════════════════════════════════════════════
        st.subheader("⬇️ Export výsledků")

        def to_excel(df_out: pd.DataFrame) -> bytes:
            buf = io.BytesIO()
            export_cols = [c for c in df_out.columns if c not in ('Měsíc', 'Hodina dne', 'KGJ_on')]
            df_exp = df_out[export_cols].copy()

            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                # List 1 – hodinová data
                df_exp.to_excel(writer, index=False, sheet_name='Hodinová data')
                wb = writer.book
                ws = writer.sheets['Hodinová data']

                fmt_hdr  = wb.add_format({'bold': True, 'bg_color': '#2c3e50', 'font_color': 'white',
                                           'border': 1, 'align': 'center', 'text_wrap': True})
                fmt_num2 = wb.add_format({'num_format': '#,##0.00', 'border': 1})
                fmt_num0 = wb.add_format({'num_format': '#,##0',    'border': 1})
                fmt_date = wb.add_format({'num_format': 'dd.mm.yyyy hh:mm', 'border': 1})
                money_c  = {'Hodinový zisk [€]', 'Kumulativní zisk [€]'}

                for ci, cn in enumerate(df_exp.columns):
                    ws.set_column(ci, ci, 20)
                    ws.write(0, ci, cn, fmt_hdr)
                for ri in range(len(df_exp)):
                    for ci, cn in enumerate(df_exp.columns):
                        cv = df_exp.iloc[ri, ci]
                        if cn == 'Čas':
                            ws.write_datetime(ri + 1, ci, pd.Timestamp(cv).to_pydatetime(), fmt_date)
                        elif cn in money_c:
                            ws.write_number(ri + 1, ci, float(cv), fmt_num0)
                        else:
                            ws.write_number(ri + 1, ci, float(cv), fmt_num2)
                ws.autofilter(0, 0, len(df_exp), len(df_exp.columns) - 1)
                ws.freeze_panes(1, 1)
                ws.set_row(0, 36)

                # List 2 – měsíční souhrn
                monthly_exp = monthly.copy()
                monthly_exp['Měsíc_str'] = monthly_exp['Měsíc'].map(month_names)
                monthly_exp = monthly_exp[['Měsíc_str', 'zisk', 'teplo_kgj',
                                           'teplo_kotel', 'teplo_ek', 'ee_export',
                                           'ee_import', 'shortfall']]
                monthly_exp.columns = ['Měsíc', 'Zisk [€]', 'KGJ teplo [MWh]',
                                       'Kotel teplo [MWh]', 'EK teplo [MWh]',
                                       'EE export [MWh]', 'EE import [MWh]', 'Shortfall [MWh]']
                monthly_exp.to_excel(writer, index=False, sheet_name='Měsíční souhrn')
                ws2 = writer.sheets['Měsíční souhrn']
                for ci, cn in enumerate(monthly_exp.columns):
                    ws2.set_column(ci, ci, 18)
                    ws2.write(0, ci, cn, fmt_hdr)
                ws2.set_row(0, 30)

                # List 3 – parametry (pro reprodukovatelnost)
                params_data = [
                    ('Penalizace shortfall [€/MWh]', p['shortfall_penalty']),
                    ('Cena tepla [€/MWh]',           p['h_price']),
                    ('Min. pokrytí [-]',              p['h_cover']),
                    ('Distribuce nákup EE [€/MWh]',  p['dist_ee_buy']),
                    ('Distribuce prodej EE [€/MWh]',  p['dist_ee_sell']),
                    ('Distribuce plyn [€/MWh]',       p['gas_dist']),
                ]
                if use_kgj:
                    params_data += [
                        ('KGJ k_th [MW]',     p['k_th']),
                        ('KGJ η_th [-]',      p['k_eff_th']),
                        ('KGJ η_el [-]',      p['k_eff_el']),
                        ('KGJ min zatížení',  p['k_min']),
                        ('KGJ start cost [€]',p['k_start_cost']),
                    ]
                pd.DataFrame(params_data, columns=['Parametr', 'Hodnota']).to_excel(
                    writer, index=False, sheet_name='Parametry')
                ws3 = writer.sheets['Parametry']
                ws3.set_column(0, 0, 30)
                ws3.set_column(1, 1, 15)

            return buf.getvalue()

        xlsx_bytes = to_excel(res.round(4))
        st.download_button(
            label="📥 Stáhnout výsledky (Excel .xlsx)",
            data=xlsx_bytes,
            file_name="kgj_optimalizace.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
