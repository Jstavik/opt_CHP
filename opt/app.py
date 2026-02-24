import io
import streamlit as st
import pandas as pd
import numpy as np
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

# ── Session state ────────────────────────────────
for key, default in [
    ('fwd_data', None), ('avg_ee_raw', 100.0), ('avg_gas_raw', 50.0),
    ('ee_new', 100.0), ('gas_new', 50.0), ('results', None), ('df_main', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

MONTH_NAMES = {1:'Led',2:'Úno',3:'Bře',4:'Dub',5:'Kvě',6:'Čvn',
               7:'Čvc',8:'Srp',9:'Zář',10:'Říj',11:'Lis',12:'Pro'}

# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Technologie na lokalitě")
    use_kgj      = st.checkbox("Kogenerace (KGJ)",    value=True)
    use_boil     = st.checkbox("Plynový kotel",        value=True)
    use_ek       = st.checkbox("Elektrokotel",         value=True)
    use_tes      = st.checkbox("Nádrž (TES)",          value=True)
    use_bess     = st.checkbox("Baterie (BESS)",       value=True)
    use_fve      = st.checkbox("Fotovoltaika (FVE)",   value=True)
    use_ext_heat = st.checkbox("Nákup tepla (Import)", value=True)

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
        tab_ee, tab_gas, tab_dur = st.tabs(["Elektřina [€/MWh]", "Plyn [€/MWh]", "Křivky trvání"])
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
            fig.update_layout(height=340, hovermode='x unified', margin=dict(t=30))
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
            fig.update_layout(height=340, hovermode='x unified', margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)
        with tab_dur:
            ee_s  = df_fwd['ee_price'].sort_values(ascending=False).values
            gas_s = df_fwd['gas_price'].sort_values(ascending=False).values
            hrs   = list(range(1, len(ee_s)+1))
            fig = make_subplots(rows=1, cols=2,
                subplot_titles=("Křivka trvání – EE", "Křivka trvání – Plyn"))
            fig.add_trace(go.Scatter(x=hrs, y=ee_s, name='EE',
                line=dict(color='#2ecc71', width=2), fill='tozeroy',
                fillcolor='rgba(46,204,113,0.15)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hrs, y=gas_s, name='Plyn',
                line=dict(color='#e67e22', width=2), fill='tozeroy',
                fillcolor='rgba(230,126,34,0.15)'), row=1, col=2)
            fig.update_xaxes(title_text="Hodiny [h]")
            fig.update_yaxes(title_text="€/MWh")
            fig.update_layout(height=340, showlegend=False)
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
        p['internal_ee_use']   = st.checkbox(
            "Ušetřit distribuci při interní spotřebě EE", value=True,
            help="Pokud spotřebu EE (EK, BESS) pokrývá lokální výroba (KGJ, FVE), distribuci neplatíme.")
        p['h_price']           = st.number_input("Prodejní cena tepla [€/MWh]",   value=120.0)
        p['h_cover']           = st.slider("Minimální pokrytí poptávky tepla", 0.0, 1.0, 0.99, step=0.01)
        p['shortfall_penalty'] = st.number_input("Penalizace za nedodání tepla [€/MWh]", value=500.0,
            help="Doporučeno 3–5× cena tepla. Vyšší = silnější priorita pokrytí poptávky.")

with t_tech:
    # ── KGJ ──────────────────────────────────────
    if use_kgj:
        st.subheader("Kogenerace (KGJ)")
        c1, c2 = st.columns(2)
        with c1:
            p['k_th']          = st.number_input("Jmenovitý tepelný výkon [MW]",  value=1.09)
            p['k_eff_th']      = st.number_input("Tepelná účinnost η_th [-]",      value=0.46)
            p['k_eff_el']      = st.number_input("Elektrická účinnost η_el [-]",   value=0.40)
            p['k_min']         = st.slider("Min. zatížení [%]", 0, 100, 55) / 100
        with c2:
            p['k_start_cost']  = st.number_input("Náklady na start [€/start]",    value=1200.0)
            p['k_min_runtime'] = st.number_input("Min. doba běhu [hod]",          value=4, min_value=1)
        k_el_derived = p['k_th'] * (p['k_eff_el'] / p['k_eff_th'])
        p['k_el'] = k_el_derived
        st.caption(f"ℹ️ Odvozený el. výkon: **{k_el_derived:.3f} MW** | "
                   f"Celková účinnost: **{p['k_eff_th']+p['k_eff_el']:.2f}**")
        # Roční limit hodin
        p['kgj_hour_limit_on'] = st.checkbox("Omezit max. počet provozních hodin KGJ / rok", value=False)
        if p['kgj_hour_limit_on']:
            p['kgj_hour_limit'] = st.number_input("Max. hodin provozu KGJ / rok", value=6000, min_value=1)
        p['kgj_gas_fix'] = st.checkbox("Fixní cena plynu pro KGJ")
        if p['kgj_gas_fix']:
            p['kgj_gas_fix_price'] = st.number_input("Fixní cena plynu – KGJ [€/MWh]",
                value=float(st.session_state.avg_gas_raw))

    # ── Kotel ─────────────────────────────────────
    if use_boil:
        st.subheader("Plynový kotel")
        p['b_max']    = st.number_input("Max. výkon [MW]",    value=3.91)
        p['boil_eff'] = st.number_input("Účinnost kotle [-]", value=0.95)
        p['boil_hour_limit_on'] = st.checkbox("Omezit max. počet provozních hodin kotle / rok", value=False)
        if p['boil_hour_limit_on']:
            p['boil_hour_limit'] = st.number_input("Max. hodin provozu kotle / rok", value=4000, min_value=1)
        p['boil_gas_fix'] = st.checkbox("Fixní cena plynu pro kotel")
        if p['boil_gas_fix']:
            p['boil_gas_fix_price'] = st.number_input("Fixní cena plynu – kotel [€/MWh]",
                value=float(st.session_state.avg_gas_raw))

    # ── Elektrokotel ──────────────────────────────
    if use_ek:
        st.subheader("Elektrokotel")
        p['ek_max'] = st.number_input("Max. výkon [MW]",  value=0.61)
        p['ek_eff'] = st.number_input("Účinnost EK [-]",  value=0.98)
        p['ek_ee_fix'] = st.checkbox("Fixní cena EE pro elektrokotel")
        if p['ek_ee_fix']:
            p['ek_ee_fix_price'] = st.number_input("Fixní cena EE – EK [€/MWh]",
                value=float(st.session_state.avg_ee_raw))

    # ── TES ───────────────────────────────────────
    if use_tes:
        st.subheader("Nádrž TES")
        p['tes_cap']  = st.number_input("Kapacita [MWh]", value=10.0)
        p['tes_loss'] = st.number_input("Ztráta [%/h]",   value=0.5) / 100

    # ── BESS ──────────────────────────────────────
    if use_bess:
        st.subheader("Baterie BESS")
        c1, c2 = st.columns(2)
        with c1:
            p['bess_cap']        = st.number_input("Kapacita [MWh]",                 value=1.0)
            p['bess_p']          = st.number_input("Max. výkon [MW]",                 value=0.5)
            p['bess_eff']        = st.number_input("Účinnost nabíjení/vybíjení [-]",  value=0.90)
            p['bess_cycle_cost'] = st.number_input("Náklady na opotřebení [€/MWh]",   value=5.0)
        with c2:
            st.markdown("**Distribuce pro arbitráž**")
            p['bess_dist_buy']  = st.checkbox("Účtovat distribuci NÁKUP do BESS",  value=False)
            p['bess_dist_sell'] = st.checkbox("Účtovat distribuci PRODEJ z BESS",  value=False)
            st.caption("💡 Interní arbitráž distribuci neplatí při zapnuté volbě 'Ušetřit distribuci'.")
        p['bess_ee_fix'] = st.checkbox("Fixní cena EE pro BESS")
        if p['bess_ee_fix']:
            p['bess_ee_fix_price'] = st.number_input("Fixní cena EE – BESS [€/MWh]",
                value=float(st.session_state.avg_ee_raw))

    # ── FVE ───────────────────────────────────────
    if use_fve:
        st.subheader("Fotovoltaika FVE")
        p['fve_installed_p'] = st.number_input("Instalovaný výkon [MW]", value=1.0,
            help="Profil FVE v lokálních datech = capacity factor 0–1.")
        p['fve_dist_sell'] = st.checkbox("Účtovat distribuci PRODEJ z FVE do sítě", value=False)

    # ── Import tepla ──────────────────────────────
    if use_ext_heat:
        st.subheader("Nákup tepla (Import)")
        p['imp_max']   = st.number_input("Max. výkon [MW]",      value=2.0)
        p['imp_price'] = st.number_input("Cena importu [€/MWh]", value=150.0)
        p['imp_hour_limit_on'] = st.checkbox("Omezit max. počet hodin importu tepla / rok", value=False)
        if p['imp_hour_limit_on']:
            p['imp_hour_limit'] = st.number_input("Max. hodin importu tepla / rok", value=2000, min_value=1)

# ────────────────────────────────────────────────
# SOLVER
# ────────────────────────────────────────────────
def run_optimization(df, params, uses, ee_delta=0.0, gas_delta=0.0,
                     h_price_override=None, time_limit=300):
    p        = params
    u        = uses
    T        = len(df)
    h_price  = h_price_override if h_price_override is not None else p['h_price']
    boil_eff = p.get('boil_eff', 0.95)
    ek_eff   = p.get('ek_eff',   0.98)

    model = pulp.LpProblem("KGJ_Dispatch", pulp.LpMaximize)

    # ── Proměnné ─────────────────────────────────
    if u['kgj']:
        q_kgj = pulp.LpVariable.dicts("q_KGJ",  range(T), 0, p['k_th'])
        on    = pulp.LpVariable.dicts("on",      range(T), 0, 1, "Binary")
        start = pulp.LpVariable.dicts("start",   range(T), 0, 1, "Binary")
    else:
        q_kgj = on = start = {t: 0 for t in range(T)}

    if u['boil']:
        q_boil  = pulp.LpVariable.dicts("q_Boil",   range(T), 0, p['b_max'])
        on_boil = pulp.LpVariable.dicts("on_boil",  range(T), 0, 1, "Binary")
    else:
        q_boil  = {t: 0 for t in range(T)}
        on_boil = {t: 0 for t in range(T)}

    q_ek  = pulp.LpVariable.dicts("q_EK",   range(T), 0, p['ek_max'])  if u['ek']  else {t: 0 for t in range(T)}

    if u['ext_heat']:
        q_imp  = pulp.LpVariable.dicts("q_Imp",   range(T), 0, p['imp_max'])
        on_imp = pulp.LpVariable.dicts("on_imp",  range(T), 0, 1, "Binary")
    else:
        q_imp  = {t: 0 for t in range(T)}
        on_imp = {t: 0 for t in range(T)}

    if u['tes']:
        tes_soc = pulp.LpVariable.dicts("TES_SOC", range(T+1), 0, p['tes_cap'])
        tes_in  = pulp.LpVariable.dicts("TES_In",  range(T), 0)
        tes_out = pulp.LpVariable.dicts("TES_Out", range(T), 0)
        model  += tes_soc[0] == p['tes_cap'] * 0.5
    else:
        tes_soc = {t: 0 for t in range(T+1)}
        tes_in = tes_out = {t: 0 for t in range(T)}

    if u['bess']:
        bess_soc = pulp.LpVariable.dicts("BESS_SOC", range(T+1), 0, p['bess_cap'])
        bess_cha = pulp.LpVariable.dicts("BESS_Cha", range(T), 0, p['bess_p'])
        bess_dis = pulp.LpVariable.dicts("BESS_Dis", range(T), 0, p['bess_p'])
        model   += bess_soc[0] == p['bess_cap'] * 0.2
    else:
        bess_soc = {t: 0 for t in range(T+1)}
        bess_cha = bess_dis = {t: 0 for t in range(T)}

    ee_export      = pulp.LpVariable.dicts("ee_export",  range(T), 0)
    ee_import      = pulp.LpVariable.dicts("ee_import",  range(T), 0)
    heat_shortfall = pulp.LpVariable.dicts("shortfall",  range(T), 0)

    # ── KGJ provozní omezení ─────────────────────
    if u['kgj']:
        for t in range(T):
            model += q_kgj[t] <= p['k_th'] * on[t]
            model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]
        model += start[0] == on[0]
        for t in range(1, T):
            model += start[t] >= on[t] - on[t-1]
            model += start[t] <= on[t]
            model += start[t] <= 1 - on[t-1]
        min_rt = int(p['k_min_runtime'])
        for t in range(T):
            for dt in range(1, min_rt):
                if t + dt < T:
                    model += on[t+dt] >= start[t]
        # Roční limit hodin
        if p.get('kgj_hour_limit_on') and p.get('kgj_hour_limit'):
            model += pulp.lpSum(on[t] for t in range(T)) <= p['kgj_hour_limit']

    # ── Kotel – on/off + roční limit ─────────────
    if u['boil']:
        for t in range(T):
            model += q_boil[t] <= p['b_max'] * on_boil[t]
        if p.get('boil_hour_limit_on') and p.get('boil_hour_limit'):
            model += pulp.lpSum(on_boil[t] for t in range(T)) <= p['boil_hour_limit']

    # ── Import tepla – on/off + roční limit ──────
    if u['ext_heat']:
        for t in range(T):
            model += q_imp[t] <= p['imp_max'] * on_imp[t]
        if p.get('imp_hour_limit_on') and p.get('imp_hour_limit'):
            model += pulp.lpSum(on_imp[t] for t in range(T)) <= p['imp_hour_limit']

    # ── Hlavní smyčka ─────────────────────────────
    obj = []
    for t in range(T):
        p_ee_m  = df['ee_price'].iloc[t]  + ee_delta
        p_gas_m = df['gas_price'].iloc[t] + gas_delta

        p_gas_kgj  = p.get('kgj_gas_fix_price',  p_gas_m) if (u['kgj']  and p.get('kgj_gas_fix'))  else p_gas_m
        p_gas_boil = p.get('boil_gas_fix_price', p_gas_m) if (u['boil'] and p.get('boil_gas_fix')) else p_gas_m
        p_ee_ek    = p.get('ek_ee_fix_price',    p_ee_m)  if (u['ek']   and p.get('ek_ee_fix'))   else p_ee_m

        h_dem = df['Poptávka po teple (MW)'].iloc[t]
        fve_p = float(df['FVE (MW)'].iloc[t]) if (u['fve'] and 'FVE (MW)' in df.columns) else 0.0

        if u['tes']:
            model += tes_soc[t+1] == tes_soc[t] * (1 - p['tes_loss']) + tes_in[t] - tes_out[t]
        if u['bess']:
            model += bess_soc[t+1] == bess_soc[t] + bess_cha[t]*p['bess_eff'] - bess_dis[t]/p['bess_eff']

        heat_delivered = q_kgj[t] + q_boil[t] + q_ek[t] + q_imp[t] + tes_out[t] - tes_in[t]
        model += heat_delivered + heat_shortfall[t] >= h_dem * p['h_cover']
        model += heat_delivered <= h_dem + 1e-3

        ee_kgj_out = q_kgj[t] * (p['k_eff_el'] / p['k_eff_th']) if u['kgj'] else 0
        ee_ek_in   = q_ek[t] / ek_eff                            if u['ek']  else 0
        model += ee_kgj_out + fve_p + ee_import[t] + bess_dis[t] == ee_ek_in + bess_cha[t] + ee_export[t]

        dist_sell_net       = p['dist_ee_sell'] if not p['internal_ee_use'] else 0.0
        dist_buy_net        = p['dist_ee_buy']  if not p['internal_ee_use'] else 0.0
        fve_dist_sell_cost  = p['dist_ee_sell'] if (u['fve'] and p.get('fve_dist_sell')) else 0.0
        bess_dist_buy_cost  = p['dist_ee_buy']  * bess_cha[t] if (u['bess'] and p.get('bess_dist_buy'))  else 0
        bess_dist_sell_cost = p['dist_ee_sell'] * bess_dis[t] if (u['bess'] and p.get('bess_dist_sell')) else 0

        revenue = (h_price * heat_delivered
                   + (p_ee_m - dist_sell_net - fve_dist_sell_cost) * ee_export[t])
        costs = (
            ((p_gas_kgj  + p['gas_dist']) * (q_kgj[t]  / p['k_eff_th']) if u['kgj']      else 0) +
            ((p_gas_boil + p['gas_dist']) * (q_boil[t] / boil_eff)       if u['boil']     else 0) +
            (p_ee_m + dist_buy_net) * ee_import[t] +
            ((p_ee_ek + dist_buy_net) * ee_ek_in                          if u['ek']       else 0) +
            (p['imp_price'] * q_imp[t]                                    if u['ext_heat'] else 0) +
            (p['k_start_cost'] * start[t]                                 if u['kgj']      else 0) +
            (p['bess_cycle_cost'] * (bess_cha[t] + bess_dis[t])           if u['bess']     else 0) +
            bess_dist_buy_cost + bess_dist_sell_cost +
            p['shortfall_penalty'] * heat_shortfall[t]
        )
        obj.append(revenue - costs)

    model += pulp.lpSum(obj)
    status = model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    if status not in (1, 2):
        return None

    def vv(v, t):
        x = v[t]
        return float(x) if isinstance(x, (int, float)) else float(pulp.value(x) or 0)

    res = pd.DataFrame({
        'Čas':                  df['datetime'],
        'Poptávka tepla [MW]':  df['Poptávka po teple (MW)'],
        'KGJ [MW_th]':          [vv(q_kgj,  t) for t in range(T)],
        'Kotel [MW_th]':        [vv(q_boil, t) for t in range(T)],
        'Elektrokotel [MW_th]': [vv(q_ek,   t) for t in range(T)],
        'Import tepla [MW_th]': [vv(q_imp,  t) for t in range(T)],
        'TES příjem [MW_th]':   [vv(tes_in,  t) for t in range(T)],
        'TES výdej [MW_th]':    [vv(tes_out, t) for t in range(T)],
        'TES SOC [MWh]':        [vv(tes_soc, t+1) for t in range(T)],
        'BESS nabíjení [MW]':   [vv(bess_cha, t) for t in range(T)],
        'BESS vybíjení [MW]':   [vv(bess_dis, t) for t in range(T)],
        'BESS SOC [MWh]':       [vv(bess_soc, t+1) for t in range(T)],
        'Shortfall [MW]':       [vv(heat_shortfall, t) for t in range(T)],
        'EE export [MW]':       [vv(ee_export, t) for t in range(T)],
        'EE import [MW]':       [vv(ee_import, t) for t in range(T)],
        'EE z KGJ [MW]':        [vv(q_kgj, t)*(p['k_eff_el']/p['k_eff_th']) if u['kgj'] else 0.0 for t in range(T)],
        'EE z FVE [MW]':        [float(df['FVE (MW)'].iloc[t]) if (u['fve'] and 'FVE (MW)' in df.columns) else 0.0 for t in range(T)],
        'EE do EK [MW]':        [vv(q_ek, t)/ek_eff if u['ek'] else 0.0 for t in range(T)],
        'Cena EE [€/MWh]':     (df['ee_price'] + ee_delta).values,
        'Cena plyn [€/MWh]':   (df['gas_price'] + gas_delta).values,
        'KGJ on':               [vv(on, t) for t in range(T)],
        'Kotel on':             [vv(on_boil, t) for t in range(T)],
        'Import tepla on':      [vv(on_imp, t) for t in range(T)],
    })
    res['TES netto [MW_th]'] = res['TES výdej [MW_th]'] - res['TES příjem [MW_th]']
    res['Dodáno tepla [MW]'] = (res['KGJ [MW_th]'] + res['Kotel [MW_th]'] +
                                res['Elektrokotel [MW_th]'] + res['Import tepla [MW_th]'] +
                                res['TES netto [MW_th]'])
    res['Měsíc']      = pd.to_datetime(res['Čas']).dt.month
    res['Hodina dne'] = pd.to_datetime(res['Čas']).dt.hour

    # ── Hodinové ekonomické toky ──────────────────
    rev_teplo_h, rev_ee_h = [], []
    c_gas_kgj_h, c_gas_boil_h = [], []
    c_ee_imp_h, c_ee_ek_h, c_imp_heat_h = [], [], []
    c_start_h, c_bess_h, c_penalty_h = [], [], []

    for t in range(T):
        p_ee_m   = df['ee_price'].iloc[t]  + ee_delta
        p_gas_m  = df['gas_price'].iloc[t] + gas_delta
        p_gas_kj = p.get('kgj_gas_fix_price',  p_gas_m) if (u['kgj']  and p.get('kgj_gas_fix'))  else p_gas_m
        p_gas_bh = p.get('boil_gas_fix_price', p_gas_m) if (u['boil'] and p.get('boil_gas_fix')) else p_gas_m
        p_ee_ekh = p.get('ek_ee_fix_price',    p_ee_m)  if (u['ek']   and p.get('ek_ee_fix'))   else p_ee_m

        fve_ds   = p['dist_ee_sell'] if (u['fve'] and p.get('fve_dist_sell')) else 0.0
        dist_s   = p['dist_ee_sell'] if not p['internal_ee_use'] else 0.0
        dist_b   = p['dist_ee_buy']  if not p['internal_ee_use'] else 0.0

        rt  = h_price * res['Dodáno tepla [MW]'].iloc[t]
        re  = (p_ee_m - dist_s - fve_ds) * res['EE export [MW]'].iloc[t]
        cg1 = (p_gas_kj + p['gas_dist']) * (res['KGJ [MW_th]'].iloc[t]  / p['k_eff_th']) if u['kgj']  else 0
        cg2 = (p_gas_bh + p['gas_dist']) * (res['Kotel [MW_th]'].iloc[t] / boil_eff)      if u['boil'] else 0
        ce1 = (p_ee_m + dist_b) * res['EE import [MW]'].iloc[t]
        ce2 = (p_ee_ekh + dist_b) * res['EE do EK [MW]'].iloc[t] if u['ek'] else 0
        ci  = p['imp_price'] * res['Import tepla [MW_th]'].iloc[t] if u['ext_heat'] else 0
        cs  = p['k_start_cost'] * vv(start, t) if u['kgj'] else 0
        cb  = (p['bess_cycle_cost'] * (res['BESS nabíjení [MW]'].iloc[t] + res['BESS vybíjení [MW]'].iloc[t])
               if u['bess'] else 0)
        cp  = p['shortfall_penalty'] * res['Shortfall [MW]'].iloc[t]

        rev_teplo_h.append(rt);  rev_ee_h.append(re)
        c_gas_kgj_h.append(cg1); c_gas_boil_h.append(cg2)
        c_ee_imp_h.append(ce1);  c_ee_ek_h.append(ce2)
        c_imp_heat_h.append(ci); c_start_h.append(cs)
        c_bess_h.append(cb);     c_penalty_h.append(cp)

    res['Rev teplo [€]']      = rev_teplo_h
    res['Rev EE [€]']         = rev_ee_h
    res['Nákl plyn KGJ [€]']  = c_gas_kgj_h
    res['Nákl plyn kotel [€]']= c_gas_boil_h
    res['Nákl EE import [€]'] = c_ee_imp_h
    res['Nákl EE EK [€]']     = c_ee_ek_h
    res['Nákl imp tepla [€]'] = c_imp_heat_h
    res['Nákl starty [€]']    = c_start_h
    res['Nákl BESS [€]']      = c_bess_h
    res['Nákl penalizace [€]']= c_penalty_h
    res['Hodinový zisk [€]']  = [
        rev_teplo_h[t] + rev_ee_h[t]
        - c_gas_kgj_h[t] - c_gas_boil_h[t]
        - c_ee_imp_h[t] - c_ee_ek_h[t]
        - c_imp_heat_h[t] - c_start_h[t]
        - c_bess_h[t] - c_penalty_h[t]
        for t in range(T)
    ]
    res['Kumulativní zisk [€]'] = res['Hodinový zisk [€]'].cumsum()

    return {'res': res, 'start': start, 'on': on, 'on_boil': on_boil, 'on_imp': on_imp,
            'status': status, 'total_profit': res['Hodinový zisk [€]'].sum()}

# ────────────────────────────────────────────────
# LOKÁLNÍ DATA + SPUŠTĚNÍ
# ────────────────────────────────────────────────
st.divider()
st.markdown("**Formát lokálních dat:** 1. sloupec = datetime | `Poptávka po teple (MW)` "
            "| `FVE (MW)` jako capacity factor **0–1**.")
loc_file = st.file_uploader("📂 Lokální data (poptávka tepla, FVE profil, ...)", type=["xlsx"])

if st.session_state.fwd_data is not None and loc_file is not None:
    df_loc = pd.read_excel(loc_file)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    df_loc.rename(columns={df_loc.columns[0]: 'datetime'}, inplace=True)
    df_loc['datetime'] = pd.to_datetime(df_loc['datetime'], dayfirst=True)
    df = pd.merge(st.session_state.fwd_data, df_loc, on='datetime', how='inner').fillna(0)
    if use_fve and 'fve_installed_p' in p and 'FVE (MW)' in df.columns:
        df['FVE (MW)'] = df['FVE (MW)'].clip(0, 1) * p['fve_installed_p']
    T = len(df)
    st.info(f"Načteno **{T}** hodin ({df['datetime'].min().date()} → {df['datetime'].max().date()})")

    uses = dict(kgj=use_kgj, boil=use_boil, ek=use_ek, tes=use_tes,
                bess=use_bess, fve=use_fve, ext_heat=use_ext_heat)

    if st.button("🏁 Spustit optimalizaci", type="primary"):
        with st.spinner("Probíhá optimalizace …"):
            result = run_optimization(df, p, uses)
        if result is None:
            st.error("Optimalizace nenašla přijatelné řešení. Zkontroluj parametry.")
            st.stop()
        st.session_state.results = result
        st.session_state.df_main = df.copy()
        st.session_state.uses    = uses
        st.success("Optimalizace dokončena ✔")

# ────────────────────────────────────────────────
# VÝSLEDKY
# ────────────────────────────────────────────────
if st.session_state.results is not None and st.session_state.df_main is not None:
    result  = st.session_state.results
    df      = st.session_state.df_main
    res     = result['res']
    uses    = st.session_state.get('uses', dict(kgj=use_kgj, boil=use_boil, ek=use_ek,
                tes=use_tes, bess=use_bess, fve=use_fve, ext_heat=use_ext_heat))
    T       = len(df)
    boil_eff= p.get('boil_eff', 0.95)

    # ── Agregované ekonomické hodnoty ────────────
    total_profit     = res['Hodinový zisk [€]'].sum()
    total_shortfall  = res['Shortfall [MW]'].sum()
    target_heat      = (res['Poptávka tepla [MW]'] * p['h_cover']).sum()
    coverage         = 100*(1 - total_shortfall/target_heat) if target_heat > 0 else 100.0
    total_ee_gen     = res['EE z KGJ [MW]'].sum() + res['EE z FVE [MW]'].sum()
    kgj_hours        = int(res['KGJ on'].sum())

    rev_teplo_total  = res['Rev teplo [€]'].sum()
    rev_ee_total     = res['Rev EE [€]'].sum()
    c_gas_total      = res['Nákl plyn KGJ [€]'].sum() + res['Nákl plyn kotel [€]'].sum()
    c_ee_total       = res['Nákl EE import [€]'].sum() + res['Nákl EE EK [€]'].sum()
    c_imp_total      = res['Nákl imp tepla [€]'].sum()
    c_other_total    = res['Nákl starty [€]'].sum() + res['Nákl BESS [€]'].sum()

    # ── METRIKY – řada 1: základní ───────────────
    st.subheader("📊 Klíčové metriky")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Celkový zisk",          f"{total_profit:,.0f} €")
    m2.metric("Shortfall celkem",      f"{total_shortfall:,.1f} MWh")
    m3.metric("Pokrytí poptávky",      f"{coverage:.1f} %")
    m4.metric("Export EE",             f"{res['EE export [MW]'].sum():,.1f} MWh")
    m5.metric("Výroba EE (KGJ+FVE)",  f"{total_ee_gen:,.1f} MWh")
    m6.metric("Provozní hodiny KGJ",   f"{kgj_hours:,} h")

    # ── METRIKY – řada 2: rozpad po komoditách ───
    st.markdown("**Rozpad zisku po komoditách**")
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("🔥 Příjmy teplo",       f"{rev_teplo_total:,.0f} €",
              help="Prodej tepla = dodáno teplo × cena tepla")
    r2.metric("⚡ Příjmy EE",          f"{rev_ee_total:,.0f} €",
              help="Export EE do sítě × (cena EE − distribuce)")
    r3.metric("🔴 Náklady plyn",       f"{c_gas_total:,.0f} €",
              help="Spotřeba plynu KGJ + kotel × (cena plynu + distribuce)")
    r4.metric("🔴 Náklady EE",         f"{c_ee_total:,.0f} €",
              help="Import EE + EE do EK × (cena EE + distribuce)")
    r5.metric("🔴 Náklady import tepla",f"{c_imp_total:,.0f} €",
              help="Nákup tepla z externího zdroje")
    r6.metric("🔴 Ostatní náklady",    f"{c_other_total:,.0f} €",
              help="Náklady na starty KGJ + opotřebení BESS")

    if total_shortfall > 0.5:
        st.warning(f"⚠️ Celkový shortfall {total_shortfall:.1f} MWh – zvyš penalizaci nebo kapacity zdrojů.")

    # ────────────────────────────────────────────
    # GRAF 1 – Pokrytí tepla
    # ────────────────────────────────────────────
    st.subheader("🔥 Pokrytí tepelné poptávky")
    fig = go.Figure()
    for col, name, color in [
        ('KGJ [MW_th]',          'KGJ',         '#27ae60'),
        ('Kotel [MW_th]',        'Kotel',        '#3498db'),
        ('Elektrokotel [MW_th]', 'Elektrokotel', '#9b59b6'),
        ('Import tepla [MW_th]', 'Import tepla', '#e74c3c'),
        ('TES netto [MW_th]',    'TES netto',    '#f39c12'),
    ]:
        fig.add_trace(go.Scatter(x=res['Čas'], y=res[col].clip(lower=0),
            name=name, stackgroup='teplo', fillcolor=color, line_width=0))
    fig.add_trace(go.Scatter(x=res['Čas'], y=res['Shortfall [MW]'],
        name='Nedodáno ⚠️', stackgroup='teplo', fillcolor='rgba(200,0,0,0.45)', line_width=0))
    fig.add_trace(go.Scatter(x=res['Čas'], y=res['Poptávka tepla [MW]']*p['h_cover'],
        name='Cílová poptávka', mode='lines', line=dict(color='black', width=2, dash='dot')))
    fig.update_layout(height=480, hovermode='x unified', title="Složení tepelné dodávky v čase")
    st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────
    # GRAF 2 – EE bilance
    # ────────────────────────────────────────────
    st.subheader("⚡ Bilance elektřiny")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.5, 0.5], subplot_titles=("Zdroje EE [MW]", "Spotřeba / export EE [MW]"))
    for col, name, color in [
        ('EE z KGJ [MW]',      'KGJ',       '#2ecc71'),
        ('EE z FVE [MW]',      'FVE',        '#f1c40f'),
        ('EE import [MW]',     'Import EE',  '#2980b9'),
        ('BESS vybíjení [MW]', 'BESS výdej', '#8e44ad'),
    ]:
        fig.add_trace(go.Scatter(x=res['Čas'], y=res[col], name=name,
            stackgroup='vyroba', fillcolor=color), row=1, col=1)
    for col, name, color in [
        ('EE do EK [MW]',      'EK',            '#e74c3c'),
        ('BESS nabíjení [MW]', 'BESS nabíjení', '#34495e'),
        ('EE export [MW]',     'Export EE',     '#16a085'),
    ]:
        fig.add_trace(go.Scatter(x=res['Čas'], y=-res[col], name=name,
            stackgroup='spotreba', fillcolor=color), row=2, col=1)
    fig.update_layout(height=640, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────
    # GRAF 3 – Stavy akumulátorů
    # ────────────────────────────────────────────
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

    # ────────────────────────────────────────────
    # GRAF 4 – Kumulativní zisk
    # ────────────────────────────────────────────
    st.subheader("💰 Kumulativní zisk")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['Čas'], y=res['Kumulativní zisk [€]'],
        fill='tozeroy', fillcolor='rgba(39,174,96,0.2)', line_color='#27ae60', name='Kum. zisk'))
    fig.update_layout(height=360, title="Průběh kumulativního zisku v čase")
    st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────
    # GRAF 5 – Měsíční analýza rozšířená
    # ────────────────────────────────────────────
    st.subheader("📅 Měsíční analýza – příjmy a náklady po zdrojích")

    monthly = res.groupby('Měsíc').agg(
        # Produkce [MWh]
        teplo_kgj   =('KGJ [MW_th]',          'sum'),
        teplo_kotel =('Kotel [MW_th]',         'sum'),
        teplo_ek    =('Elektrokotel [MW_th]',  'sum'),
        teplo_imp   =('Import tepla [MW_th]',  'sum'),
        ee_export   =('EE export [MW]',        'sum'),
        ee_import   =('EE import [MW]',        'sum'),
        shortfall   =('Shortfall [MW]',        'sum'),
        kgj_h       =('KGJ on',                'sum'),
        kotel_h     =('Kotel on',              'sum'),
        imp_h       =('Import tepla on',       'sum'),
        # Ekonomika [€]
        rev_teplo   =('Rev teplo [€]',         'sum'),
        rev_ee      =('Rev EE [€]',            'sum'),
        c_gas_kgj   =('Nákl plyn KGJ [€]',    'sum'),
        c_gas_kotel =('Nákl plyn kotel [€]',  'sum'),
        c_ee_imp    =('Nákl EE import [€]',   'sum'),
        c_ee_ek     =('Nákl EE EK [€]',       'sum'),
        c_imp_tepla =('Nákl imp tepla [€]',   'sum'),
        c_starty    =('Nákl starty [€]',       'sum'),
        c_bess      =('Nákl BESS [€]',        'sum'),
        c_pen       =('Nákl penalizace [€]',  'sum'),
        zisk        =('Hodinový zisk [€]',     'sum'),
    ).reset_index()
    monthly['Měsíc_str'] = monthly['Měsíc'].map(MONTH_NAMES)
    monthly['c_total']   = (monthly['c_gas_kgj'] + monthly['c_gas_kotel'] +
                            monthly['c_ee_imp']  + monthly['c_ee_ek'] +
                            monthly['c_imp_tepla']+ monthly['c_starty'] + monthly['c_bess'])

    # Graf 5a – Příjmy vs. náklady po měsících (stacked bar)
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Příjmy po zdrojích [€]", "Náklady po komoditách [€]"),
        horizontal_spacing=0.08)

    # Příjmy
    for col, name, color in [
        ('rev_teplo', 'Příjmy teplo', '#27ae60'),
        ('rev_ee',    'Příjmy EE',    '#f1c40f'),
    ]:
        fig.add_trace(go.Bar(x=monthly['Měsíc_str'], y=monthly[col],
            name=name, marker_color=color,
            customdata=monthly[col],
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{customdata:,.0f}} €<extra></extra>"),
            row=1, col=1)

    # Náklady (záporné → zobrazíme jako kladné hodnoty ve vlastním grafu)
    for col, name, color in [
        ('c_gas_kgj',   'Plyn – KGJ',        '#e74c3c'),
        ('c_gas_kotel', 'Plyn – kotel',       '#c0392b'),
        ('c_ee_imp',    'EE import',          '#2980b9'),
        ('c_ee_ek',     'EE – elektrokotel',  '#8e44ad'),
        ('c_imp_tepla', 'Import tepla',       '#e67e22'),
        ('c_starty',    'Starty KGJ',         '#7f8c8d'),
        ('c_bess',      'BESS opotřebení',    '#34495e'),
    ]:
        fig.add_trace(go.Bar(x=monthly['Měsíc_str'], y=monthly[col],
            name=name, marker_color=color,
            customdata=monthly[col],
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{customdata:,.0f}} €<extra></extra>"),
            row=1, col=2)

    # Zisk jako linie přes oba grafy
    fig.add_trace(go.Scatter(x=monthly['Měsíc_str'], y=monthly['zisk'],
        name='Čistý zisk', mode='lines+markers',
        line=dict(color='black', width=2.5),
        marker=dict(size=7, symbol='diamond'),
        hovertemplate="<b>Čistý zisk</b><br>%{x}: %{y:,.0f} €<extra></extra>"),
        row=1, col=1)

    fig.update_layout(height=500, barmode='stack', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=-0.35))
    st.plotly_chart(fig, use_container_width=True)

    # Graf 5b – Provozní hodiny po měsících
    st.markdown("**Provozní hodiny zdrojů po měsících**")
    fig = go.Figure()
    for col, name, color in [
        ('kgj_h',   'KGJ',         '#27ae60'),
        ('kotel_h', 'Kotel',        '#3498db'),
        ('imp_h',   'Import tepla', '#e74c3c'),
    ]:
        fig.add_trace(go.Bar(x=monthly['Měsíc_str'], y=monthly[col],
            name=name, marker_color=color,
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.0f}} h<extra></extra>"))
    fig.update_layout(height=340, barmode='group', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────
    # GRAF 6 – Průměrný denní profil
    # ────────────────────────────────────────────
    st.subheader("🕐 Průměrný denní profil")
    havg = res.groupby('Hodina dne').agg(
        teplo_popt =('Poptávka tepla [MW]', 'mean'),
        teplo_kgj  =('KGJ [MW_th]',         'mean'),
        teplo_kotel=('Kotel [MW_th]',        'mean'),
        teplo_ek   =('Elektrokotel [MW_th]', 'mean'),
        ee_kgj     =('EE z KGJ [MW]',        'mean'),
        ee_fve     =('EE z FVE [MW]',         'mean'),
        ee_import  =('EE import [MW]',        'mean'),
        cena_ee    =('Cena EE [€/MWh]',      'mean'),
    ).reset_index()
    hx = havg['Hodina dne']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
        vertical_spacing=0.08,
        subplot_titles=("Průměrná tepelná produkce [MW]", "Průměrná EE bilance [MW]"))
    for col, name, color in [
        ('teplo_kgj',   'KGJ',         '#27ae60'),
        ('teplo_kotel', 'Kotel',        '#3498db'),
        ('teplo_ek',    'Elektrokotel', '#9b59b6'),
    ]:
        fig.add_trace(go.Bar(x=hx, y=havg[col], name=name, marker_color=color), row=1, col=1)
    fig.add_trace(go.Scatter(x=hx, y=havg['teplo_popt'], name='Poptávka',
        mode='lines', line=dict(color='black', width=2, dash='dot')), row=1, col=1)
    for col, name, color in [
        ('ee_kgj',    'KGJ',    '#2ecc71'),
        ('ee_fve',    'FVE',    '#f1c40f'),
        ('ee_import', 'Import', '#2980b9'),
    ]:
        fig.add_trace(go.Bar(x=hx, y=havg[col], name=name, marker_color=color), row=2, col=1)
    fig.add_trace(go.Scatter(x=hx, y=havg['cena_ee'], name='Cena EE',
        mode='lines', line=dict(color='orange', width=2, dash='dot')), row=2, col=1)
    fig.update_layout(height=600, barmode='stack', hovermode='x unified',
        xaxis2=dict(title='Hodina dne'))
    st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────
    # GRAF 7 – Heatmapa zisku
    # ────────────────────────────────────────────
    st.subheader("🗓️ Heatmapa hodinového zisku")
    res_hm = res.copy()
    res_hm['Den']    = pd.to_datetime(res_hm['Čas']).dt.dayofyear
    res_hm['Hodina'] = pd.to_datetime(res_hm['Čas']).dt.hour
    pivot = res_hm.pivot_table(index='Hodina', columns='Den',
        values='Hodinový zisk [€]', aggfunc='sum')
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale='RdYlGn', colorbar=dict(title='€/hod'), zmid=0))
    fig.update_layout(height=420, xaxis_title="Den v roce", yaxis_title="Hodina dne",
        title="Hodinový zisk – den vs. hodina")
    st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────
    # GRAF 8 – Scatter KGJ citlivost
    # ────────────────────────────────────────────
    if use_kgj:
        st.subheader("🔍 Citlivost KGJ na cenu EE a plynu")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res['Cena EE [€/MWh]'], y=res['Cena plyn [€/MWh]'], mode='markers',
            marker=dict(color=res['KGJ on'], colorscale=[[0,'#e74c3c'],[1,'#27ae60']],
                size=4, opacity=0.5,
                colorbar=dict(title='KGJ', tickvals=[0,1], ticktext=['Off','On'])),
            text=[f"EE:{e:.1f} Plyn:{g:.1f} {'ON' if o>0.5 else 'OFF'}"
                  for e,g,o in zip(res['Cena EE [€/MWh]'], res['Cena plyn [€/MWh]'], res['KGJ on'])],
            hovertemplate='%{text}<extra></extra>'))
        fig.update_layout(height=450, xaxis_title='Cena EE [€/MWh]',
            yaxis_title='Cena plynu [€/MWh]',
            title='Provoz KGJ v závislosti na cenách EE a plynu')
        st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────
    # GRAF 9 – Waterfall
    # ────────────────────────────────────────────
    st.subheader("💵 Rozpad zisku – waterfall")
    wf_vals = [rev_teplo_total, rev_ee_total,
               -res['Nákl plyn KGJ [€]'].sum(), -res['Nákl plyn kotel [€]'].sum(),
               -res['Nákl EE import [€]'].sum(), -res['Nákl EE EK [€]'].sum(),
               -res['Nákl imp tepla [€]'].sum(),
               -res['Nákl starty [€]'].sum(), -res['Nákl BESS [€]'].sum(),
               -res['Nákl penalizace [€]'].sum(), total_profit]
    wf_lbls = ['Příjmy teplo','Příjmy EE','Plyn KGJ','Plyn kotel',
               'EE import','EE do EK','Import tepla',
               'Starty KGJ','BESS opotřebení','Penalizace','Celkový zisk']
    fig = go.Figure(go.Waterfall(
        orientation='v', measure=['relative']*(len(wf_vals)-1)+['total'],
        x=wf_lbls, y=wf_vals,
        connector=dict(line=dict(color='#bdc3c7', width=1)),
        decreasing=dict(marker_color='#e74c3c'),
        increasing=dict(marker_color='#27ae60'),
        totals=dict(marker_color='#2980b9'),
        text=[f"{v:,.0f} €" for v in wf_vals], textposition='outside'))
    fig.update_layout(height=500, title="Waterfall – rozpad příjmů a nákladů")
    st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════
    # CITLIVOSTNÍ ANALÝZA
    # ════════════════════════════════════════════
    st.divider()
    st.subheader("🎯 Citlivostní analýza – Spark spread (EE vs. Plyn)")
    with st.expander("⚙️ Nastavení citlivostní analýzy", expanded=True):
        sa1, sa2, sa3 = st.columns(3)
        with sa1:
            sa_steps     = st.select_slider("Kroků na osu (N² runů)", options=[3,5,7], value=5)
            sa_ee_range  = st.number_input("Rozsah EE ± [€/MWh]",   value=30.0, step=5.0)
            sa_gas_range = st.number_input("Rozsah plyn ± [€/MWh]", value=20.0, step=5.0)
        with sa2:
            sa_h_low    = st.number_input("Cena tepla min [€/MWh]", value=p['h_price']*0.8)
            sa_h_high   = st.number_input("Cena tepla max [€/MWh]", value=p['h_price']*1.2)
            sa_h_steps  = st.select_slider("Kroků ceny tepla", options=[3,5,7], value=3)
        with sa3:
            sa_tl = st.number_input("Časový limit / run [s]", value=60, min_value=10)
            st.caption(f"📐 Spark spread: {sa_steps}×{sa_steps} = **{sa_steps**2} runů**\n\n"
                       f"📐 Tornado teplo: **{sa_h_steps} runů**\n\n"
                       f"⏱️ Max ~{(sa_steps**2+sa_h_steps)*sa_tl//60+1} min")

    if st.button("🔬 Spustit citlivostní analýzu", type="secondary"):
        ee_deltas  = np.linspace(-sa_ee_range,  sa_ee_range,  sa_steps)
        gas_deltas = np.linspace(-sa_gas_range, sa_gas_range, sa_steps)
        h_prices   = np.linspace(sa_h_low, sa_h_high, sa_h_steps)
        total_runs = sa_steps**2 + sa_h_steps
        prog  = st.progress(0, text="Spark spread…")
        run_i = 0

        profit_mx = np.zeros((sa_steps, sa_steps))
        kgj_mx    = np.zeros((sa_steps, sa_steps))

        for i, g_d in enumerate(gas_deltas):
            for j, e_d in enumerate(ee_deltas):
                run_i += 1
                prog.progress(run_i/total_runs,
                    text=f"Run {run_i}/{total_runs} | EE Δ={e_d:+.0f} | Plyn Δ={g_d:+.0f} €/MWh")
                r = run_optimization(df, p, uses, ee_delta=e_d, gas_delta=g_d, time_limit=sa_tl)
                profit_mx[i, j] = r['total_profit'] if r else float('nan')
                kgj_mx[i, j]    = r['res']['KGJ on'].sum() if (r and use_kgj) else float('nan')

        ee_lbls  = [f"{st.session_state.ee_new+d:.0f}"  for d in ee_deltas]
        gas_lbls = [f"{st.session_state.gas_new+d:.0f}" for d in gas_deltas]

        # Tornado cena tepla
        prog.progress(sa_steps**2/total_runs, text="Tornado cena tepla…")
        tornado = []
        for hi, hp in enumerate(h_prices):
            run_i += 1
            prog.progress(run_i/total_runs, text=f"Teplo run {hi+1}/{sa_h_steps} | {hp:.0f} €/MWh")
            r = run_optimization(df, p, uses, h_price_override=hp, time_limit=sa_tl)
            if r:
                tornado.append({'cena_tepla': hp, 'zisk': r['total_profit']})
        prog.progress(1.0, text="Hotovo ✔")

        # Uložit vše do session_state – grafy se vykreslí níže mimo button blok
        st.session_state['sa_profit_mx']   = profit_mx
        st.session_state['sa_kgj_mx']      = kgj_mx
        st.session_state['sa_ee_lbls']     = ee_lbls
        st.session_state['sa_gas_lbls']    = gas_lbls
        st.session_state['sa_tornado']     = tornado
        st.session_state['sa_base_profit'] = total_profit
        st.session_state['sa_steps_done']  = sa_steps
        st.session_state['sa_use_kgj']     = use_kgj

    # ── Vykreslení výsledků citlivosti (perzistentní – mimo button blok) ──
    if st.session_state.get('sa_profit_mx') is not None:
        profit_mx  = st.session_state['sa_profit_mx']
        kgj_mx     = st.session_state['sa_kgj_mx']
        ee_lbls    = st.session_state['sa_ee_lbls']
        gas_lbls   = st.session_state['sa_gas_lbls']
        tornado    = st.session_state['sa_tornado']
        base_profit= st.session_state.get('sa_base_profit', total_profit)
        n_steps    = st.session_state.get('sa_steps_done', 5)
        sa_use_kgj = st.session_state.get('sa_use_kgj', use_kgj)

        st.markdown("#### Spark spread – výsledky")
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=("Celkový zisk [tis. €]", "Provozní hodiny KGJ [h]"),
            horizontal_spacing=0.12)
        fig.add_trace(go.Heatmap(
            z=profit_mx/1000, x=ee_lbls, y=gas_lbls,
            colorscale='RdYlGn', zmid=0, colorbar=dict(title='tis. €', x=0.45),
            text=[[f"{v:.0f} k€" for v in row] for row in profit_mx/1000],
            texttemplate="%{text}", textfont=dict(size=11),
            hovertemplate="EE: %{x} €/MWh<br>Plyn: %{y} €/MWh<br>Zisk: %{z:.1f} tis. €<extra></extra>"),
            row=1, col=1)
        fig.add_trace(go.Heatmap(
            z=kgj_mx, x=ee_lbls, y=gas_lbls,
            colorscale='Blues', colorbar=dict(title='hod', x=1.0),
            text=[[f"{v:.0f} h" for v in row] for row in kgj_mx],
            texttemplate="%{text}", textfont=dict(size=11),
            hovertemplate="EE: %{x} €/MWh<br>Plyn: %{y} €/MWh<br>KGJ h: %{z:.0f}<extra></extra>"),
            row=1, col=2)
        mid = n_steps // 2
        for c in [1, 2]:
            fig.add_trace(go.Scatter(
                x=[ee_lbls[mid]], y=[gas_lbls[mid]],
                mode='markers', marker=dict(symbol='star', size=18, color='black'),
                name='Základní scénář', showlegend=(c == 1)), row=1, col=c)
        fig.update_xaxes(title_text="Cena EE [€/MWh]")
        fig.update_yaxes(title_text="Cena plynu [€/MWh]")
        fig.update_layout(height=520, title="Spark spread analýza")
        st.plotly_chart(fig, use_container_width=True)

        if tornado:
            st.markdown("#### Citlivost zisku na cenu tepla")
            df_tor = pd.DataFrame(tornado)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_tor['cena_tepla'], y=df_tor['zisk']/1000,
                mode='lines+markers', line=dict(color='#3498db', width=2), marker=dict(size=8),
                hovertemplate="Cena tepla: %{x:.0f} €/MWh<br>Zisk: %{y:.0f} tis. €<extra></extra>"))
            fig.add_hline(y=base_profit/1000, line_dash="dot", line_color='#27ae60',
                annotation_text=f"Základ {base_profit/1000:.0f} tis. €")
            fig.add_hline(y=0, line_dash="dash", line_color='#e74c3c',
                annotation_text="Break-even")
            fig.update_layout(height=400, xaxis_title='Cena tepla [€/MWh]',
                yaxis_title='Celkový zisk [tis. €]',
                title='Citlivost zisku na cenu tepla', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════
    # EXCEL EXPORT
    # ════════════════════════════════════════════
    st.divider()
    st.subheader("⬇️ Export výsledků")

    def to_excel(df_res, df_input, params, monthly_df):
        buf = io.BytesIO()
        skip_cols = {'Měsíc','Hodina dne','KGJ on','Kotel on','Import tepla on'}
        df_exp    = df_res[[c for c in df_res.columns if c not in skip_cols]].copy()

        monthly_exp = monthly_df.copy()
        monthly_exp['Měsíc'] = monthly_exp['Měsíc'].map(MONTH_NAMES)
        monthly_exp = monthly_exp.rename(columns={
            'Měsíc_str':'Měsíc_label','teplo_kgj':'KGJ teplo [MWh]','teplo_kotel':'Kotel teplo [MWh]',
            'teplo_ek':'EK teplo [MWh]','teplo_imp':'Import tepla [MWh]',
            'ee_export':'EE export [MWh]','ee_import':'EE import [MWh]','shortfall':'Shortfall [MWh]',
            'kgj_h':'KGJ hodiny [h]','kotel_h':'Kotel hodiny [h]','imp_h':'Import hodiny [h]',
            'rev_teplo':'Příjmy teplo [€]','rev_ee':'Příjmy EE [€]',
            'c_gas_kgj':'Nákl plyn KGJ [€]','c_gas_kotel':'Nákl plyn kotel [€]',
            'c_ee_imp':'Nákl EE import [€]','c_ee_ek':'Nákl EE EK [€]',
            'c_imp_tepla':'Nákl import tepla [€]','c_starty':'Nákl starty [€]',
            'c_bess':'Nákl BESS [€]','c_pen':'Nákl penalizace [€]',
            'c_total':'Celkové náklady [€]','zisk':'Čistý zisk [€]'
        })
        exp_monthly_cols = [
            'Měsíc_label','KGJ teplo [MWh]','Kotel teplo [MWh]','EK teplo [MWh]',
            'Import tepla [MWh]','EE export [MWh]','EE import [MWh]','Shortfall [MWh]',
            'KGJ hodiny [h]','Kotel hodiny [h]','Import hodiny [h]',
            'Příjmy teplo [€]','Příjmy EE [€]',
            'Nákl plyn KGJ [€]','Nákl plyn kotel [€]','Nákl EE import [€]',
            'Nákl EE EK [€]','Nákl import tepla [€]','Nákl starty [€]',
            'Nákl BESS [€]','Nákl penalizace [€]','Celkové náklady [€]','Čistý zisk [€]'
        ]
        monthly_exp = monthly_exp[[c for c in exp_monthly_cols if c in monthly_exp.columns]]

        param_rows = [
            ('=== OBECNÉ ===',''),
            ('Cena tepla [€/MWh]', params['h_price']),
            ('Min. pokrytí [-]',   params['h_cover']),
            ('Penalizace shortfall [€/MWh]', params['shortfall_penalty']),
            ('Distribuce nákup EE [€/MWh]',  params['dist_ee_buy']),
            ('Distribuce prodej EE [€/MWh]', params['dist_ee_sell']),
            ('Distribuce plyn [€/MWh]',      params['gas_dist']),
            ('Interní spotřeba – bez dist.',  params['internal_ee_use']),
        ]
        if use_kgj:
            param_rows += [('=== KGJ ===',''),
                ('k_th [MW]', params['k_th']), ('η_th', params['k_eff_th']),
                ('η_el', params['k_eff_el']), ('k_el odv. [MW]', params['k_el']),
                ('Min. zatížení', params['k_min']), ('Start cost [€]', params['k_start_cost']),
                ('Min. runtime [h]', params['k_min_runtime']),
                ('Roční limit hodin', params.get('kgj_hour_limit_on', False)),
                ('Max hodin / rok', params.get('kgj_hour_limit', '-'))]
        if use_boil:
            param_rows += [('=== KOTEL ===',''), ('b_max [MW]', params['b_max']),
                ('Účinnost', params['boil_eff']),
                ('Roční limit hodin', params.get('boil_hour_limit_on', False)),
                ('Max hodin / rok', params.get('boil_hour_limit', '-'))]
        if use_ek:
            param_rows += [('=== ELEKTROKOTEL ===',''),
                ('ek_max [MW]', params['ek_max']), ('Účinnost', params['ek_eff'])]
        if use_tes:
            param_rows += [('=== TES ===',''),
                ('Kapacita [MWh]', params['tes_cap']), ('Ztráta [%/h]', params['tes_loss']*100)]
        if use_bess:
            param_rows += [('=== BESS ===',''),
                ('Kapacita [MWh]', params['bess_cap']), ('Výkon [MW]', params['bess_p']),
                ('Účinnost', params['bess_eff']), ('Opotřebení [€/MWh]', params['bess_cycle_cost']),
                ('Distribuce nákup', params.get('bess_dist_buy', False)),
                ('Distribuce prodej', params.get('bess_dist_sell', False))]
        if use_fve:
            param_rows += [('=== FVE ===',''),
                ('Instalovaný výkon [MW]', params['fve_installed_p']),
                ('Distribuce prodej FVE',  params.get('fve_dist_sell', False))]
        if use_ext_heat:
            param_rows += [('=== IMPORT TEPLA ===',''),
                ('Max výkon [MW]', params['imp_max']), ('Cena [€/MWh]', params['imp_price']),
                ('Roční limit hodin', params.get('imp_hour_limit_on', False)),
                ('Max hodin / rok', params.get('imp_hour_limit', '-'))]

        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            wb = writer.book
            fmt_h  = wb.add_format({'bold':True,'bg_color':'#2c3e50','font_color':'white',
                                     'border':1,'align':'center','text_wrap':True})
            fmt_h2 = wb.add_format({'bold':True,'bg_color':'#1a5276','font_color':'white',
                                     'border':1,'align':'center'})
            fmt_n2 = wb.add_format({'num_format':'#,##0.00','border':1})
            fmt_n0 = wb.add_format({'num_format':'#,##0','border':1})
            fmt_dt = wb.add_format({'num_format':'dd.mm.yyyy hh:mm','border':1})
            fmt_sc = wb.add_format({'bold':True,'bg_color':'#d5e8d4','border':1})
            money  = {'Hodinový zisk [€]','Kumulativní zisk [€]'}

            def write_df(ws, dfx, hfmt, mcols):
                for ci, cn in enumerate(dfx.columns):
                    ws.set_column(ci, ci, 20); ws.write(0, ci, cn, hfmt)
                for ri in range(len(dfx)):
                    for ci, cn in enumerate(dfx.columns):
                        cv = dfx.iloc[ri, ci]
                        if cn in ('Čas','datetime'):
                            try: ws.write_datetime(ri+1,ci,pd.Timestamp(cv).to_pydatetime(),fmt_dt)
                            except: ws.write(ri+1,ci,str(cv))
                        elif cn in mcols:
                            try: ws.write_number(ri+1,ci,float(cv),fmt_n0)
                            except: ws.write(ri+1,ci,str(cv))
                        else:
                            try: ws.write_number(ri+1,ci,float(cv),fmt_n2)
                            except: ws.write(ri+1,ci,str(cv))
                ws.autofilter(0,0,len(dfx),len(dfx.columns)-1)
                ws.freeze_panes(1,1); ws.set_row(0,36)

            # List 1 – Hodinová data
            df_exp.to_excel(writer, index=False, sheet_name='Hodinová data')
            write_df(writer.sheets['Hodinová data'], df_exp, fmt_h, money)

            # List 2 – Měsíční souhrn (rozšířený)
            monthly_exp.to_excel(writer, index=False, sheet_name='Měsíční souhrn')
            all_money = {c for c in monthly_exp.columns if '€' in c}
            write_df(writer.sheets['Měsíční souhrn'], monthly_exp, fmt_h, all_money)

            # List 3 – Vstupní data
            df_input.to_excel(writer, index=False, sheet_name='Vstupní data')
            ws3 = writer.sheets['Vstupní data']
            for ci, cn in enumerate(df_input.columns):
                ws3.set_column(ci, ci, 20); ws3.write(0, ci, cn, fmt_h2)
            for ri in range(len(df_input)):
                for ci, cn in enumerate(df_input.columns):
                    cv = df_input.iloc[ri, ci]
                    if cn == 'datetime':
                        try: ws3.write_datetime(ri+1,ci,pd.Timestamp(cv).to_pydatetime(),fmt_dt)
                        except: ws3.write(ri+1,ci,str(cv))
                    else:
                        try: ws3.write_number(ri+1,ci,float(cv),fmt_n2)
                        except: ws3.write(ri+1,ci,str(cv))
            ws3.freeze_panes(1,0); ws3.set_row(0,30)

            # List 4 – Parametry
            ws4 = wb.add_worksheet('Parametry')
            ws4.set_column(0,0,35); ws4.set_column(1,1,20)
            ws4.write(0,0,'Parametr',fmt_h); ws4.write(0,1,'Hodnota',fmt_h)
            for ri,(pn,pv) in enumerate(param_rows):
                if str(pv)=='':
                    ws4.write(ri+1,0,pn,fmt_sc); ws4.write(ri+1,1,'',fmt_sc)
                else:
                    ws4.write(ri+1,0,pn)
                    try: ws4.write_number(ri+1,1,float(pv),fmt_n2)
                    except: ws4.write(ri+1,1,str(pv))

            # List 5 – Citlivostní analýza (pokud proběhla)
            if 'sa_profit_mx' in st.session_state:
                pm  = st.session_state['sa_profit_mx']
                km  = st.session_state['sa_kgj_mx']
                eel = st.session_state['sa_ee_lbls']
                gl  = st.session_state['sa_gas_lbls']
                ws5 = wb.add_worksheet('Citlivost – Spark spread')
                ws5.write(0,0,'Zisk [€] – Plyn↓ EE→', fmt_sc)
                for j,l in enumerate(eel): ws5.write(0,j+1,f"EE={l}",fmt_h)
                for i,g in enumerate(gl):
                    ws5.write(i+1,0,f"Plyn={g}",fmt_h)
                    for j in range(len(eel)):
                        clr = '#c8e6c9' if pm[i,j]>=0 else '#ffcdd2'
                        ws5.write_number(i+1,j+1,float(pm[i,j]),
                            wb.add_format({'num_format':'#,##0','border':1,'bg_color':clr}))
                row_off = len(gl)+3
                ws5.write(row_off,0,'Hodiny KGJ – Plyn↓ EE→',fmt_sc)
                for j,l in enumerate(eel): ws5.write(row_off,j+1,f"EE={l}",fmt_h)
                for i,g in enumerate(gl):
                    ws5.write(row_off+i+1,0,f"Plyn={g}",fmt_h)
                    for j in range(len(eel)):
                        ws5.write_number(row_off+i+1,j+1,float(km[i,j]),
                            wb.add_format({'num_format':'#,##0','border':1}))

                if st.session_state.get('sa_tornado'):
                    df_tor2 = pd.DataFrame(st.session_state['sa_tornado'])
                    df_tor2.columns = ['Cena tepla [€/MWh]','Celkový zisk [€]']
                    df_tor2.to_excel(writer, index=False, sheet_name='Citlivost – Teplo')
                    wst = writer.sheets['Citlivost – Teplo']
                    wst.set_column(0,0,22); wst.set_column(1,1,20)
                    wst.write(0,0,'Cena tepla [€/MWh]',fmt_h)
                    wst.write(0,1,'Celkový zisk [€]',  fmt_h)

        return buf.getvalue()

    xlsx = to_excel(res.round(4), df, p, monthly)
    st.download_button(
        label="📥 Stáhnout výsledky + citlivost (Excel .xlsx)",
        data=xlsx,
        file_name="kgj_optimalizace.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
