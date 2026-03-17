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
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

MONTH_NAMES = {1:'Led',2:'Úno',3:'Bře',4:'Dub',5:'Kvě',6:'Čvn',
               7:'Čvc',8:'Srp',9:'Zář',10:'Říj',11:'Lis',12:'Pro'}

# ════════════════════════════════════════════════════════════════════
# SCHEDULING PROFILES & SCENARIO MANAGEMENT
# ════════════════════════════════════════════════════════════════════

def create_profile_constraints(df, profile_type, custom_hours=None):
    """
    Vytvoří binary constrainty pro KGJ provoz dle profilu
    Returns: list[bool] kde True = povoleno běžet, False = musí být OFF
    """
    df_work = df.copy()
    df_work['hour'] = pd.to_datetime(df_work['datetime']).dt.hour
    
    if profile_type == 'base':
        # Celý den (0-23)
        constraints = [True] * len(df_work)
    
    elif profile_type == 'peak':
        # 9-21 (Peak hours)
        constraints = [h in range(9, 22) for h in df_work['hour']]
    
    elif profile_type == 'extpeak':
        # 6-22 (Extended Peak)
        constraints = [h in range(6, 23) for h in df_work['hour']]
    
    elif profile_type == 'custom' and custom_hours:
        # Custom hours zadané uživatelem
        constraints = [h in custom_hours for h in df_work['hour']]
    
    else:
        # 'free' nebo neurčeno = bez omezení
        constraints = [True] * len(df_work)
    
    return constraints


def apply_profile_constraints_to_model(model, on, constraints, T):
    """Aplikuj profile constrainty do PuLP modelu"""
    if constraints is None:
        return model
    
    for t in range(T):
        if not constraints[t]:
            model += on[t] == 0, f"profile_constraint_{t}"
    
    return model


def calculate_smoothness_metrics(res):
    """
    Spočítej metriky hladkosti provozu KGJ
    """
    kgj_on = res['KGJ on'].values
    
    # Počet ON→OFF a OFF→ON přechodů
    transitions = np.sum(np.abs(np.diff(kgj_on)) > 0.5)
    
    # Délky kontinuálních běhů
    run_lengths = []
    current_run = 0
    for i in range(len(kgj_on)):
        if kgj_on[i] > 0.5:
            current_run += 1
        else:
            if current_run > 0:
                run_lengths.append(current_run)
            current_run = 0
    if current_run > 0:
        run_lengths.append(current_run)
    
    avg_run_length = np.mean(run_lengths) if run_lengths else 0
    min_run_length = np.min(run_lengths) if run_lengths else 0
    max_run_length = np.max(run_lengths) if run_lengths else 0
    
    # Stabilita skóre (0-100%, méně transakcí = vyšší skóre)
    stability_score = max(0, 100 * (1 - transitions / (len(kgj_on) / 2))) if len(kgj_on) > 0 else 0
    
    return {
        'transitions': int(transitions),
        'stability_score': stability_score,
        'avg_run_hours': avg_run_length,
        'min_run_hours': min_run_length,
        'max_run_hours': max_run_length,
        'total_on_hours': int(np.sum(kgj_on)),
    }


def create_scenario_comparison_df(scenarios):
    """Vytvoř DataFrame s porovnáním scénářů"""
    data = []
    
    for profile_name, scenario in scenarios.items():
        if scenario['result'] is None:
            continue
        res = scenario['result']['res']
        smooth = scenario['smoothness']
        
        profit = scenario['result']['total_profit']
        shortfall = res['Shortfall [MW]'].sum() if 'Shortfall [MW]' in res.columns else 0
        
        data.append({
            'Profil': profile_name.upper(),
            'Zisk [€]': f"{profit:,.0f}",
            'Transitions': smooth['transitions'],
            'Stabilita [%]': f"{smooth['stability_score']:.1f}",
            'Avg Runtime [h]': f"{smooth['avg_run_hours']:.1f}",
            'Total Hours': smooth['total_on_hours'],
            'Shortfall [MWh]': f"{shortfall:.1f}",
        })
    
    return pd.DataFrame(data)


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

    # ── NOVÉ: KGJ Scheduling Profily ──
    st.divider()
    st.header("📅 Analýza Období & Profily KGJ")
    
    # Výběr Období
    st.subheader("1️⃣ Období Analýzy")
    analysis_mode = st.radio("Vyberte režim:", ["Celá data", "Vlastní rozsah"])
    
    period_start = None
    period_end = None
    if analysis_mode == "Vlastní rozsah" and st.session_state.fwd_data is not None:
        min_date = pd.to_datetime(st.session_state.fwd_data['datetime']).min().date()
        max_date = pd.to_datetime(st.session_state.fwd_data['datetime']).max().date()
        
        col_ps, col_pe = st.columns(2)
        with col_ps:
            period_start = st.date_input("Od", value=min_date, key="period_start")
        with col_pe:
            period_end = st.date_input("Do", value=max_date, key="period_end")
    
    # KGJ Scheduling Profily
    st.subheader("2️⃣ Scheduling Profily KGJ")
    
    profiles_to_run = st.multiselect(
        "Které profily testovat?",
        options=['free', 'base', 'peak', 'extpeak', 'custom'],
        default=['free', 'base', 'peak', 'extpeak'],
        help="Spusť optimalizaci pro vybrané profily a porovnej je"
    )
    
    profile_definitions = {
        'free': {'name': 'Volná Opt.', 'hours': None, 'desc': 'Bez omezení'},
        'base': {'name': 'Base (0-24h)', 'hours': list(range(24)), 'desc': 'Celý den'},
        'peak': {'name': 'Peak (9-21h)', 'hours': list(range(9, 22)), 'desc': '12 hodin'},
        'extpeak': {'name': 'ExtPeak (6-22h)', 'hours': list(range(6, 23)), 'desc': '16 hodin'},
    }
    
    custom_hours = None
    if 'custom' in profiles_to_run:
        st.write("**Custom Profil** - Vyberte hodiny:")
        custom_hours = st.multiselect(
            "Povolené hodiny (0-23):",
            options=list(range(24)),
            default=list(range(6, 23)),
            key="custom_hours_selector"
        )
        profile_definitions['custom'] = {
            'name': f'Custom ({len(custom_hours)}h)',
            'hours': custom_hours,
            'desc': f"Custom"
        }
    
    # Provozní Omezení
    st.subheader("3️⃣ Omezení Provozování")
    
    use_month_start_limit = st.checkbox("Omezit max. startů za měsíc", value=False)
    max_starts_per_month = None
    if use_month_start_limit:
        max_starts_per_month = st.number_input("Max. startů/měsíc", value=5, min_value=1, max_value=30)
    
    # Scenario Mode
    st.subheader("4️⃣ Režim Porovnání")
    scenario_mode = st.radio(
        "Spusť optimalizaci:",
        ["Pouze aktuální nastavení", "Všechny vybrané profily"],
        help="'Všechny profily' = porovnání scénářů"
    )

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
# ENHANCED SOLVER S PROFILE SUPPORT
# ────────────────────────────────────────────────

def run_optimization_with_profile(df, params, uses, profile_type='free', custom_hours=None,
                                   ee_delta=0.0, gas_delta=0.0, h_price_override=None, 
                                   time_limit=300, max_starts_per_month=None, period_mask=None):
    """
    Enhanced solver s podporou KGJ scheduling profilů
    
    Parameters:
    - profile_type: 'free', 'base', 'peak', 'extpeak', 'custom'
    - custom_hours: list hodin (0-23) pokud profile_type=='custom'
    - max_starts_per_month: omezení startů na měsíc
    - period_mask: boolean array pro filtrování časového období
    """
    
    p        = params
    u        = uses
    T        = len(df)
    h_price  = h_price_override if h_price_override is not None else p['h_price']
    boil_eff = p.get('boil_eff', 0.95)
    ek_eff   = p.get('ek_eff',   0.98)

    # Filtruj podle období
    if period_mask is not None:
        df = df[period_mask].reset_index(drop=True)
        T = len(df)
    
    # Vytvoř profile constrainty
    profile_constraints = create_profile_constraints(df, profile_type, custom_hours)

    model = pulp.LpProblem("KGJ_Dispatch_Profile", pulp.LpMaximize)

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
            
            # PROFILE CONSTRAINT - zakažeme běh mimo dovolené hodiny
            if not profile_constraints[t]:
                model += on[t] == 0, f"profile_off_{t}"
        
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
        
        # NOVÉ: Limit startů za měsíc
        if max_starts_per_month is not None and u['kgj']:
            df_month = df.copy()
            df_month['month'] = pd.to_datetime(df_month['datetime']).dt.to_period('M')
            for month in df_month['month'].unique():
                month_indices = df_month[df_month['month'] == month].index.tolist()
                if len(month_indices) > 0:
                    model += pulp.lpSum(start[t] for t in month_indices) <= max_starts_per_month, f"starts_limit_{month}"

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


def run_scenario_analysis(df, params, uses, profiles_to_run, custom_hours=None, 
                          period_start=None, period_end=None, max_starts_per_month=None):
    """
    Spusť optimalizaci pro všechny vybrané profily a vrať porovnání
    """
    scenarios = {}
    
    # Vytvoř period mask pokud je zadáno
    period_mask = None
    if period_start is not None and period_end is not None:
        df_work = df.copy()
        df_work['date'] = pd.to_datetime(df_work['datetime']).dt.date
        period_mask = (df_work['date'] >= period_start) & (df_work['date'] <= period_end)
    
    progress_container = st.container()
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    for idx, profile in enumerate(profiles_to_run):
        status_text.write(f"⏳ Optimalizuji profil: **{profile.upper()}**...")
        
        profile_custom_hours = custom_hours if profile == 'custom' else None
        
        result = run_optimization_with_profile(
            df=df,
            params=params,
            uses=uses,
            profile_type=profile,
            custom_hours=profile_custom_hours,
            max_starts_per_month=max_starts_per_month,
            period_mask=period_mask
        )
        
        if result is not None:
            smoothness = calculate_smoothness_metrics(result['res'])
            scenarios[profile] = {
                'result': result,
                'smoothness': smoothness,
                'profile_name': profile.upper(),
            }
        
        progress_bar.progress((idx + 1) / len(profiles_to_run))
    
    status_text.write("✅ Scenáře spočítány!")
    progress_bar.empty()
    
    return scenarios

# ────────────────────────────────────────────────
# LOKÁLNÍ DATA + SPUŠTĚNÍ OPTIMALIZACE
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

    # ════════════════════════════════════════════════
    # SCENARIO MODE SELECTION & EXECUTION
    # ════════════════════════════════════════════════
    
    col_mode_1, col_mode_2 = st.columns([2, 1])
    
    with col_mode_1:
        if scenario_mode == "Pouze aktuální nastavení":
            st.markdown("### 🎯 Režim: Jednoduché Spuštění")
            st.markdown(
                f"Optimalizace se spustí s **aktuálním profilem KGJ** a všemi zadanými parametry. "
                f"Vhodné pro detailní analýzu jednoho scénáře."
            )
            
            if st.button("🚀 Spustit optimalizaci", type="primary", key="btn_single"):
                with st.spinner("⏳ Probíhá optimalizace …"):
                    result = run_optimization_with_profile(
                        df=df,
                        params=p,
                        uses=uses,
                        profile_type='free',  # Volná optimalizace
                        max_starts_per_month=max_starts_per_month if use_month_start_limit else None,
                        period_mask=None
                    )
                
                if result is None:
                    st.error("❌ Optimalizace nenašla přijatelné řešení. Zkontroluj parametry.")
                    st.stop()
                
                st.session_state.results = result
                st.session_state.df_main = df.copy()
                st.session_state.uses = uses
                st.session_state.scenario_results = None  # Vymaž scenario results
                st.success("✅ Optimalizace dokončena!")
        
        else:  # Scenario Mode
            st.markdown("### 📊 Režim: Porovnání Profilů")
            st.markdown(
                f"Spustit se budou profily: **{', '.join([p.upper() for p in profiles_to_run])}**. "
                f"Poté se zobrazí přehledné porovnání jejich ekonomiky a kvality provozu."
            )
            
            if st.button("🔬 Spustit analýzu scénářů", type="primary", key="btn_scenarios"):
                with st.spinner("⏳ Probíhá scenáristická analýza …"):
                    scenarios = run_scenario_analysis(
                        df=df,
                        params=p,
                        uses=uses,
                        profiles_to_run=profiles_to_run,
                        custom_hours=custom_hours if 'custom' in profiles_to_run else None,
                        period_start=period_start,
                        period_end=period_end,
                        max_starts_per_month=max_starts_per_month if use_month_start_limit else None
                    )
                
                if not scenarios:
                    st.error("❌ Žádný scenář nebyl úspěšně vypočten!")
                    st.stop()
                
                st.session_state.scenario_results = scenarios
                st.session_state.results = None  # Vymaž single result
                st.success("✅ Scenáristická analýza dokončena!")
    
    with col_mode_2:
        st.markdown("")
        st.markdown("")
        if scenario_mode == "Všechny vybrané profily":
            selected_profile = st.radio(
                "Profil k detailu:",
                options=profiles_to_run,
                format_func=lambda x: x.upper(),
                key="profile_selector"
            )
            st.session_state.selected_profile = selected_profile

# ────────────────────────────────────────────────
# SCENARIO COMPARISON VIEW
# ────────────────────────────────────────────────
if st.session_state.scenario_results is not None:
    scenarios = st.session_state.scenario_results
    
    st.divider()
    st.subheader("📋 Porovnání Scénářů (Scenario Comparison)")
    
    # Comparison Table
    comparison_df = create_scenario_comparison_df(scenarios)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Metric Comparison Charts
    col_chart_1, col_chart_2 = st.columns(2)
    
    with col_chart_1:
        st.markdown("**Zisk vs. Stabilita**")
        chart_data = []
        for profile, scenario in scenarios.items():
            if scenario['result'] is not None:
                profit = scenario['result']['total_profit']
                stability = scenario['smoothness']['stability_score']
                chart_data.append({
                    'Profil': profile.upper(),
                    'Zisk [k€]': profit / 1000,
                    'Stabilita [%]': stability
                })
        
        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_chart['Profil'],
                y=df_chart['Zisk [k€]'],
                name='Zisk [k€]',
                marker_color='#2ecc71',
                yaxis='y1'
            ))
            fig.add_trace(go.Scatter(
                x=df_chart['Profil'],
                y=df_chart['Stabilita [%]'],
                name='Stabilita [%]',
                line=dict(color='#3498db', width=3),
                marker=dict(size=10),
                yaxis='y2'
            ))
            fig.update_layout(
                height=380,
                yaxis=dict(title="Zisk [k€]", side='left'),
                yaxis2=dict(title="Stabilita [%]", overlaying='y', side='right'),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col_chart_2:
        st.markdown("**Charakteristiky Provozu**")
        chart_data2 = []
        for profile, scenario in scenarios.items():
            if scenario['result'] is not None:
                smooth = scenario['smoothness']
                chart_data2.append({
                    'Profil': profile.upper(),
                    'Transitions': smooth['transitions'],
                    'Avg Runtime [h]': smooth['avg_run_hours']
                })
        
        if chart_data2:
            df_chart2 = pd.DataFrame(chart_data2)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=df_chart2['Profil'],
                y=df_chart2['Transitions'],
                name='Přechodů ON↔OFF',
                marker_color='#e74c3c',
                yaxis='y1'
            ))
            fig2.add_trace(go.Scatter(
                x=df_chart2['Profil'],
                y=df_chart2['Avg Runtime [h]'],
                name='Avg Runtime [h]',
                line=dict(color='#f39c12', width=3),
                marker=dict(size=10),
                yaxis='y2'
            ))
            fig2.update_layout(
                height=380,
                yaxis=dict(title="Přechodů", side='left'),
                yaxis2=dict(title="Avg Runtime [h]", overlaying='y', side='right'),
                hovermode='x unified'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # ── Detailní view vybraného profilu ──
    st.divider()
    st.subheader(f"🔍 Detailní Analýza: {st.session_state.selected_profile.upper()}")
    
    selected_scenario = scenarios.get(st.session_state.selected_profile)
    if selected_scenario and selected_scenario['result'] is not None:
        result = selected_scenario['result']
        res = result['res']
        smoothness = selected_scenario['smoothness']
        
        # Smoothness Metrics Box
        col_sm_1, col_sm_2, col_sm_3, col_sm_4, col_sm_5 = st.columns(5)
        
        with col_sm_1:
            st.metric(
                "Přechodů ON↔OFF",
                f"{smoothness['transitions']}",
                help="Počet start-stop cyklů (méně = lépe)"
            )
        
        with col_sm_2:
            st.metric(
                "Stabilita Skóre",
                f"{smoothness['stability_score']:.1f}%",
                help="0-100% (100% = velmi hladký provoz)"
            )
        
        with col_sm_3:
            st.metric(
                "Avg Runtime",
                f"{smoothness['avg_run_hours']:.1f} h",
                help="Průměrná délka provozu"
            )
        
        with col_sm_4:
            st.metric(
                "Min Runtime",
                f"{smoothness['min_run_hours']:.0f} h",
                help="Nejkratší běh"
            )
        
        with col_sm_5:
            st.metric(
                "Max Runtime",
                f"{smoothness['max_run_hours']:.0f} h",
                help="Nejdelší běh"
            )
        
        # Pokud máme scenario results, zobraz ALL metriky normálně
        total_profit     = result['total_profit']
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
        
        st.markdown("#### 📊 Klíčové Metriky")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Celkový zisk", f"{total_profit:,.0f} €")
        m2.metric("Shortfall", f"{total_shortfall:,.1f} MWh")
        m3.metric("Pokrytí poptávky", f"{coverage:.1f} %")
        m4.metric("Export EE", f"{res['EE export [MW]'].sum():,.1f} MWh")
        m5.metric("Výroba EE", f"{total_ee_gen:,.1f} MWh")
        m6.metric("Provozní hodiny KGJ", f"{kgj_hours:,} h")
        
        st.markdown("#### 💰 Rozpad Zisku")
        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("🔥 Příjmy teplo", f"{rev_teplo_total:,.0f} €")
        r2.metric("⚡ Příjmy EE", f"{rev_ee_total:,.0f} €")
        r3.metric("🔴 Nákl plyn", f"{c_gas_total:,.0f} €")
        r4.metric("🔴 Nákl EE", f"{c_ee_total:,.0f} €")
        r5.metric("🔴 Nákl import", f"{c_imp_total:,.0f} €")
        r6.metric("🔴 Ostatní", f"{c_other_total:,.0f} €")
        
        # ── Grafy pro vybraný profil ──
        st.markdown("#### 🔥 Pokrytí Tepelné Poptávky")
        fig = go.Figure()
        for col, name, color in [
            ('KGJ [MW_th]',          'KGJ',         '#27ae60'),
            ('Kotel [MW_th]',        'Kotel',        '#3498db'),
            ('Elektrokotel [MW_th]', 'Elektrokotel', '#9b59b6'),
            ('Import tepla [MW_th]', 'Import tepla', '#e74c3c'),
            ('TES netto [MW_th]',    'TES netto',    '#f39c12'),
        ]:
            if col in res.columns:
                fig.add_trace(go.Scatter(x=res['Čas'], y=res[col].clip(lower=0),
                    name=name, stackgroup='teplo', fillcolor=color, line_width=0))
        fig.add_trace(go.Scatter(x=res['Čas'], y=res['Shortfall [MW]'],
            name='Nedodáno ⚠️', stackgroup='teplo', fillcolor='rgba(200,0,0,0.45)', line_width=0))
        fig.add_trace(go.Scatter(x=res['Čas'], y=res['Poptávka tepla [MW]']*p['h_cover'],
            name='Cílová poptávka', mode='lines', line=dict(color='black', width=2, dash='dot')))
        fig.update_layout(height=450, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### ⚡ Bilance Elektřiny")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.5, 0.5], subplot_titles=("Zdroje EE [MW]", "Spotřeba / export EE [MW]"))
        for col, name, color in [
            ('EE z KGJ [MW]',      'KGJ',       '#2ecc71'),
            ('EE z FVE [MW]',      'FVE',        '#f1c40f'),
            ('EE import [MW]',     'Import EE',  '#2980b9'),
            ('BESS vybíjení [MW]', 'BESS výdej', '#8e44ad'),
        ]:
            if col in res.columns:
                fig.add_trace(go.Scatter(x=res['Čas'], y=res[col], name=name,
                    stackgroup='vyroba', fillcolor=color), row=1, col=1)
        for col, name, color in [
            ('EE do EK [MW]',      'EK',            '#e74c3c'),
            ('BESS nabíjení [MW]', 'BESS nabíjení', '#34495e'),
            ('EE export [MW]',     'Export EE',     '#16a085'),
        ]:
            if col in res.columns:
                fig.add_trace(go.Scatter(x=res['Čas'], y=-res[col], name=name,
                    stackgroup='spotreba', fillcolor=color), row=2, col=1)
        fig.update_layout(height=600, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 💰 Kumulativní Zisk v Čase")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['Čas'], y=res['Kumulativní zisk [€]'],
            fill='tozeroy', fillcolor='rgba(39,174,96,0.2)', line_color='#27ae60', name='Kum. zisk'))
        fig.update_layout(height=350, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────
# SINGLE RESULT VIEW (když není scenario mode)
# ────────────────────────────────────────────────
if st.session_state.results is not None and st.session_state.scenario_results is None:
    result  = st.session_state.results
    df      = st.session_state.df_main
    res     = result['res']
    uses    = st.session_state.get('uses', dict(kgj=use_kgj, boil=use_boil, ek=use_ek,
                tes=use_tes, bess=use_bess, fve=use_fve, ext_heat=use_ext_heat))
    T       = len(df)
    boil_eff= p.get('boil_eff', 0.95)

    st.divider()
    st.subheader("📊 Detailní Analýza Výsledků")

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
    # VŠECHNY OSTATNÍ GRAFY (původní obsah)
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

    st.subheader("💰 Kumulativní zisk")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['Čas'], y=res['Kumulativní zisk [€]'],
        fill='tozeroy', fillcolor='rgba(39,174,96,0.2)', line_color='#27ae60', name='Kum. zisk'))
    fig.update_layout(height=360, title="Průběh kumulativního zisku v čase")
    st.plotly_chart(fig, use_container_width=True)

    # Monthly Analysis
    st.subheader("📅 Měsíční analýza")
    monthly = res.groupby('Měsíc').agg(
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
    
    st.dataframe(monthly[['Měsíc_str', 'teplo_kgj', 'ee_export', 'zisk', 'kgj_h']], 
                 use_container_width=True, hide_index=True)

    # Heatmapa zisku
    st.subheader("🗓️ Heatmapa hodinového zisku")
    res_hm = res.copy()
    res_hm['Den']    = pd.to_datetime(res_hm['Čas']).dt.dayofyear
    res_hm['Hodina'] = pd.to_datetime(res_hm['Čas']).dt.hour
    pivot = res_hm.pivot_table(index='Hodina', columns='Den',
        values='Hodinový zisk [€]', aggfunc='sum')
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale='RdYlGn', colorbar=dict(title='€/hod'), zmid=0))
    fig.update_layout(height=420, xaxis_title="Den v roce", yaxis_title="Hodina dne")
    st.plotly_chart(fig, use_container_width=True)

    # Excel Export
    st.divider()
    st.subheader("⬇️ Export výsledků")
    
    def to_excel(df_res, df_input, params, monthly_df):
        buf = io.BytesIO()
        skip_cols = {'Měsíc','Hodina dne','KGJ on','Kotel on','Import tepla on'}
        df_exp    = df_res[[c for c in df_res.columns if c not in skip_cols]].copy()
        
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            df_exp.to_excel(writer, index=False, sheet_name='Hodinová data')
            monthly_df.to_excel(writer, index=False, sheet_name='Měsíční souhrn')
            df_input.to_excel(writer, index=False, sheet_name='Vstupní data')
        
        return buf.getvalue()
    
    xlsx = to_excel(res.round(4), df, p, monthly)
    st.download_button(
        label="📥 Stáhnout výsledky (Excel .xlsx)",
        data=xlsx,
        file_name="kgj_optimalizace.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
