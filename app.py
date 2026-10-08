"""
Simulateur de rentabilité - Business de sous-location Airbnb (rent arbitrage)
================================================================================
Modèle : vous louez un appartement à un propriétaire (loyer fixe mensuel),
vous l'aménagez, puis vous le relouez en courte durée type Airbnb.
La marge = revenus Airbnb (nuitées + ménage facturé - commission plateforme
- coût réel du ménage) - loyer versé au propriétaire - charges fixes.

Le simulateur fait tourner une simulation Monte Carlo mois par mois, en
tenant compte de la saisonnalité et de la volatilité du taux d'occupation,
pour estimer la distribution des résultats possibles (pas juste un seul
scénario moyen).

Lancer avec :
    pip install streamlit plotly numpy pandas
    streamlit run app.py
"""

import calendar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CURRENCY = "FCFA"
MONTH_NAMES_FR = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre",
]

st.set_page_config(page_title="Simulateur Rentabilité Airbnb", layout="wide")


def fmt(x: float) -> str:
    """Format a number as XXX XXX FCFA."""
    return f"{x:,.0f}".replace(",", " ") + f" {CURRENCY}"


# --------------------------------------------------------------------------
# SIDEBAR - VARIABLES DU MODELE
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Paramètres du business")

with st.sidebar.expander("🏠 Coûts liés au logement", expanded=True):
    loyer_mensuel = st.number_input(
        "Loyer mensuel versé au propriétaire (FCFA)",
        min_value=0, value=150_000, step=5_000)
    caution = st.number_input(
        "Caution versée au propriétaire (one-shot, souvent récupérable)",
        min_value=0, value=300_000, step=10_000)
    cout_amenagement = st.number_input(
        "Aménagement / mobilier / déco (investissement de départ)",
        min_value=0, value=800_000, step=10_000)
    charges_fixes = st.number_input(
        "Charges fixes mensuelles (internet, électricité, assurance, abonnements)",
        min_value=0, value=35_000, step=5_000)

with st.sidebar.expander("💰 Tarification Airbnb", expanded=True):
    prix_nuitee = st.number_input(
        "Prix moyen par nuitée (FCFA)", min_value=0, value=25_000, step=1_000)
    frais_menage_facture = st.number_input(
        "Frais de ménage facturé au voyageur (FCFA / réservation)",
        min_value=0, value=0, step=500)
    duree_sejour = st.number_input(
        "Durée moyenne d'un séjour (nuits)", min_value=1.0, value=1.0, step=0.5)
    commission_plateforme = st.slider(
        "Commission de la plateforme (Airbnb, ~3%)", 0.0, 20.0, 0.0, step=0.5) / 100

with st.sidebar.expander("🧹 Coûts variables", expanded=True):
    cout_menage_reel = st.number_input(
        "Coût réel du ménage par réservation (femme de ménage, linge, consommables)",
        min_value=0, value=4_000, step=500)

with st.sidebar.expander("📅 Occupation & saisonnalité", expanded=True):
    taux_occupation_moyen = st.slider(
        "Taux d'occupation moyen annuel (%)", 0, 100, 55) / 100
    mois_haute_saison = st.multiselect(
        "Mois de haute saison",
        options=MONTH_NAMES_FR,
        default=["Juillet", "Août", "Décembre"])
    mult_haute = st.slider(
        "Multiplicateur haute saison", 1.0, 2.5, 1.4, step=0.05)
    mult_basse = st.slider(
        "Multiplicateur basse saison", 0.3, 1.0, 0.8, step=0.05)
    volatilite = st.slider(
        "Volatilité du taux d'occupation (écart-type, %) — incertitude mois/mois",
        0, 40, 15) / 100

with st.sidebar.expander("🎲 Simulation", expanded=True):
    duree_mois = st.slider("Durée simulée (mois)", 6, 60, 24)
    nb_simulations = st.slider("Nombre de simulations Monte Carlo", 100, 3000, 800, step=100)
    seed = st.number_input("Graine aléatoire (pour reproductibilité)", min_value=0, value=42, step=1)

with st.sidebar.expander("👥 Répartition entre associés (4)", expanded=True):
    mode_parts = st.radio(
        "Comment calculer la part de chaque associé ?",
        ["Apport en capital (prorata)", "Parts fixes (%)"],
    )
    noms_par_defaut = ["Princesse", "Marina", "Stacy", "Ronald"]
    NB_ASSOCIES = len(noms_par_defaut)

    noms_associes = []
    valeurs_associes = []  # apport en FCFA (mode prorata) ou % (mode fixe)
    default_apport = max(int((cout_amenagement + caution) / NB_ASSOCIES), 0)
    default_part = 100.0 / NB_ASSOCIES  # 25 % chacun

    for i, nom_defaut in enumerate(noms_par_defaut, start=1):
        col_nom, col_val = st.columns([1.3, 1])
        nom = col_nom.text_input(f"Nom associé {i}", value=nom_defaut, key=f"nom_{i}")
        if mode_parts == "Apport en capital (prorata)":
            val = col_val.number_input(
                "Apport (FCFA)", min_value=0, value=default_apport, step=10_000, key=f"apport_{i}"
            )
        else:
            val = col_val.number_input(
                "Part (%)", min_value=0.0, max_value=100.0, value=default_part, step=1.0, key=f"part_{i}"
            )
        noms_associes.append(nom)
        valeurs_associes.append(val)

    if mode_parts == "Parts fixes (%)":
        total_saisi = sum(valeurs_associes)
        if abs(total_saisi - 100) > 0.01:
            st.warning(
                f"⚠️ La somme des parts saisies fait {total_saisi:.1f}% (au lieu de 100%). "
                "Les parts seront normalisées automatiquement pour la suite du calcul."
            )

# --------------------------------------------------------------------------
# MOTEUR DE SIMULATION
# --------------------------------------------------------------------------
mois_haute_idx = {MONTH_NAMES_FR.index(m) for m in mois_haute_saison}


def run_single_simulation(rng: np.random.Generator) -> np.ndarray:
    """Retourne le cashflow cumulé mois par mois (longueur duree_mois+1)."""
    cashflow = np.zeros(duree_mois + 1)
    cashflow[0] = -(cout_amenagement + caution)

    for m in range(1, duree_mois + 1):
        month_idx = (m - 1) % 12
        # Jours du mois (approché en cycle non-bissextile pour la simplicité)
        jours_du_mois = calendar.monthrange(2025, month_idx + 1)[1]

        seasonal_mult = mult_haute if month_idx in mois_haute_idx else mult_basse
        base_occ = np.clip(taux_occupation_moyen * seasonal_mult, 0, 1)

        # Bruit aléatoire (Monte Carlo) autour du taux d'occupation attendu
        occ = np.clip(rng.normal(base_occ, volatilite), 0, 1)

        nuits_occupees = occ * jours_du_mois
        nb_reservations = nuits_occupees / duree_sejour if duree_sejour > 0 else 0

        revenu_nuitees = nuits_occupees * prix_nuitee
        revenu_menage = nb_reservations * frais_menage_facture
        revenu_brut = revenu_nuitees + revenu_menage

        commission = revenu_brut * commission_plateforme
        cout_menage_total = nb_reservations * cout_menage_reel

        revenu_net = revenu_brut - commission - cout_menage_total
        profit_mensuel = revenu_net - loyer_mensuel - charges_fixes

        cashflow[m] = cashflow[m - 1] + profit_mensuel

    return cashflow


@st.cache_data(show_spinner=False)
def run_monte_carlo(params_key, n_sims, months, seed):
    rng = np.random.default_rng(seed)
    trajectories = np.zeros((n_sims, months + 1))
    for i in range(n_sims):
        trajectories[i] = run_single_simulation(rng)
    return trajectories


# La clé de cache doit changer dès qu'un paramètre change
params_key = (
    loyer_mensuel, caution, cout_amenagement, charges_fixes, prix_nuitee,
    frais_menage_facture, duree_sejour, commission_plateforme, cout_menage_reel,
    taux_occupation_moyen, tuple(sorted(mois_haute_idx)), mult_haute, mult_basse,
    volatilite, duree_mois, nb_simulations, seed,
)

trajectories = run_monte_carlo(params_key, nb_simulations, duree_mois, seed)

# --------------------------------------------------------------------------
# STATISTIQUES CLES
# --------------------------------------------------------------------------
final_cashflow = trajectories[:, -1]
investissement_initial = cout_amenagement + caution
proba_rentable = float(np.mean(final_cashflow > 0)) * 100

# Mois de retour sur investissement (break-even) par simulation
payback_months = []
for traj in trajectories:
    idx = np.argmax(traj > 0)
    if traj[idx] > 0:
        payback_months.append(idx)
median_payback = int(np.median(payback_months)) if payback_months else None

monthly_profits_all = np.diff(trajectories, axis=1)  # profit mensuel (hors mois 0)
profit_mensuel_moyen = float(np.mean(monthly_profits_all))

roi_median = (np.median(final_cashflow) / investissement_initial) * 100 if investissement_initial else 0

# --------------------------------------------------------------------------
# AFFICHAGE
# --------------------------------------------------------------------------
st.title("🏘️ Simulateur de rentabilité — Sous-location Airbnb")
st.caption(
    "Modèle de rent arbitrage : loyer fixe payé au propriétaire, "
    "revenus variables via Airbnb, simulation Monte Carlo mensuelle."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Investissement initial", fmt(investissement_initial))
c2.metric("Profit mensuel moyen", fmt(profit_mensuel_moyen))
c3.metric("Probabilité d'être rentable", f"{proba_rentable:.0f} %")
c4.metric(
    "Retour sur investissement (médian)",
    f"{median_payback} mois" if median_payback is not None else "Non atteint",
)
c5.metric("ROI médian sur la période", f"{roi_median:.0f} %")

if profit_mensuel_moyen < 0:
    st.error(
        "⚠️ Le profit mensuel moyen est négatif : ce business, avec ces "
        "paramètres, perd de l'argent en moyenne. Ajustez le loyer, le prix "
        "des nuitées ou le taux d'occupation."
    )
elif proba_rentable < 60:
    st.warning(
        "⚠️ Le modèle est rentable en moyenne, mais avec une probabilité "
        "de succès modérée — le résultat dépend beaucoup de la saisonnalité "
        "et de la variabilité de l'occupation."
    )
else:
    st.success("✅ Le modèle semble rentable dans la majorité des scénarios simulés.")

st.divider()

# ---- Graphique en éventail (fan chart) du cashflow cumulé ----
st.subheader("💵 Trésorerie cumulée dans le temps (bande d'incertitude)")

percentiles = [5, 25, 50, 75, 95]
pct_values = {p: np.percentile(trajectories, p, axis=0) for p in percentiles}
months_axis = list(range(duree_mois + 1))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=months_axis + months_axis[::-1],
    y=list(pct_values[95]) + list(pct_values[5])[::-1],
    fill="toself", fillcolor="rgba(99,110,250,0.15)",
    line=dict(color="rgba(255,255,255,0)"), name="Intervalle 5–95 %",
    showlegend=True,
))
fig.add_trace(go.Scatter(
    x=months_axis + months_axis[::-1],
    y=list(pct_values[75]) + list(pct_values[25])[::-1],
    fill="toself", fillcolor="rgba(99,110,250,0.30)",
    line=dict(color="rgba(255,255,255,0)"), name="Intervalle 25–75 %",
    showlegend=True,
))
fig.add_trace(go.Scatter(
    x=months_axis, y=pct_values[50], line=dict(color="#636EFA", width=3),
    name="Médiane",
))
fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Seuil de rentabilité")
fig.update_layout(
    xaxis_title="Mois", yaxis_title=f"Trésorerie cumulée ({CURRENCY})",
    hovermode="x unified", height=450,
)
st.plotly_chart(fig, width='stretch')

# ---- Distribution du résultat final ----
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Distribution du résultat final")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=final_cashflow, nbinsx=40, marker_color="#636EFA"))
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
    fig_hist.update_layout(
        xaxis_title=f"Trésorerie cumulée après {duree_mois} mois ({CURRENCY})",
        yaxis_title="Nombre de simulations", height=350,
    )
    st.plotly_chart(fig_hist, width='stretch')

with col_b:
    st.subheader("🗓️ Saisonnalité appliquée")
    saison_df = pd.DataFrame({
        "Mois": MONTH_NAMES_FR,
        "Type": ["Haute saison" if i in mois_haute_idx else "Basse saison" for i in range(12)],
        "Multiplicateur": [mult_haute if i in mois_haute_idx else mult_basse for i in range(12)],
        "Taux d'occupation cible": [
            f"{min(taux_occupation_moyen * (mult_haute if i in mois_haute_idx else mult_basse), 1) * 100:.0f} %"
            for i in range(12)
        ],
    })
    st.dataframe(saison_df, width='stretch', hide_index=True)

st.divider()

# ---- Détail d'une simulation représentative (la médiane) ----
st.subheader("📋 Détail d'un scénario représentatif (proche de la médiane)")
closest_idx = int(np.argmin(np.abs(final_cashflow - np.median(final_cashflow))))
detail_traj = trajectories[closest_idx]
detail_df = pd.DataFrame({
    "Mois": months_axis,
    "Trésorerie cumulée": detail_traj,
})
detail_df["Profit du mois"] = detail_df["Trésorerie cumulée"].diff().fillna(detail_df["Trésorerie cumulée"].iloc[0])
st.dataframe(
    detail_df.style.format({"Trésorerie cumulée": "{:,.0f}", "Profit du mois": "{:,.0f}"}),
    width='stretch', hide_index=True,
)

csv = detail_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Télécharger ce scénario en CSV", data=csv,
    file_name="simulation_airbnb.csv", mime="text/csv",
)

st.divider()

# --------------------------------------------------------------------------
# REPARTITION ENTRE ASSOCIES
# --------------------------------------------------------------------------
st.subheader("👥 Répartition entre associés")

# Parts de départ (%) selon le mode choisi dans la sidebar
if mode_parts == "Apport en capital (prorata)":
    total_val = sum(valeurs_associes)
    shares_initiales = [v / total_val for v in valeurs_associes] if total_val > 0 else [1 / 5] * 5
    apports_affiches = valeurs_associes
else:
    total_val = sum(valeurs_associes)
    shares_initiales = [v / total_val for v in valeurs_associes] if total_val > 0 else [1 / 5] * 5
    # Investissement implicite de chacun = sa part x investissement initial total
    apports_affiches = [s * investissement_initial for s in shares_initiales]

# ---- Transferts de parts entre associés ----
st.markdown("**🔄 Transferts de parts entre associés (optionnel)**")
st.caption(
    "Ajoutez une ligne par transfert : un associé cède une partie de ses parts à un autre, "
    "à un mois donné. Le montant en FCFA est converti en points de %% sur la base de "
    "l'investissement initial total (%s)." % fmt(investissement_initial)
)

transferts_vides = pd.DataFrame({
    "Mois": pd.Series(dtype="int"),
    "De (cède)": pd.Series(dtype="str"),
    "Vers (reçoit)": pd.Series(dtype="str"),
    "Montant (FCFA)": pd.Series(dtype="int"),
})

transferts_df = st.data_editor(
    transferts_vides,
    num_rows="dynamic",
    width='stretch',
    hide_index=True,
    column_config={
        "Mois": st.column_config.NumberColumn(min_value=0, max_value=duree_mois, step=1),
        "De (cède)": st.column_config.SelectboxColumn(options=noms_associes),
        "Vers (reçoit)": st.column_config.SelectboxColumn(options=noms_associes),
        "Montant (FCFA)": st.column_config.NumberColumn(min_value=0, step=10_000),
    },
    key="transferts_editor",
)

# ---- Construction des parts dans le temps (mois par mois) ----
n = len(noms_associes)
idx_of = {nom: i for i, nom in enumerate(noms_associes)}
# Toutes les colonnes démarrent avec les parts initiales ; les transferts appliquent
# ensuite un décalage cumulatif à partir de leur mois, colonne par colonne.
shares_series = np.tile(np.array(shares_initiales).reshape(-1, 1), (1, duree_mois + 1))

transferts_valides = transferts_df.dropna(subset=["Mois", "De (cède)", "Vers (reçoit)", "Montant (FCFA)"])
transferts_valides = transferts_valides.sort_values("Mois")

avertissements = []
for _, row in transferts_valides.iterrows():
    mois_t = int(row["Mois"])
    de_nom, vers_nom = row["De (cède)"], row["Vers (reçoit)"]
    montant = float(row["Montant (FCFA)"])

    if de_nom == vers_nom:
        avertissements.append(f"Transfert ignoré au mois {mois_t} : « {de_nom} » ne peut pas se céder des parts à lui-même.")
        continue
    if mois_t < 1 or mois_t > duree_mois:
        avertissements.append(f"Transfert ignoré : le mois {mois_t} est hors de la période simulée (1–{duree_mois}).")
        continue

    points = montant / investissement_initial if investissement_initial > 0 else 0
    de_idx, vers_idx = idx_of[de_nom], idx_of[vers_nom]
    disponible = shares_series[de_idx, mois_t - 1]  # part disponible juste avant le transfert

    if points > disponible:
        avertissements.append(
            f"⚠️ Au mois {mois_t}, {de_nom} ne possède que {disponible*100:.1f}% de parts "
            f"(le transfert demandé équivaut à {points*100:.1f}%) — le montant a été plafonné."
        )
        points = disponible

    shares_series[de_idx, mois_t:] -= points
    shares_series[vers_idx, mois_t:] += points

for msg in avertissements:
    st.warning(msg)

# ---- Position de trésorerie de chaque associé, avec parts évolutives ----
monthly_profit_median = np.diff(detail_traj)  # profit du mois m (m = 1..duree_mois), basé sur le scénario médian
positions = np.zeros((n, duree_mois + 1))
positions[:, 0] = shares_series[:, 0] * detail_traj[0]
for m in range(1, duree_mois + 1):
    positions[:, m] = positions[:, m - 1] + shares_series[:, m] * monthly_profit_median[m - 1]

associes_traj = {nom: positions[i] for i, nom in enumerate(noms_associes)}

col_pie, col_evol = st.columns([1, 1.6])

with col_pie:
    st.markdown("**Répartition des parts (mois 0)**")
    fig_pie = go.Figure(data=[go.Pie(
        labels=noms_associes,
        values=shares_series[:, 0],
        hole=0.4,
        texttemplate="%{label}<br>%{percent}",
    )])
    fig_pie.update_layout(height=380, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_pie, width='stretch')

with col_evol:
    st.markdown("**Évolution de la position de trésorerie de chaque associé (scénario médian)**")
    fig_assoc = go.Figure()
    for nom in noms_associes:
        fig_assoc.add_trace(go.Scatter(
            x=months_axis, y=associes_traj[nom], mode="lines", name=nom,
        ))
    fig_assoc.add_hline(y=0, line_dash="dash", line_color="red")
    fig_assoc.update_layout(
        xaxis_title="Mois", yaxis_title=f"Trésorerie cumulée par associé ({CURRENCY})",
        hovermode="x unified", height=380, legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_assoc, width='stretch')

# ---- Matérialisation de l'évolution des PARTS (%) dans le temps ----
st.markdown("**📈 Évolution des parts (%) de chaque associé au fil des mois**")
fig_parts = go.Figure()
for i, nom in enumerate(noms_associes):
    fig_parts.add_trace(go.Scatter(
        x=months_axis, y=shares_series[i] * 100, mode="lines", name=nom,
        stackgroup="parts", line=dict(width=0.5),
    ))
fig_parts.update_layout(
    xaxis_title="Mois", yaxis_title="Part de capital (%)",
    yaxis=dict(range=[0, 100]), hovermode="x unified", height=380,
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig_parts, width='stretch')
if not transferts_valides.empty:
    st.caption("Les marches dans le graphique correspondent aux mois où un transfert de parts a lieu.")

# Tableau récapitulatif final par associé
recap_rows = []
for i, nom in enumerate(noms_associes):
    apport_i = apports_affiches[i]
    position_finale_i = associes_traj[nom][-1]
    part_initiale_i = shares_series[i, 0] * 100
    part_finale_i = shares_series[i, -1] * 100
    roi_i = (position_finale_i / apport_i) * 100 if apport_i > 0 else 0
    recap_rows.append({
        "Associé": nom,
        "Part initiale": f"{part_initiale_i:.1f} %",
        "Part finale": f"{part_finale_i:.1f} %",
        "Apport initial": apport_i,
        "Position finale (mois %d)" % duree_mois: position_finale_i,
        "ROI individuel": f"{roi_i:.0f} %",
    })

recap_df = pd.DataFrame(recap_rows)
st.dataframe(
    recap_df.style.format({
        "Apport initial": "{:,.0f}",
        "Position finale (mois %d)" % duree_mois: "{:,.0f}",
    }),
    width='stretch', hide_index=True,
)

st.caption(
    "La « position finale » de chaque associé cumule sa part des profits mois par mois "
    "(à part variable si des transferts ont eu lieu) et intègre sa part de l'investissement "
    "initial versé au mois 0. Un ROI individuel positif signifie que l'associé a récupéré "
    "plus que sa mise de départ sur la période simulée."
)

csv_associes = recap_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Télécharger la répartition des associés en CSV", data=csv_associes,
    file_name="repartition_associes.csv", mime="text/csv",
)

st.divider()
st.caption(
    "⚠️ Ceci est un outil de simulation indicatif, pas un conseil financier. "
    "Vérifiez la légalité de la sous-location courte durée dans votre contrat "
    "de bail et votre juridiction avant de vous lancer."
)
