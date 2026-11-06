import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Repository root
ROOT_DIR = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Output directory
FIGURES_DIR = ROOT_DIR / "results" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Global style ──────────────────────────────────────────────────────────────
DARK_BG   = "#0A0F1E"
CARD_BG   = "#141B2E"
CARD2_BG  = "#1E2A45"
TEXT      = "#E8EDF5"
MUTED     = "#8A9BB5"
GOLD      = "#C9A84C"
BLUE      = "#4A7FA5"
RED       = "#C0392B"
GREEN     = "#2E7D52"
AMBER     = "#D4B483"

PALETTE   = [BLUE, GOLD, RED, GREEN, AMBER, "#7B68EE"]

sns.set_theme(style="dark", rc={
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    CARD2_BG,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "text.color":        TEXT,
    "grid.color":        CARD2_BG,
    "grid.linewidth":    0.7,
    "legend.facecolor":  CARD_BG,
    "legend.edgecolor":  CARD2_BG,
    "legend.fontsize":   9,
    "font.family":       "DejaVu Sans",
    "figure.dpi":        130,
    "savefig.dpi":       160,
    "savefig.facecolor": DARK_BG,
    "savefig.bbox":      "tight",
})

def gold_title(ax, title, pad=10):
    ax.set_title(title, color=GOLD, pad=pad, fontsize=12, fontweight="bold")

def save(name):
    plt.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    plt.close()
    print(f"  saved figures/{name}.png") 


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD THE ALREADY-CLEANED DATASETS
# ═══════════════════════════════════════════════════════════════════════════════
# Your screenshot shows these three files inside Downloads\cleaned, so this
# version reads them directly and does not require the original raw CSV files.

print("\n── Loading cleaned datasets ────────────────────────────────────────────")

pvdaq_path = PROCESSED_DIR / "PVDAQ_2107_clean.csv"
sim_path   = PROCESSED_DIR / "PV_simulation_clean.csv"
res_path   = PROCESSED_DIR / "PVDAQ_residuals_clean.csv"

for path in (pvdaq_path, sim_path, res_path):
    if not path.exists():
        raise FileNotFoundError(
    f"Could not find {path}. Check that the file exists in data/processed/."
)
        

D = pd.read_csv(pvdaq_path, parse_dates=["timestamp"], index_col="timestamp")
SIM = pd.read_csv(sim_path)
RES = pd.read_csv(res_path, parse_dates=["timestamp"], index_col="timestamp")

# Ensure date-derived features exist (the cleaned files normally already contain them).
if "month" not in D.columns:
    D["month"] = D.index.month
if "hour" not in D.columns:
    D["hour"] = D.index.hour
if "year" not in D.columns:
    D["year"] = D.index.year
if "season" not in D.columns:
    D["season"] = D["month"].map({
        12:"Winter", 1:"Winter", 2:"Winter", 3:"Spring", 4:"Spring", 5:"Spring",
        6:"Summer", 7:"Summer", 8:"Summer", 9:"Autumn", 10:"Autumn", 11:"Autumn"
    })
if "g_regime" not in D.columns:
    D["g_regime"] = pd.cut(D["poa_irradiance_wm2"], bins=[0,200,500,800,1400],
                            labels=["Low\n(0–200)", "Medium\n(200–500)",
                                    "High\n(500–800)", "Very high\n(800+)"])
if "t_regime" not in D.columns:
    D["t_regime"] = pd.cut(D["ambient_temperature_c"], bins=[-15,10,20,30,45,60],
                            labels=["Cold\n(<10°C)", "Cool\n(10–20°C)",
                                    "Mild\n(20–30°C)", "Warm\n(30–45°C)", "Hot\n(>45°C)"])
if "shading_pct" not in SIM.columns and "shading_fraction" in SIM.columns:
    SIM["shading_pct"] = (SIM["shading_fraction"] * 100).astype(int)
if "irr_label" not in SIM.columns and "irradiance_wm2" in SIM.columns:
    SIM["irr_label"] = SIM["irradiance_wm2"].astype(str) + " W/m²"
if "efficiency_loss_pct" not in SIM.columns:
    SIM["efficiency_loss_pct"] = (1 - SIM["p_max_global_w"] /
        SIM.groupby(["irradiance_wm2", "temperature_c"])["p_max_global_w"].transform("max")) * 100
if "delta_p_pct" not in RES.columns:
    RES["delta_p_pct"] = 100 * RES["delta_p_kw"] / RES["dc_power_kw"].clip(lower=1)
if "month" not in RES.columns:
    RES["month"] = RES.index.month
if "season" not in RES.columns:
    RES["season"] = RES["month"].map({
        12:"Winter", 1:"Winter", 2:"Winter", 3:"Spring", 4:"Spring", 5:"Spring",
        6:"Summer", 7:"Summer", 8:"Summer", 9:"Autumn", 10:"Autumn", 11:"Autumn"
    })

# The original raw-data cleaning history is not available from cleaned files alone.
# Keep Figure 2 useful by showing the number of PVDAQ rows loaded.
steps = {"Rows in cleaned dataset": len(D)}

print(f"  PVDAQ 2107:       {len(D):,} rows")
print(f"  PV Simulation:    {len(SIM):,} rows")
print(f"  PVDAQ Residuals:  {len(RES):,} rows")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — DATASET OVERVIEW (3 panels)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Generating figures ──────────────────────────────────────────────────")
print("  [1/9] Dataset overview")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Dataset Overview — NREL PVDAQ 2107  &  PV Simulation",
             color=GOLD, fontsize=14, fontweight="bold", y=1.02)

# 1a. PVDAQ: annual available rows
ax = axes[0]
ann = D.groupby("year").size().reset_index(name="rows")
sns.barplot(data=ann, x="year", y="rows", ax=ax,
            palette=[BLUE if y < 2024 else GOLD for y in ann["year"]])
gold_title(ax, "PVDAQ 2107 — Data Coverage by Year")
ax.set_xlabel("Year"); ax.set_ylabel("Daytime rows")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x/1000)}k"))
ax.tick_params(axis='x', rotation=45)

# Annotate source
for bar, yr in zip(ax.patches, ann["year"]):
    label = "AC inv." if yr < 2024 else "DC meter"
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
            label, ha="center", fontsize=7, color=MUTED, rotation=70)

# 1b. PVDAQ: irradiance vs power scatter coloured by temperature
ax = axes[1]
samp = D.sample(min(6000, len(D)), random_state=42)
sc = ax.scatter(samp["poa_irradiance_wm2"], samp["pmax_kw"],
                c=samp["ambient_temperature_c"], cmap="plasma",
                s=4, alpha=0.45, rasterized=True)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Ambient T (°C)", color=TEXT)
cbar.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)
gold_title(ax, "Irradiance vs Power (coloured by Temperature)")
ax.set_xlabel("POA Irradiance (W/m²)")
ax.set_ylabel("Observed Power (kW)")

# 1c. Simulation: Pmax vs shading fraction (heatmap-style line)
ax = axes[2]
for irr, grp in SIM[SIM["temperature_c"]==35].groupby("irradiance_wm2"):
    grp_full = grp[grp["shading_fraction"]<=0.5].sort_values("shading_fraction")
    alpha = 0.3 + 0.7*(irr - 500)/500
    ax.plot(grp_full["shading_pct"], grp_full["p_max_global_w"],
            color=BLUE, alpha=alpha, lw=1.5)
# Bold 1000 W/m² line
top = SIM[(SIM["irradiance_wm2"]==1000)&(SIM["temperature_c"]==35)].sort_values("shading_fraction")
ax.plot(top["shading_pct"], top["p_max_global_w"], color=GOLD, lw=2.5, label="G = 1000 W/m²")
ax.plot(SIM[(SIM["irradiance_wm2"]==500)&(SIM["temperature_c"]==35)].sort_values("shading_fraction")["shading_pct"],
        SIM[(SIM["irradiance_wm2"]==500)&(SIM["temperature_c"]==35)].sort_values("shading_fraction")["p_max_global_w"],
        color=AMBER, lw=2.5, label="G = 500 W/m²")
ax.legend()
gold_title(ax, "Simulation: Pmax vs Shading  (T = 35°C)")
ax.set_xlabel("Shading fraction (%)")
ax.set_ylabel("Pmax (W)")

plt.tight_layout()
save("fig1_dataset_overview")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — PVDAQ CLEANING WATERFALL
# ═══════════════════════════════════════════════════════════════════════════════
print("  [2/9] Cleaned dataset size")

fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle("NREL PVDAQ 2107 — Cleaned Dataset Size",
             color=GOLD, fontsize=13, fontweight="bold")

labels = list(steps.keys())
values = list(steps.values())
colors = [BLUE] + [GREEN if i > 0 else BLUE for i in range(len(values)-1)]
bars   = ax.barh(range(len(labels)), values, color=colors, edgecolor="none", alpha=0.85)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x/1000)}k"))
ax.set_xlabel("Row count")
ax.invert_yaxis()
for bar, val in zip(bars, values):
    ax.text(bar.get_width()+500, bar.get_y()+bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9, color=TEXT)
plt.tight_layout()
save("fig2_cleaning_waterfall")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — PVDAQ DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("  [3/9] PVDAQ distributions")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("NREL PVDAQ 2107 — Feature Distributions (cleaned)",
             color=GOLD, fontsize=13, fontweight="bold")

# Irradiance
ax = axes[0, 0]
sns.histplot(D["poa_irradiance_wm2"], bins=60, color=BLUE, ax=ax, edgecolor="none")
ax.axvline(D["poa_irradiance_wm2"].mean(), color=GOLD, lw=1.5, ls="--",
           label=f"Mean {D['poa_irradiance_wm2'].mean():.0f} W/m²")
ax.legend(); gold_title(ax, "Irradiance Distribution")
ax.set_xlabel("POA Irradiance (W/m²)"); ax.set_ylabel("Count")

# Temperature
ax = axes[0, 1]
sns.histplot(D["ambient_temperature_c"], bins=55, color=RED, ax=ax, edgecolor="none")
ax.axvline(D["ambient_temperature_c"].mean(), color=GOLD, lw=1.5, ls="--",
           label=f"Mean {D['ambient_temperature_c'].mean():.1f}°C")
ax.legend(); gold_title(ax, "Temperature Distribution")
ax.set_xlabel("Ambient Temperature (°C)"); ax.set_ylabel("Count")

# Power
ax = axes[0, 2]
sns.histplot(D["pmax_kw"], bins=60, color=GREEN, ax=ax, edgecolor="none")
ax.axvline(D["pmax_kw"].median(), color=GOLD, lw=1.5, ls="--",
           label=f"Median {D['pmax_kw'].median():.0f} kW")
ax.legend(); gold_title(ax, "Observed Power Distribution")
ax.set_xlabel("Pmax (kW)"); ax.set_ylabel("Count")

# Monthly mean power
ax = axes[1, 0]
monthly = D.groupby("month")["pmax_kw"].mean()
bars = ax.bar(monthly.index, monthly.values,
              color=[GOLD if m in [4,5,6,7,8] else BLUE for m in monthly.index],
              edgecolor="none", alpha=0.9)
ax.set_xticks(range(1,13))
ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
gold_title(ax, "Monthly Average Power")
ax.set_xlabel("Month"); ax.set_ylabel("Mean Power (kW)")

# Diurnal profile by season
ax = axes[1, 1]
s_pal = {"Spring": GREEN, "Summer": GOLD, "Autumn": AMBER, "Winter": BLUE}
for season, grp in D.groupby("season"):
    h = grp.groupby("hour")["pmax_kw"].mean()
    ax.plot(h.index, h.values, label=season, color=s_pal.get(season, MUTED), lw=2)
ax.legend(); gold_title(ax, "Diurnal Power Profile by Season")
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Mean Power (kW)")

# Wind speed vs power
ax = axes[1, 2]
wsamp = D.dropna(subset=["wind_speed_ms"]).sample(min(4000, len(D)), random_state=1)
sc = ax.scatter(wsamp["wind_speed_ms"], wsamp["pmax_kw"],
                c=wsamp["poa_irradiance_wm2"], cmap="YlOrRd",
                s=5, alpha=0.4, rasterized=True)
plt.colorbar(sc, ax=ax, label="G (W/m²)")
gold_title(ax, "Wind Speed vs Power")
ax.set_xlabel("Wind Speed (m/s)"); ax.set_ylabel("Power (kW)")

plt.tight_layout()
save("fig3_pvdaq_distributions")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — PVDAQ RESIDUAL ANALYSIS (RQ1)
# ═══════════════════════════════════════════════════════════════════════════════
print("  [4/9] Residual analysis (RQ1)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("RQ1 — Where Does the Physics Model Fail?  (NREL PVDAQ 2107)",
             color=GOLD, fontsize=13, fontweight="bold")

# 4a. Predicted vs measured
ax = axes[0, 0]
samp = RES.sample(min(5000, len(RES)), random_state=42)
sc = ax.scatter(samp["p_physics_kw"], samp["dc_power_kw"],
                c=samp["poa_irradiance_wm2"], cmap="YlOrRd",
                s=5, alpha=0.4, rasterized=True)
lim = [0, max(RES["p_physics_kw"].max(), RES["dc_power_kw"].max()) * 1.05]
ax.plot(lim, lim, "w--", lw=1, alpha=0.6, label="Perfect fit")
ax.set_xlim(lim); ax.set_ylim(lim)
plt.colorbar(sc, ax=ax, label="G (W/m²)")
r2 = 1 - ((RES["delta_p_kw"]**2).sum() /
           ((RES["dc_power_kw"] - RES["dc_power_kw"].mean())**2).sum())
gold_title(ax, f"Predicted vs Measured  (R² = {r2:.3f})")
ax.set_xlabel("P_physics (kW)"); ax.set_ylabel("P_measured (kW)")
ax.legend()

# 4b. ΔP vs irradiance with binned mean
ax = axes[0, 1]
ax.scatter(samp["poa_irradiance_wm2"], samp["delta_p_kw"],
           c=samp["ambient_temperature_c"], cmap="coolwarm",
           s=5, alpha=0.35, rasterized=True)
ax.axhline(0, color="white", lw=1, ls="--", alpha=0.5)
bins = np.arange(0, 1300, 80)
RES["g_bin"] = pd.cut(RES["poa_irradiance_wm2"], bins=bins, labels=bins[:-1]+40)
bmean = RES.groupby("g_bin", observed=True)["delta_p_kw"].mean()
ax.plot(bmean.index.astype(float), bmean.values,
        color=GOLD, lw=2.5, marker="o", ms=4, label="Bin mean")
ax.legend()
gold_title(ax, "Residual ΔP vs Irradiance")
ax.set_xlabel("POA Irradiance (W/m²)"); ax.set_ylabel("ΔP (kW)")

# 4c. ΔP boxplot by irradiance regime
ax = axes[1, 0]
order = ["Low\n(0–200)", "Medium\n(200–500)", "High\n(500–800)", "Very high\n(800+)"]
RES["g_regime"] = pd.cut(RES["poa_irradiance_wm2"],
                          bins=[0,200,500,800,1400],
                          labels=order)
sns.boxplot(data=RES, x="g_regime", y="delta_p_kw", order=order,
            palette=[BLUE, BLUE, GOLD, RED], ax=ax,
            flierprops=dict(marker=".", color=MUTED, alpha=0.3, markersize=2),
            width=0.55)
ax.axhline(0, color="white", lw=1, ls="--", alpha=0.5)
gold_title(ax, "ΔP Distribution by Irradiance Regime")
ax.set_xlabel("Irradiance regime"); ax.set_ylabel("ΔP (kW)")

# 4d. ΔP boxplot by season
ax = axes[1, 1]
s_order = ["Winter","Spring","Summer","Autumn"]
sns.boxplot(data=RES, x="season", y="delta_p_kw", order=s_order,
            palette=[BLUE, GREEN, GOLD, AMBER], ax=ax,
            flierprops=dict(marker=".", color=MUTED, alpha=0.3, markersize=2),
            width=0.55)
ax.axhline(0, color="white", lw=1, ls="--", alpha=0.5)
gold_title(ax, "ΔP Distribution by Season")
ax.set_xlabel("Season"); ax.set_ylabel("ΔP (kW)")

plt.tight_layout()
save("fig4_residual_analysis")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — ERROR HEATMAPS  G × T
# ═══════════════════════════════════════════════════════════════════════════════
print("  [5/9] G×T error heatmaps")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Regime-Based Residual Analysis  (RQ1) — |ΔP| across G × T space",
             color=GOLD, fontsize=13, fontweight="bold")

g_edges = np.arange(0, 1300, 100)
t_edges = np.arange(0, 45,  5)
RES["g_bin_c"] = pd.cut(RES["poa_irradiance_wm2"], bins=g_edges, labels=g_edges[:-1]+50)
RES["t_bin_c"] = pd.cut(RES["ambient_temperature_c"], bins=t_edges, labels=t_edges[:-1]+2.5)

pivot_mae  = RES.pivot_table("delta_p_kw", "t_bin_c", "g_bin_c",
                              aggfunc=lambda x: x.abs().mean())
pivot_mean = RES.pivot_table("delta_p_kw", "t_bin_c", "g_bin_c", aggfunc="mean")
pivot_n    = RES.pivot_table("delta_p_kw", "t_bin_c", "g_bin_c", aggfunc="count")

for ax, data, cmap, label, title in [
    (axes[0], pivot_mae,  "YlOrRd", "|ΔP| MAE (kW)",    "|ΔP| across G × T"),
    (axes[1], pivot_mean, "RdBu_r", "Mean ΔP (kW)",     "Mean ΔP  (blue = SDM over-predicts)"),
    (axes[2], pivot_n,    "Blues",  "Sample count",      "Data density"),
]:
    mask = data.isna()
    sns.heatmap(data.astype(float), ax=ax, cmap=cmap, mask=mask,
                cbar_kws={"label": label, "shrink": 0.8},
                linewidths=0.4, linecolor=DARK_BG,
                annot=True if data is pivot_n else False,
                fmt=".0f" if data is pivot_n else "",
                annot_kws={"size": 6})
    gold_title(ax, title, pad=6)
    ax.set_xlabel("Irradiance bin centre (W/m²)", fontsize=10)
    ax.set_ylabel("Temperature bin centre (°C)", fontsize=10)

plt.tight_layout()
save("fig5_error_heatmaps")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — SIMULATION: SHADING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("  [6/9] Simulation shading analysis")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("PV Simulation Dataset — Shading Effect on Module Performance",
             color=GOLD, fontsize=13, fontweight="bold")

# 6a. Pmax vs temperature per shading level (G=1000)
ax = axes[0, 0]
sub = SIM[SIM["irradiance_wm2"] == 1000].sort_values("temperature_c")
shade_pal = dict(zip([0,10,20,30,40,50], [BLUE, GREEN, GOLD, AMBER, RED, "#7B68EE"]))
for sf, grp in sub.groupby("shading_pct"):
    ax.plot(grp["temperature_c"], grp["p_max_global_w"],
            color=shade_pal.get(sf, MUTED), lw=2,
            label=f"{sf}% shading")
ax.legend(fontsize=8, ncol=2)
gold_title(ax, "Pmax vs Temperature  (G = 1000 W/m²)")
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Pmax (W)")

# 6b. Physics model error vs shading
ax = axes[0, 1]
err_by_shade = SIM.groupby("shading_pct")["delta_p_w"].agg(
    mae=lambda x: x.abs().mean(), mean="mean").reset_index()
ax.bar(err_by_shade["shading_pct"], err_by_shade["mae"],
       color=[shade_pal.get(s, BLUE) for s in err_by_shade["shading_pct"]],
       edgecolor="none", alpha=0.85, width=7, label="MAE |ΔP|")
ax.plot(err_by_shade["shading_pct"], err_by_shade["mean"],
        color=GOLD, marker="o", lw=2, ms=6, label="Mean ΔP")
ax.axhline(0, color="white", lw=1, ls="--", alpha=0.5)
ax.legend()
gold_title(ax, "Physics Model Error vs Shading Level")
ax.set_xlabel("Shading fraction (%)"); ax.set_ylabel("Error (W)")

# 6c. Normalised Pmax loss vs shading for each irradiance
ax = axes[1, 0]
t35 = SIM[SIM["temperature_c"]==35].copy()
unshaded = t35[t35["shading_fraction"]==0][["irradiance_wm2","p_max_global_w"]].rename(
    columns={"p_max_global_w":"p_unshaded"})
t35 = t35.merge(unshaded, on="irradiance_wm2", how="left")
t35["pmax_norm"] = t35["p_max_global_w"] / t35["p_unshaded"]

for irr, grp in t35.groupby("irradiance_wm2"):
    alpha = 0.3 + 0.7*(irr-500)/500
    ax.plot(grp["shading_pct"], grp["pmax_norm"]*100,
            color=BLUE, alpha=alpha, lw=1.5)
# Highlight 1000 W/m²
g1000 = t35[t35["irradiance_wm2"]==1000].sort_values("shading_pct")
ax.plot(g1000["shading_pct"], g1000["pmax_norm"]*100,
        color=GOLD, lw=2.5, label="G = 1000 W/m²")
ax.set_xlabel("Shading fraction (%)"); ax.set_ylabel("Normalised Pmax (%)")
gold_title(ax, "Pmax Loss vs Shading  (T = 35°C, all irradiances)")
ax.legend(); ax.set_ylim(0, 105)

# 6d. Secondary peak vs global peak power (PS only)
ax = axes[1, 1]
ps = SIM[SIM["p_max_local_w"].notna()].copy()
ax.scatter(ps["p_max_global_w"], ps["p_max_local_w"],
           c=ps["shading_pct"], cmap="YlOrRd", s=15, alpha=0.7)
lim2 = [0, ps[["p_max_global_w","p_max_local_w"]].max().max()*1.05]
ax.plot(lim2, lim2, "w--", lw=1, alpha=0.5, label="Equal peaks")
sm = plt.cm.ScalarMappable(cmap="YlOrRd",
     norm=plt.Normalize(vmin=ps["shading_pct"].min(), vmax=ps["shading_pct"].max()))
cbar = plt.colorbar(sm, ax=ax); cbar.set_label("Shading %")
ax.legend()
gold_title(ax, "Global vs Secondary P-V Peak\n(partial shading multi-peak behaviour)")
ax.set_xlabel("Global Pmax (W)"); ax.set_ylabel("Secondary peak power (W)")

plt.tight_layout()
save("fig6_simulation_shading")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — CROSS-DATASET: SHADING ERROR COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("  [7/9] Cross-dataset comparison")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Cross-Dataset Comparison — Physics Model Residual Structure",
             color=GOLD, fontsize=13, fontweight="bold")

# 7a. Simulation: ΔP % by shading — violin
ax = axes[0]
SIM["shading_label"] = SIM["shading_pct"].astype(str) + "%"
order_s = ["0%","10%","20%","30%","40%","50%"]
sns.violinplot(data=SIM, x="shading_label", y="delta_p_pct",
               order=order_s, palette=list(shade_pal.values()),
               inner="box", ax=ax, cut=0, linewidth=0.8)
ax.axhline(0, color="white", lw=1.2, ls="--", alpha=0.6)
gold_title(ax, "Simulation: SDM ΔP % by Shading Level")
ax.set_xlabel("Shading fraction"); ax.set_ylabel("ΔP / P_ref (%)")

# 7b. PVDAQ: ΔP distribution by irradiance regime — violin
ax = axes[1]
regime_order = ["Low\n(0–200)","Medium\n(200–500)","High\n(500–800)","Very high\n(800+)"]
sns.violinplot(data=RES.dropna(subset=["g_regime"]),
               x="g_regime", y="delta_p_kw",
               order=regime_order, palette=[BLUE, BLUE, GOLD, RED],
               inner="box", ax=ax, cut=0, linewidth=0.8)
ax.axhline(0, color="white", lw=1.2, ls="--", alpha=0.6)
gold_title(ax, "PVDAQ Farm: SDM ΔP (kW) by Irradiance Regime")
ax.set_xlabel("Irradiance regime"); ax.set_ylabel("ΔP (kW)")

plt.tight_layout()
save("fig7_cross_dataset")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — PVDAQ TIME SERIES & LONG-TERM TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
print("  [8/9] Long-term trends")

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle("NREL PVDAQ 2107 — Long-Term Trends (2017–2025)",
             color=GOLD, fontsize=13, fontweight="bold")

# 8a. Annual median power vs irradiance ratio (proxy for system degradation)
ax = axes[0, 0]
D["pr_simple"] = D["pmax_kw"] / (D["poa_irradiance_wm2"] / 1000).clip(lower=0.05)
ann_pr = D.groupby("year")["pr_simple"].median().reset_index()
sns.lineplot(data=ann_pr, x="year", y="pr_simple", ax=ax,
             color=GOLD, marker="o", ms=7, lw=2)
ax.fill_between(ann_pr["year"], ann_pr["pr_simple"]*0.98, ann_pr["pr_simple"]*1.02,
                alpha=0.15, color=GOLD)
gold_title(ax, "Annual Median Pmax / G Ratio\n(proxy for degradation trend)")
ax.set_xlabel("Year"); ax.set_ylabel("kW per kW/m²")

# 8b. Monthly mean irradiance by year
ax = axes[0, 1]
pvt = D.pivot_table("poa_irradiance_wm2", "month", "year", aggfunc="mean")
sns.heatmap(pvt, ax=ax, cmap="YlOrRd",
            linewidths=0.3, linecolor=DARK_BG,
            cbar_kws={"label": "Mean G (W/m²)", "shrink": 0.8})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_yticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], rotation=0)
gold_title(ax, "Monthly Mean Irradiance by Year")
ax.set_xlabel("Year"); ax.set_ylabel("Month")

# 8c. Temperature distribution shift over years
ax = axes[1, 0]
years_avail = sorted(D["year"].unique())
cmap_yr = plt.cm.plasma
norm_yr  = plt.Normalize(vmin=min(years_avail), vmax=max(years_avail))
for yr in years_avail:
    sub_yr = D[D["year"]==yr]["ambient_temperature_c"]
    if len(sub_yr) < 50: continue
    sub_yr.plot.kde(ax=ax, color=cmap_yr(norm_yr(yr)), lw=1.5, label=str(yr))
ax.legend(fontsize=8, ncol=2)
ax.set_xlim(-10, 55)
gold_title(ax, "Temperature Distribution by Year")
ax.set_xlabel("Ambient Temperature (°C)"); ax.set_ylabel("Density")

# 8d. System efficiency over time (monthly median, 2017-2023 AC data)
ax = axes[1, 1]
eff = D[D["system_efficiency_pct"].notna()].copy()
if len(eff) > 100:
    eff["ym"] = eff.index.to_period("M")
    eff_m = eff.groupby("ym")["system_efficiency_pct"].median().reset_index()
    eff_m["ym_dt"] = eff_m["ym"].dt.to_timestamp()
    ax.plot(eff_m["ym_dt"], eff_m["system_efficiency_pct"], color=BLUE, lw=1.2, alpha=0.7)
    ax.axhline(eff_m["system_efficiency_pct"].mean(), color=GOLD, lw=1.5, ls="--",
               label=f"Mean {eff_m['system_efficiency_pct'].mean():.1f}%")
    ax.legend()
    gold_title(ax, "Monthly Median System Efficiency\n(AC / DC, inverter level)")
    ax.set_xlabel("Date"); ax.set_ylabel("Efficiency (%)")
    ax.set_ylim(50, 105)
else:
    ax.text(0.5, 0.5, "Insufficient efficiency\ndata for this period",
            ha="center", va="center", transform=ax.transAxes, color=MUTED, fontsize=11)
    gold_title(ax, "System Efficiency (unavailable)")

plt.tight_layout()
save("fig8_long_term_trends")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — SUMMARY METRICS TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("  [9/9] Summary metrics")

mae_pvdaq  = RES["delta_p_kw"].abs().mean()
rmse_pvdaq = np.sqrt((RES["delta_p_kw"]**2).mean())
r2_pvdaq   = 1 - ((RES["delta_p_kw"]**2).sum() /
                   ((RES["dc_power_kw"]-RES["dc_power_kw"].mean())**2).sum())

mae_sim    = SIM["delta_p_w"].abs().mean()
r2_sim     = 1 - ((SIM["delta_p_w"]**2).sum() /
                   ((SIM["p_max_global_w"]-SIM["p_max_global_w"].mean())**2).sum())

fig, ax = plt.subplots(figsize=(13, 4.5))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)
ax.axis("off")
fig.suptitle("Physics Baseline (SDM) — Performance Summary Across Both Datasets",
             color=GOLD, fontsize=13, fontweight="bold", y=0.98)

table_data = [
    ["Metric",             "PVDAQ 2107 (real farm)",       "PV Simulation (controlled)"],
    ["Data type",          "Measured, 15-min intervals",    "Simulated I-V curves"],
    ["Time span",          "2017–2025 (8 years)",           "Static conditions"],
    ["Rows (analysis)",    f"{len(RES):,}",                 f"{len(SIM):,}"],
    ["G range",            "10–1277 W/m²",                  "500–1000 W/m²"],
    ["T range",            "0–41 °C",                       "20–50 °C"],
    ["Shading info",       "Implicit in residual",          "Explicit 0–50%"],
    ["SDM R²",             f"{r2_pvdaq:.4f}",               f"{r2_sim:.4f}"],
    ["SDM MAE",            f"{mae_pvdaq:.1f} kW",           f"{mae_sim:.1f} W"],
    ["SDM RMSE",           f"{rmse_pvdaq:.1f} kW",          "—"],
    ["Key finding",        "Structured ΔP across G×T",     "2.7× more error at 50% shade"],
]

col_w = [0.32, 0.34, 0.34]
col_x = [0.01, 0.33, 0.67]
row_h = 0.085
for r, row in enumerate(table_data):
    for c, (cell, cx, cw) in enumerate(zip(row, col_x, col_w)):
        bg = CARD_BG if r == 0 else (DARK_BG if r % 2 == 0 else CARD2_BG)
        ax.add_patch(plt.Rectangle((cx, 1-(r+1)*row_h), cw-0.005, row_h-0.005,
                                    transform=ax.transAxes, color=bg, zorder=0))
        fc = GOLD if r == 0 else (GOLD if r in [7,8,9,10] else TEXT)
        fw = "bold" if r in [0,10] else "normal"
        ax.text(cx+0.01, 1-(r+0.5)*row_h, cell,
                transform=ax.transAxes, fontsize=10,
                color=fc, fontweight=fw, va="center")

plt.tight_layout()
save("fig9_summary_metrics")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Done ────────────────────────────────────────────────────────────────")
print(f"  Figures saved in  {FIGURES_DIR}")
print(f"  Processed data in  {PROCESSED_DIR}")
print(f"  PVDAQ clean:      {len(D):,} rows")
print(f"  Simulation clean: {len(SIM):,} rows")
print(f"  Residuals clean:  {len(RES):,} rows")