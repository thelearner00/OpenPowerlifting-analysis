# ============================================================
# OpenPowerlifting Dataset — Full Analysis Script v3
# Fixes: Analysis 4    (sparse year filter), Analysis 5 (bar labels),
#        Analysis 6 (split chart for readability)
# Added:  Analysis 9 (correlation heatmap)
#
# STRUCTURE:
#   load_data()  — loads, joins, and cleans both CSVs.
#                  Returns (df, valid) for use by other scripts.
#   The rest of the file runs the 9 analyses and saves charts.
#   When imported by another script, only load_data() is called —
#   the analyses do NOT run automatically.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────
# SHARED FUNCTION — called by this script AND ml_model.py
# ─────────────────────────────────────────────────────────────
def load_data(meets_path="meets.csv", opl_path="openpowerlifting.csv", verbose=True):
    """
    Loads, joins, and cleans the OpenPowerlifting dataset.

    Parameters:
        meets_path : path to meets.csv
        opl_path   : path to openpowerlifting.csv
        verbose    : if True, prints progress (set False when importing)

    Returns:
        df    — full merged & cleaned DataFrame (all entries)
        valid — filtered DataFrame (no DQ/NS, TotalKg present)
    """
    def log(msg):
        if verbose:
            print(msg)

    # Load
    meets = pd.read_csv(meets_path)
    opl   = pd.read_csv(opl_path, dtype={"Place": str})
    log(f"meets.csv     : {meets.shape[0]:,} rows × {meets.shape[1]} columns")
    log(f"opl.csv       : {opl.shape[0]:,} rows × {opl.shape[1]} columns")

    # Join
    df = opl.merge(meets, on="MeetID", how="left")
    log(f"Merged shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"]  = df["Date"].dt.year

    # Place flags
    df["Place"]    = df["Place"].str.strip()
    df["is_dq"]    = df["Place"] == "DQ"
    df["is_ns"]    = df["Place"] == "NS"
    df["is_guest"] = df["Place"] == "G"

    # AgeGroup bins
    def assign_age_group(age):
        if pd.isna(age): return "Unknown"
        if age < 18:     return "Sub-Junior (<18)"
        if age < 24:     return "Junior (18-23)"
        if age < 40:     return "Open (24-39)"
        if age < 50:     return "Master 40-49"
        if age < 60:     return "Master 50-59"
        return                  "Master 60+"

    df["AgeGroup"] = df["Age"].apply(assign_age_group)

    # Drop 4th-attempt columns (99%+ null)
    df.drop(columns=["Squat4Kg", "Bench4Kg", "Deadlift4Kg"], inplace=True)

    # Valid entries base
    valid = df[~df["is_dq"] & ~df["is_ns"] & df["TotalKg"].notna()].copy()

    log(f"Valid entries  : {len(valid):,} rows")
    return df, valid


# ─────────────────────────────────────────────────────────────
# MAIN — only runs when this file is executed directly,
#        NOT when imported by ml_model.py
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("STEP 1 — Loading data")
    print("=" * 60)
    df, valid = load_data(verbose=True)

    print("\n" + "=" * 60)
    print("STEP 4 — Data quality snapshot")
    print("=" * 60)
    key_cols = ["Age", "BodyweightKg", "BestSquatKg", "BestBenchKg",
                "BestDeadliftKg", "TotalKg", "Wilks", "Division"]
    null_pct = df[key_cols].isnull().mean().mul(100).round(1)
    print(null_pct.rename("Null %").to_string())

    # ── ANALYSIS 1 — Gender Score Comparison ─────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS 1 — Gender-based score comparison")
    print("=" * 60)

    gender = valid.groupby("Sex").agg(
        Participants   = ("Name",         "count"),
        Avg_TotalKg    = ("TotalKg",      "mean"),
        Avg_Wilks      = ("Wilks",        "mean"),
        Avg_Bodyweight = ("BodyweightKg", "mean")
    ).round(2)
    print(gender.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Gender Comparison", fontsize=13, fontweight="bold")
    axes[0].bar(gender.index, gender["Avg_TotalKg"], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Average Total (kg)")
    axes[0].set_ylabel("kg")
    for i, (idx, row) in enumerate(gender.iterrows()):
        axes[0].text(i, row["Avg_TotalKg"] + 5, f"{row['Avg_TotalKg']:.1f}",
                     ha="center", fontsize=10, fontweight="bold")
    axes[1].bar(gender.index, gender["Avg_Wilks"], color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Average Wilks Score")
    for i, (idx, row) in enumerate(gender.iterrows()):
        axes[1].text(i, row["Avg_Wilks"] + 2, f"{row['Avg_Wilks']:.1f}",
                     ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("analysis_1_gender.png", dpi=120)
    plt.close()
    print("Saved → analysis_1_gender.png")

    # ── ANALYSIS 2 — Equipment Breakdown ─────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS 2 — Equipment breakdown")
    print("=" * 60)

    equip = valid.groupby("Equipment").agg(
        Count       = ("Name",    "count"),
        Avg_TotalKg = ("TotalKg", "mean"),
        Avg_Wilks   = ("Wilks",   "mean")
    ).sort_values("Avg_TotalKg", ascending=False).round(2)
    print(equip.to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(equip.index, equip["Avg_TotalKg"], color="#5A9BD5")
    ax.set_xlabel("Average Total (kg)")
    ax.set_title("Average Total by Equipment Type", fontweight="bold")
    for bar in bars:
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("analysis_2_equipment.png", dpi=120)
    plt.close()
    print("Saved → analysis_2_equipment.png")

    # ── ANALYSIS 3 — DQ Analysis ──────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS 3 — Disqualified athlete analysis")
    print("=" * 60)

    dq_df = df[df["is_dq"]].copy()
    print(f"Total DQ entries   : {len(dq_df):,}")
    print(f"Unique DQ athletes : {dq_df['Name'].nunique():,}")

    repeat_dq = (dq_df.groupby("Name").size()
                 .reset_index(name="DQ_Count")
                 .query("DQ_Count > 1")
                 .sort_values("DQ_Count", ascending=False))
    print(f"Repeat DQ athletes : {len(repeat_dq):,}")
    print("\nTop 10 most DQ'd athletes:")
    print(repeat_dq.head(10).to_string(index=False))

    dq_by_fed = df.groupby("Federation").agg(
        Total = ("Name",  "count"),
        DQs   = ("is_dq", "sum")
    )
    dq_by_fed["DQ_Rate_%"] = (dq_by_fed["DQs"] / dq_by_fed["Total"] * 100).round(2)
    dq_by_fed = dq_by_fed[dq_by_fed["Total"] >= 200]
    top_dq_feds = dq_by_fed.sort_values("DQ_Rate_%", ascending=False).head(15)
    print("\nTop 15 federations by DQ rate (min 200 entries):")
    print(top_dq_feds.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top_dq_feds.index[::-1], top_dq_feds["DQ_Rate_%"][::-1], color="#E05C5C")
    ax.set_xlabel("DQ Rate (%)")
    ax.set_title("Top 15 Federations by DQ Rate", fontweight="bold")
    for bar in bars:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("analysis_3_dq.png", dpi=120)
    plt.close()
    print("Saved → analysis_3_dq.png")

    # ── ANALYSIS 4 — Time Series ──────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS 4 — Time series (meets & participants per year)")
    print("=" * 60)

    ts = valid[valid["Year"] >= 1990].copy()
    yearly = ts.groupby("Year").agg(
        Participants = ("Name",    "count"),
        Avg_TotalKg  = ("TotalKg", "mean"),
        Avg_Wilks    = ("Wilks",   "mean")
    ).round(2)
    meets_per_year = (df[df["Year"] >= 1990]
                      .groupby("Year")["MeetID"].nunique()
                      .rename("Meets"))
    yearly = yearly.join(meets_per_year)
    yearly_clean = yearly[yearly["Participants"] >= 500]
    print(yearly_clean.to_string())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("Powerlifting Growth Over Time (1998–2018)", fontsize=13, fontweight="bold")
    ax1.plot(yearly_clean.index, yearly_clean["Participants"],
             marker="o", color="#4C72B0", linewidth=2)
    ax1.set_ylabel("Participants")
    ax1.set_title("Participants per Year")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.plot(yearly_clean.index, yearly_clean["Avg_Wilks"],
             marker="s", color="#55A868", linewidth=2)
    ax2.set_ylabel("Avg Wilks Score")
    ax2.set_title("Average Wilks Score per Year")
    ax2.set_xlabel("Year")
    plt.tight_layout()
    plt.savefig("analysis_4_timeseries.png", dpi=120)
    plt.close()
    print("Saved → analysis_4_timeseries.png")

    # ── ANALYSIS 5 — Federation Performance ──────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS 5 — Federation performance (top 15 by participants)")
    print("=" * 60)

    fed = df[df["TotalKg"].notna()].groupby("Federation").agg(
        Participants = ("Name",    "count"),
        Avg_TotalKg  = ("TotalKg", "mean"),
        Avg_Wilks    = ("Wilks",   "mean"),
        DQ_Rate_pct  = ("is_dq",   "mean")
    ).round(4)
    fed["DQ_Rate_pct"] = (fed["DQ_Rate_pct"] * 100).round(2)
    top_feds = fed.sort_values("Participants", ascending=False).head(15)
    print(top_feds.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top_feds.index[::-1], top_feds["Participants"][::-1], color="#6A9FD4")
    ax.set_xlabel("Number of Participants")
    ax.set_title("Top 15 Federations by Participant Count", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, val in zip(bars, top_feds["Participants"][::-1]):
        ax.text(bar.get_width() + 400, bar.get_y() + bar.get_height() / 2,
                f"{int(val):,}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("analysis_5_federations.png", dpi=120)
    plt.close()
    print("Saved → analysis_5_federations.png")

    # ── ANALYSIS 6 — Country Breakdown ───────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS 6 — Country breakdown")
    print("=" * 60)

    country = (valid.groupby("MeetCountry").agg(
        Meets        = ("MeetID", "nunique"),
        Participants = ("Name",   "count"),
        Avg_Wilks    = ("Wilks",  "mean")
    ).round(2).sort_values("Participants", ascending=False).head(20))
    print(country.to_string())

    top4 = country.head(4)
    rest = country.iloc[4:]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Top 20 Countries by Participant Count", fontsize=13, fontweight="bold")
    ax1.barh(top4.index[::-1], top4["Participants"][::-1], color="#70B8A0")
    ax1.set_xlabel("Participants")
    ax1.set_title("Top 4 Countries")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, (idx, row) in enumerate(top4[::-1].iterrows()):
        ax1.text(row["Participants"] + 1000, i, f"{int(row['Participants']):,}",
                 va="center", fontsize=9)
    ax2.barh(rest.index[::-1], rest["Participants"][::-1], color="#70B8A0")
    ax2.set_xlabel("Participants")
    ax2.set_title("Countries Ranked 5–20")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, (idx, row) in enumerate(rest[::-1].iterrows()):
        ax2.text(row["Participants"] + 100, i, f"{int(row['Participants']):,}",
                 va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("analysis_6_countries.png", dpi=120)
    plt.close()
    print("Saved → analysis_6_countries.png")

    # ── ANALYSIS 7 — Age Group Performance ───────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS 7 — Age group performance")
    print("=" * 60)

    age_order = ["Sub-Junior (<18)", "Junior (18-23)", "Open (24-39)",
                 "Master 40-49", "Master 50-59", "Master 60+"]
    age_grp = (valid[valid["AgeGroup"] != "Unknown"]
               .groupby("AgeGroup").agg(
                   Count       = ("Name",    "count"),
                   Avg_TotalKg = ("TotalKg", "mean"),
                   Avg_Wilks   = ("Wilks",   "mean")
               ).round(2).reindex(age_order))
    print(age_grp.to_string())
    print(f"\nNote: 'Unknown' excluded — {(df['AgeGroup'] == 'Unknown').sum():,} rows affected.")

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(age_grp.index, age_grp["Avg_Wilks"], color="#9B7FD4")
    ax.set_ylabel("Average Wilks Score")
    ax.set_title("Average Wilks Score by Age Group", fontweight="bold")
    ax.set_xticklabels(age_grp.index, rotation=20, ha="right")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{bar.get_height():.1f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig("analysis_7_agegroups.png", dpi=120)
    plt.close()
    print("Saved → analysis_7_agegroups.png")

    # ── ANALYSIS 8 — Raw vs Equipped ─────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS 8 — Raw vs Equipped over time")
    print("=" * 60)

    def equip_cat(e):
        if e == "Raw":   return "Raw"
        if e == "Wraps": return "Wraps"
        return "Equipped (Single/Multi-ply)"

    valid["EquipCat"] = valid["Equipment"].apply(equip_cat)
    ts8 = valid[(valid["Year"] >= 1998) & (valid["Year"] <= 2017)].copy()
    yearly_equip     = ts8.groupby(["Year", "EquipCat"]).size().unstack(fill_value=0)
    yearly_equip_pct = yearly_equip.div(yearly_equip.sum(axis=1), axis=0) * 100
    print(yearly_equip_pct.round(1).to_string())

    colors8 = {"Raw": "#4C72B0", "Wraps": "#55A868", "Equipped (Single/Multi-ply)": "#DD8452"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Raw vs Equipped Powerlifting — Shift Over Time (1998–2017)",
                 fontsize=13, fontweight="bold")
    for col in yearly_equip.columns:
        ax1.plot(yearly_equip.index, yearly_equip[col], marker="o", linewidth=2,
                 label=col, color=colors8.get(col, "gray"))
    ax1.set_ylabel("Participants")
    ax1.set_title("Participant Count by Equipment Category")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for col in yearly_equip_pct.columns:
        ax2.plot(yearly_equip_pct.index, yearly_equip_pct[col], marker="s", linewidth=2,
                 label=col, color=colors8.get(col, "gray"))
    ax2.set_ylabel("Share of Participants (%)")
    ax2.set_title("Equipment Share (%) per Year")
    ax2.set_xlabel("Year")
    ax2.legend(loc="center right", fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig("analysis_8_raw_vs_equipped.png", dpi=120)
    plt.close()
    print("Saved → analysis_8_raw_vs_equipped.png")

    # ── ANALYSIS 9 — Correlation Heatmap ─────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS 9 — Correlation heatmap (numeric features)")
    print("=" * 60)

    # Select the numeric columns that matter for performance analysis.
    # We use valid entries only so DQ/NS results don't skew the picture.
    corr_cols = ["Age", "BodyweightKg", "BestSquatKg",
                 "BestBenchKg", "BestDeadliftKg", "TotalKg", "Wilks"]

    corr_df   = valid[corr_cols].dropna()
    corr_matrix = corr_df.corr().round(2)

    print("Pearson correlation matrix:")
    print(corr_matrix.to_string())

    # ── Key observations printed to console ──────────────────
    wilks_corr = corr_matrix["Wilks"].drop("Wilks").sort_values(ascending=False)
    print("\nCorrelations with Wilks score (highest → lowest):")
    for feat, val in wilks_corr.items():
        direction = "positive" if val > 0 else "negative"
        print(f"  {feat:<20} {val:+.2f}  ({direction})")

    bw_total = corr_matrix.loc["BodyweightKg", "TotalKg"]
    print(f"\nNote: BodyweightKg ↔ TotalKg correlation = {bw_total:.2f}")
    print("(High correlation expected — heavier athletes lift more in absolute terms,")
    print(" but Wilks normalises for bodyweight, hence the lower Wilks correlation.)")

    # ── Heatmap chart ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # hide upper triangle

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,          # show correlation values inside each cell
        fmt=".2f",
        cmap="coolwarm",     # red = strong positive, blue = strong negative
        vmin=-1, vmax=1,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax
    )

    ax.set_title("Correlation Heatmap — Numeric Performance Features\n"
                 "(valid entries only, lower triangle)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=10)

    plt.tight_layout()
    plt.savefig("analysis_9_correlation_heatmap.png", dpi=120)
    plt.close()
    print("Saved → analysis_9_correlation_heatmap.png")

    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print("=" * 60)
