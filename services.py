import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "climatechange.csv"

AFRICAN_COUNTRIES = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon", "Cape Verde",
    "Central African Republic", "Chad", "Comoros", "Congo", "Democratic Republic of Congo",
    "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia",
    "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotyo", "Liberia", "Libya",
    "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia",
    "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone",
    "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda",
    "Zambia", "Zimbabwe"
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    expected = {"country", "year", "co2", "co2_per_capita", "gdp_per_capita", "population", "energy_per_capita"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    return df


def get_country(df, country):
    return df[df["country"] == country].sort_values("year")


def get_trend_data(df, country, start_year=2002, end_year=2022):
    c = get_country(df, country)
    c = c[(c["year"] >= start_year) & (c["year"] <= end_year)]
    return {
        "years": c["year"].tolist(),
        "co2": c["co2"].fillna(0).tolist(),
        "co2_per_capita": c["co2_per_capita"].fillna(0).tolist(),
    }


def get_inequality_data(df, year=None):
    years = sorted(df["year"].dropna().unique())
    if not years:
        raise ValueError("No year data available")

    if year is None:
        year = int(df["year"].max())
    else:
        year = int(year)

    if year not in years:
        if year < years[0]:
            year = years[0]
        elif year > years[-1]:
            year = years[-1]
        else:
            year = max(y for y in years if y <= year)

    world = df.groupby("year").agg({"co2": "sum", "population": "sum"}).reset_index()
    world_year = world[world["year"] == year]
    kenya_year = df[(df["country"] == "Kenya") & (df["year"] == year)]
    india_year = df[(df["country"] == "India") & (df["year"] == year)]

    if world_year.empty or kenya_year.empty or india_year.empty:
        raise ValueError(f"Inequality data not found for year {year}")

    world_year = world_year.iloc[0]
    kenya_year = kenya_year.iloc[0]
    india_year = india_year.iloc[0]

    world_pc = (world_year["co2"] * 1_000_000) / world_year["population"]
    kenya_total = float(kenya_year["co2"])
    world_total = float(world_year["co2"])
    rest_world = max(world_total - kenya_total, 0)

    return {
        "year": int(year),
        "year_min": int(years[0]),
        "year_max": int(years[-1]),
        "kenya_pc": float(kenya_year["co2_per_capita"]),
        "india_pc": float(india_year["co2_per_capita"]),
        "world_pc": float(world_pc),
        "kenya_total": kenya_total,
        "rest_world": rest_world,
        "scatter": (
            df[
                (df["country"].isin([
                    "Kenya", "India", "United States", "Germany", "China",
                    "Brazil", "Nigeria", "United Kingdom", "South Africa", "Ethiopia"
                ])) & (df["year"] == year)
            ][["country", "gdp_per_capita", "co2_per_capita", "population"]]
            .dropna()
            .to_dict(orient="records")
        ),
    }


def get_top_emitters(df, year=None, top_n=10):
    if year is None:
        year = int(df["year"].max())
    year_data = df[df["year"] == year].dropna(subset=["co2"])
    top = year_data.nlargest(top_n, "co2")[["country", "co2", "co2_per_capita"]].to_dict(orient="records")
    return top


def get_emissions_intensity(df, year=None, countries=None):
    if year is None:
        year = int(df["year"].max())
    if countries is None:
        countries = [
            "Kenya", "India", "United States", "Germany", "China",
            "Brazil", "Nigeria", "United Kingdom", "South Africa", "Ethiopia"
        ]

    year_data = df[(df["year"] == year) & (df["country"].isin(countries))].copy()
    year_data = year_data.dropna(subset=["co2", "gdp_per_capita", "population"])
    year_data["gdp_per_capita"] = pd.to_numeric(year_data["gdp_per_capita"], errors="coerce")
    year_data = year_data.dropna(subset=["gdp_per_capita"])
    year_data["intensity"] = (
        (year_data["co2"].astype(float) * 1_000_000)
        / (year_data["gdp_per_capita"].astype(float) * year_data["population"].astype(float))
    )
    return (
        year_data[["country", "intensity", "co2", "gdp_per_capita"]]
        .sort_values("intensity", ascending=False)
        .to_dict(orient="records")
    )


def get_african_countries(df):
    all_countries = df["country"].unique()
    african = sorted([c for c in all_countries if c in AFRICAN_COUNTRIES])
    return african


def get_african_comparison_trends(df, countries, start_year=2002, end_year=2022):
    data = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    data = data[data["country"].isin(countries)]
    years = sorted(data["year"].unique())

    result = {
        "years": [int(y) for y in years],
        "countries": []
    }

    for country in countries:
        subset = data[data["country"] == country].sort_values("year")
        result["countries"].append({
            "name": country,
            "values": subset["co2_per_capita"].fillna(0).astype(float).tolist()
        })

    return result


def get_african_comparison_bar(df, year, countries):
    year_data = df[(df["year"] == year) & (df["country"].isin(countries))].copy()
    year_data = year_data.dropna(subset=["co2"])
    bar_data = year_data[["country", "co2", "co2_per_capita"]].sort_values("co2", ascending=False)

    result = []
    for _, row in bar_data.iterrows():
        result.append({
            "country": str(row["country"]),
            "co2": float(row["co2"]),
            "co2_per_capita": float(row["co2_per_capita"]) if pd.notna(row["co2_per_capita"]) else None
        })
    return result


def get_inequality_trends(df):
    """
    Returns aligned year-by-year gap data: Kenya co2_per_capita minus world average.
    All series share the same years axis so charts don't mismatch.
    """
    kenya = df[df["country"] == "Kenya"].sort_values("year").copy()

    world = df.groupby("year").agg({"co2": "sum", "population": "sum"}).reset_index()
    world["world_pc"] = (world["co2"] * 1_000_000) / world["population"]

    africa = df[df["country"].isin(AFRICAN_COUNTRIES)]
    africa_grouped = africa.groupby("year").agg({"co2": "sum", "population": "sum"}).reset_index()
    africa_grouped["africa_pc"] = (africa_grouped["co2"] * 1_000_000) / africa_grouped["population"]

    # Merge all on year so every series is aligned
    merged = kenya[["year", "co2_per_capita"]].merge(
        world[["year", "world_pc"]], on="year", how="inner"
    ).merge(
        africa_grouped[["year", "africa_pc"]], on="year", how="inner"
    )

    merged = merged.sort_values("year")

    # Compute the gap series (Kenya minus world average) for the gap chart
    merged["gap"] = merged["co2_per_capita"] - merged["world_pc"]

    return {
        "years": merged["year"].astype(int).tolist(),
        "kenya_trend": merged["co2_per_capita"].fillna(0).tolist(),
        "world_trend": merged["world_pc"].fillna(0).tolist(),
        "africa_trend": merged["africa_pc"].fillna(0).tolist(),
        # ← flat list of {year, value} dicts that the gap chart expects
        "gap": [
            {"year": int(row["year"]), "value": float(row["gap"])}
            for _, row in merged.iterrows()
        ]
    }


def get_africa_distribution(df, year):
    africa = df[
        (df["country"].isin(AFRICAN_COUNTRIES)) &
        (df["year"] == year)
    ]
    africa = africa.dropna(subset=["co2_per_capita"])
    return (
        africa.sort_values("co2_per_capita", ascending=False)
        [["country", "co2_per_capita"]]
        .to_dict(orient="records")
    )


def get_africa_efficiency(df, year):
    africa = df[
        (df["country"].isin(AFRICAN_COUNTRIES)) &
        (df["year"] == year)
    ].copy()

    africa["gdp_per_capita"] = pd.to_numeric(africa["gdp_per_capita"], errors="coerce")
    africa["population"] = pd.to_numeric(africa["population"], errors="coerce")
    africa["co2"] = pd.to_numeric(africa["co2"], errors="coerce")

    # Drop invalid rows
    africa = africa.dropna(subset=["co2", "gdp_per_capita", "population"])

    # Compute efficiency
    africa["efficiency"] = (
        (africa["gdp_per_capita"] * africa["population"])
        / (africa["co2"] * 1_000_000)
    )

    africa = africa.replace([float("inf"), -float("inf")], None)
    africa = africa.dropna(subset=["efficiency"])

    return (
        africa.sort_values("efficiency", ascending=False)
        [["country", "efficiency"]]
        .to_dict(orient="records")
    )


def get_decoupling_index(df, country):
    data = df[df["country"] == country].sort_values("year").copy()

    # Use gdp_per_capita * population as a proxy if gdp column is missing
    if "gdp" not in data.columns:
        data["gdp_per_capita"] = pd.to_numeric(data["gdp_per_capita"], errors="coerce")
        data["population"] = pd.to_numeric(data["population"], errors="coerce")
        data["gdp"] = data["gdp_per_capita"] * data["population"]

    data["gdp"] = pd.to_numeric(data["gdp"], errors="coerce")
    data["co2"] = pd.to_numeric(data["co2"], errors="coerce")

    data["gdp_growth"] = data["gdp"].pct_change() * 100
    data["co2_growth"] = data["co2"].pct_change() * 100
    data["decoupling_index"] = data["gdp_growth"] - data["co2_growth"]

    return data[["year", "decoupling_index"]].dropna().to_dict(orient="records")


def get_emission_elasticity(df, country):
    data = df[df["country"] == country].sort_values("year").copy()

    if "gdp" not in data.columns:
        data["gdp_per_capita"] = pd.to_numeric(data["gdp_per_capita"], errors="coerce")
        data["population"] = pd.to_numeric(data["population"], errors="coerce")
        data["gdp"] = data["gdp_per_capita"] * data["population"]

    data["gdp"] = pd.to_numeric(data["gdp"], errors="coerce")
    data["co2"] = pd.to_numeric(data["co2"], errors="coerce")

    data["gdp_growth"] = data["gdp"].pct_change()
    data["co2_growth"] = data["co2"].pct_change()
    data["elasticity"] = data["co2_growth"] / data["gdp_growth"]

    return (
        data[["year", "elasticity"]]
        .replace([float("inf"), -float("inf")], None)
        .dropna()
        .to_dict(orient="records")
    )


def get_regression_residual(df, target_country, year):
    data = df[df["year"] == year].copy()

    data["gdp_per_capita"] = pd.to_numeric(data["gdp_per_capita"], errors="coerce")
    data["co2_per_capita"] = pd.to_numeric(data["co2_per_capita"], errors="coerce")
    data = data.dropna(subset=["gdp_per_capita", "co2_per_capita"])
    data = data[(data["gdp_per_capita"] > 0) & (data["co2_per_capita"] > 0)]

    data["log_gdp"] = np.log(data["gdp_per_capita"])
    data["log_co2"] = np.log(data["co2_per_capita"])

    others = data[data["country"] != target_country]
    target = data[data["country"] == target_country]

    if target.empty or len(others) < 5:
        return None

    slope, intercept = np.polyfit(others["log_gdp"], others["log_co2"], 1)

    actual = target["log_co2"].values[0]
    predicted = slope * target["log_gdp"].values[0] + intercept
    residual = actual - predicted

    y_pred = slope * others["log_gdp"] + intercept
    ss_res = np.sum((others["log_co2"] - y_pred) ** 2)
    ss_tot = np.sum((others["log_co2"] - others["log_co2"].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        "year": year,
        "residual": float(residual),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "n": len(others)
    }


def get_cagr_decoupling(df, country, start_year=2002, end_year=2022):
    data = df[
        (df["country"] == country) &
        (df["year"].isin([start_year, end_year]))
    ].copy().sort_values("year")

    if len(data) < 2:
        return None

    # Use gdp proxy if needed
    if "gdp" not in data.columns:
        data["gdp"] = (
            pd.to_numeric(data["gdp_per_capita"], errors="coerce") *
            pd.to_numeric(data["population"], errors="coerce")
        )

    gdp_start, gdp_end = float(data["gdp"].iloc[0]), float(data["gdp"].iloc[1])
    co2_start, co2_end = float(data["co2"].iloc[0]), float(data["co2"].iloc[1])

    if gdp_start == 0 or co2_start == 0:
        return None

    n = end_year - start_year
    gdp_cagr = (gdp_end / gdp_start) ** (1 / n) - 1
    co2_cagr = (co2_end / co2_start) ** (1 / n) - 1

    return {
        "gdp_cagr": float(gdp_cagr * 100),
        "co2_cagr": float(co2_cagr * 100),
        "decoupling_score": float((gdp_cagr - co2_cagr) * 100)
    }


def get_overview_data(df, country="Kenya"):
    data = df[df["country"] == country].sort_values("year").copy()
    data = data.dropna(subset=["co2_per_capita", "gdp_per_capita", "co2"])

    # Use gdp proxy if needed
    if "gdp" not in data.columns:
        data["gdp"] = (
            pd.to_numeric(data["gdp_per_capita"], errors="coerce") *
            pd.to_numeric(data["population"], errors="coerce")
        )

    data = data.dropna(subset=["gdp"])

    years = data["year"].astype(int).tolist()
    co2_pc = data["co2_per_capita"].astype(float).tolist()
    gdp_pc = data["gdp_per_capita"].astype(float).tolist()
    total_gdp = data["gdp"].astype(float).tolist()

    latest = data.iloc[-1]
    first = data.iloc[0]

    co2_growth = ((latest["co2"] - first["co2"]) / first["co2"]) * 100 if first["co2"] != 0 else 0
    gdp_growth = ((latest["gdp"] - first["gdp"]) / first["gdp"]) * 100 if first["gdp"] != 0 else 0

    return {
        "years": years,
        "co2_pc": co2_pc,
        "gdp_pc": gdp_pc,
        "total_gdp": total_gdp,
        "latest": {
            "co2_per_capita": float(latest["co2_per_capita"]),
            "gdp_per_capita": float(latest["gdp_per_capita"])
        },
        "growth": {
            "co2": float(co2_growth),
            "gdp": float(gdp_growth)
        }
    }

def compute_decomposition(countries, df, year_start=2010, year_end=2022):
    result = []
    for country in countries:
        d = df[df['country'] == country]
        try:
            r0 = d[d['year'] == year_start].iloc[0]
            r1 = d[d['year'] == year_end].iloc[0]

            pop0   = float(r0['population'])
            pop1   = float(r1['population'])
            gdp0   = float(r0['gdp_per_capita'])
            gdp1   = float(r1['gdp_per_capita'])
            co2_0  = float(r0['co2'])
            co2_1  = float(r1['co2'])

            pop_effect       = ((pop1 / pop0) - 1) * 100
            income_effect    = ((gdp1 / gdp0) - 1) * 100
            total_change     = ((co2_1 / co2_0) - 1) * 100
            intensity_effect = total_change - pop_effect - income_effect

            result.append({
                "country": country,
                "population_effect": round(pop_effect, 2),
                "income_effect": round(income_effect, 2),
                "intensity_effect": round(intensity_effect, 2)
            })
        except (IndexError, KeyError, ZeroDivisionError, ValueError):
            continue
    return result