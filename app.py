from flask import Flask, render_template, jsonify, request, abort
import os
import math
import numpy as np

from services import (
    load_data,
    get_country,
    get_trend_data,
    get_inequality_data,
    get_top_emitters,
    get_emissions_intensity,
    get_african_countries,
    get_african_comparison_trends,
    get_african_comparison_bar,
    get_inequality_trends,
    get_africa_distribution,
    get_africa_efficiency,
    get_decoupling_index,
    get_emission_elasticity,
    get_regression_residual,
    get_cagr_decoupling,
    get_overview_data,
    compute_decomposition
)

app = Flask(__name__)
 
def safe(val, decimals=2):
    """Convert a potentially NaN value to a rounded float or None."""
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, decimals)
    except (TypeError, ValueError):
        return None


def validate_year(year_str, df, param_name="year"):
    """Parse and clamp a year query param to the dataset's range."""
    try:
        y = int(year_str)
    except (TypeError, ValueError):
        abort(400, description=f"Invalid value for '{param_name}': must be an integer.")
    min_y = int(df["year"].min())
    max_y = int(df["year"].max())
    if not (min_y <= y <= max_y):
        abort(400, description=f"'{param_name}' must be between {min_y} and {max_y}.")
    return y

@app.route('/trends')
def trends():
    df_live = load_data()
    kenya = get_country(df_live, "Kenya").sort_values("year")

    return render_template(
        "trends.html",
        years=kenya['year'].astype(int).tolist(),
        co2=kenya['co2'].astype(float).tolist(),
        co2_per_capita=kenya['co2_per_capita'].astype(float).tolist(),
        gdp_per_capita=kenya['gdp_per_capita'].astype(float).tolist()
    )

@app.route("/inequality")
def inequality():
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    summary = get_inequality_data(df, max_year)
    top_emitters = get_top_emitters(df, max_year, 10)
    emissions_intensity = get_emissions_intensity(df, max_year)

    kenya_trend = get_trend_data(df, "Kenya")
    return render_template("inequality.html",
        year=summary["year"],
        year_min=min_year,
        year_max=max_year,
        kenya_pc=safe(summary["kenya_pc"]),
        india_pc=safe(summary["india_pc"]),
        world_pc=safe(summary["world_pc"]),
        kenya_total=safe(summary["kenya_total"]),
        rest_world=safe(summary["rest_world"]),
        scatter=summary["scatter"],
        top_emitters=top_emitters,
        emissions_intensity=emissions_intensity,
        kenya_trend_years=kenya_trend["years"],
        kenya_trend_co2=kenya_trend["co2"],
    )

@app.route("/african-comparison")
def african_comparison():
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    african_countries = get_african_countries(df)
    default_countries = (
        ["Kenya", "Nigeria", "South Africa", "Egypt", "Ethiopia"]
        if "Kenya" in african_countries
        else african_countries[:5]
    )

    trends = get_african_comparison_trends(df, default_countries, min_year, max_year)
    bar_data = get_african_comparison_bar(df, max_year, default_countries)

    return render_template("african-comparison.html",
        year_min=min_year,
        year_max=max_year,
        all_countries=african_countries,
        selected_countries=default_countries,
        trend_data=trends,
        bar_data=bar_data,
        current_year=max_year,
    )


@app.route("/api/trends")
def api_trends():
    country = request.args.get("country", "Kenya")
    start = int(request.args.get("start", 2002))
    end = int(request.args.get("end", 2022))
    if country not in df["country"].unique():
        abort(404, description="Country not found")
    result = get_trend_data(df, country, start, end)
    return jsonify(result)


@app.route("/api/inequality")
def inequality_api():
    year = validate_year(request.args.get("year", int(df["year"].max())), df)

    full_inequality = get_inequality_data(df, year)
    residual_data = get_regression_residual(df, "Kenya", year)  # ← was missing country arg
    gap_data = get_inequality_trends(df)
    top_emitters = get_top_emitters(df, year)

    return jsonify({
        "scatter": full_inequality["scatter"],   # ← extract just the array
        "residual": residual_data,
        "gap": gap_data,
        "top": top_emitters
    })

@app.route("/api/african-comparison")
def api_african_comparison():
    countries = request.args.getlist("countries")
    start = int(request.args.get("start", 2002))
    end = int(request.args.get("end", 2022))
    year = validate_year(request.args.get("year", 2022), df)

    if not countries:
        countries = ["Kenya", "Nigeria", "South Africa", "Egypt", "Ethiopia"]

    try:
        trends = get_african_comparison_trends(df, countries, start, end)
        bar_data = get_african_comparison_bar(df, year, countries)
        return jsonify({
            "trends": trends,
            "distribution": bar_data,
            "efficiency": get_africa_efficiency(df, year),
            "decomposition": compute_decomposition(countries, df),
            "scatter": (
                df[
                    (df["country"].isin(countries)) & (df["year"] == year)
                ][["country", "gdp_per_capita", "co2_per_capita", "population"]]
                .dropna()
                .to_dict(orient="records")
            ),
            "year": year
        })
    except Exception as e:
        print(f"ERROR in api_african_comparison: {e}")  # ← add this
        abort(400, description=str(e))

@app.route("/refresh")
def refresh():
    global df, kenya
    df = load_data()
    kenya = get_country(df, "Kenya")
    india = get_country(df, "India")
    return "Data refreshed", 200


@app.route("/api/advanced-metrics")
def api_advanced_metrics():
    country = request.args.get("country", "Kenya")
    year = validate_year(request.args.get("year", int(df["year"].max())), df)

    if country not in df["country"].unique():
        abort(404, description="Country not found")

    return jsonify({
        "decoupling": get_decoupling_index(df, country),
        "elasticity": get_emission_elasticity(df, country),
        "residual": get_regression_residual(df, country, year),  # Fixed: consistent 3-arg signature
        "cagr": get_cagr_decoupling(df, country)
    })


@app.route("/documentation")
def documentation():
    return render_template("documentation.html")


@app.route("/methodology")
def methodology():
    return render_template("methodology.html")

@app.route("/")
def splash():
    return render_template("splash.html")

@app.route("/overview")
def index():
    overview = get_overview_data(df)
    latest = kenya.iloc[-1]
    return render_template("index.html",
        year=int(latest["year"]),
        co2_per_capita=overview["latest"]["co2_per_capita"],
        gdp_per_capita=overview["latest"]["gdp_per_capita"],
        co2_growth=overview["growth"]["co2"],
        gdp_growth=overview["growth"]["gdp"]
    )

@app.route("/api/overview")
def api_overview():
    return jsonify(get_overview_data(df))


if __name__ == "__main__":
    app.run(debug=True)

