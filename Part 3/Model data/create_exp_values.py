import pandas as pd
from pathlib import Path


script_dir = Path(__file__).resolve().parent

df = pd.read_csv(script_dir / "wind_onshore_3_days.csv")

expected_wind = (
    df.groupby(df["time"].str[-8:])["wind_availability"]
    .mean()
    .reset_index()
)

expected_wind.columns = ["hour", "wind_availability"]
expected_wind["time"] = pd.date_range(
    start="2016-01-01 00:00:00",
    periods=24,
    freq="h"
)

expected_wind = expected_wind[["time", "wind_availability"]]

expected_wind.to_csv(script_dir / "wind_onshore_exp_values.csv", index=False)
