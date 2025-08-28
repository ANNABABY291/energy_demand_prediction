import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

def prepare(input_zip: str, output_csv: str = "hourly_energy_weather.csv"): 
    zip_path = Path(input_zip)
    if not zip_path.exists():
        raise FileNotFoundError(f"Input zip not found: {zip_path}")
    extract_dir = zip_path.parent / "uci_extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

    # Find the txt file
    txt_files = list(extract_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError("No .txt file found after extraction.")
    raw_path = txt_files[0]

    # Efficient read of necessary columns and time parsing
    df = pd.read_csv(
        raw_path,
        sep=";",
        usecols=["Date","Time","Global_active_power"],
        na_values="?",
        low_memory=True,
        parse_dates={"timestamp": ["Date","Time"]},
        dayfirst=True,
        infer_datetime_format=True
    ).dropna(subset=["timestamp","Global_active_power"])

    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
    df = df.dropna(subset=["Global_active_power"])

    # Resample to hourly means
    df["timestamp_hour"] = df["timestamp"].dt.floor("H")
    hourly = df.groupby("timestamp_hour", as_index=False)["Global_active_power"].mean()
    hourly = hourly.rename(columns={"timestamp_hour":"timestamp"})

    # Add synthetic weather + holiday flag
    idx = hourly["timestamp"]
    rng = np.random.default_rng(42)
    temperature = 12 + 8*np.sin(2*np.pi*(idx.dt.dayofyear/365.25)) + rng.normal(0, 1.2, len(idx))
    humidity = 60 + 15*np.sin(2*np.pi*(idx.dt.hour/24 + 0.2)) + rng.normal(0, 5, len(idx))
    humidity = np.clip(humidity, 20, 100)
    wind_speed = np.abs(3 + 1.5*np.sin(2*np.pi*(idx.dt.dayofyear/14))) + rng.normal(0, 0.5, len(idx))
    wind_speed = np.clip(wind_speed, 0, None)
    precipitation = rng.gamma(shape=0.6, scale=0.4, size=len(idx))
    precipitation[precipitation < 0.2] = 0.0
    is_holiday = (idx.dt.dayofweek >= 5).astype(int)

    out = pd.DataFrame({
        "timestamp": hourly["timestamp"],
        "load_kW": hourly["Global_active_power"],
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "precipitation": precipitation,
        "is_holiday": is_holiday
    })

    out.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

# Remove the argparse part and call the function directly
# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser(description="Prepare hourly_energy_weather.csv from UCI household dataset zip.")
#     p.add_argument("--input_zip", required=True, help="Path to household_power_consumption.txt.zip")
#     p.add_argument("--output_csv", default="hourly_energy_weather.csv", help="Output CSV path")
#     args = p.parse_args()
#     prepare(args.input_zip, args.output_csv)

# Call the prepare function directly with the input zip file path
prepare(input_zip="/content/household_power_consumption.txt.zip")
