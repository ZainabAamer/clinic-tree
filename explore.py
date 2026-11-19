import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:f}")

df = pd.read_csv("clinic_appointments.csv")

print(df.shape)
print(df.describe())

n_no_show = (df["no_show"] == "no-show").sum()
share = n_no_show / len(df)
print(n_no_show)
print(share)

no_show_rate = (
    df.assign(is_no_show=df["no_show"] == "no-show")
    .groupby("prior_no_shows")["is_no_show"]
    .mean()
)
print(no_show_rate)
