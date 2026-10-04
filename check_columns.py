import pandas as pd
from src import config as C

src = r"data\NGA_West2_Flatfile_RotD50_d005_public_version.xlsx"
print("Reading Excel (about a minute)...")
df = pd.read_excel(src)
df.to_csv(r"data\NGAWest2_flatfile.csv", index=False)
with open(r"data\columns.txt", "w", encoding="utf8") as f:
    f.write("\n".join(map(str, df.columns)))
print("Saved CSV:", df.shape)

cols = list(df.columns)
wanted = {
  "RSN": [C.RSN], "EVENT": [C.EVENT], "STATION": [C.STATION], "LAT": [C.LAT], "LON": [C.LON],
  "MAG": [C.MAG], "H1_FILE": [C.H1_FILE], "H2_FILE": [C.H2_FILE], "H1_AZIMUTH": [C.H1_AZIMUTH],
  "EVENT_NUM": C.EVENT_NUM, "EVENT_CAT": C.EVENT_CAT, "SITE_NUM": C.SITE_NUM, "SITE_CAT": C.SITE_CAT,
}
print("\n=== config check ===")
for group, names in wanted.items():
    for n in names:
        print(("OK       " if n in cols else "MISSING  ") + f"[{group}] {n}")

print("\n=== possible matches for anything missing ===")
keys = ["azimuth","orient","file name","lat","lon","nehrp","mechanism","rupture","dip","rake",
        "hypocenter","depth","epi","hypd","joyner","clst","vs30","pga","pgv","pgd"]
for k in keys:
    hits = [c for c in cols if k in str(c).lower()]
    print(f"{k:12s}", hits[:12])
