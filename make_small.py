import pandas as pd

fake = pd.read_csv("Fake.csv").head(5000)
real = pd.read_csv("True.csv").head(5000)

fake.to_csv("Fake_small.csv", index=False)
real.to_csv("True_small.csv", index=False)

print("Done!")