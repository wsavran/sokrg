import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import randn
from krg_utils import transform_normal_scores

r = randn(10000)
slip_sc = pd.read_csv('slip_nscore_transform_table.csv')

slip = transform_normal_scores(r, slip_sc)

avg_slip = slip_sc['x'].sum() / len(slip_sc['x'])
avg_score = slip_sc['nscore'].sum() / len(slip_sc['nscore'])
print(avg_slip, slip.mean())

# visualize
fig, ax = plt.subplots()
ax.plot(slip_sc['x'], slip_sc['nscore'])
ax.axvline(x=avg_slip, linestyle='--', color='black')
plt.show()