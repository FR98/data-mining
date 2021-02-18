"""
---------------------------------------------------------------------------------------------------
	Author
	    Francisco Rosal 18676
---------------------------------------------------------------------------------------------------
"""

import pandas
import matplotlib.pyplot as plt

data = pandas.read_csv('fortune500.csv')
data.columns = ["year", "rank_", "company", "revenue", "profit"]

print(data.head())
print("\nRecords: ", len(data))
print("\n")
print(data.info())
print("\n")

data.profit = pandas.to_numeric(data.profit, errors="coerce")
print(data.info())
print("\n")

profit_nan = data.profit.isnull()
print(data.loc[profit_nan].head())

print("\nNaN data count on profit: ", len(data.profit[profit_nan]))
print(369/len(data))

plt.hist(data.year[profit_nan], bins=range(1955, 2006))
plt.show()

data = data.loc[~profit_nan]

avg = data.loc[:, ["year", "revenue", "profit"]].groupby("year").mean()
plt.plot(avg.index, avg.profit)
plt.show()

plt.plot(avg.index, avg.revenue)
plt.show()

# Shaded plot
plt.plot(avg.index, avg.profit)
std_profit_list = list(data.loc[:, ["year", "revenue", "profit"]].groupby("year").std()["profit"])
avg_profit_list = list(avg.profit)

y1 = [avg_profit_list[i] - std_profit_list[i] for i in range(len(avg_profit_list))]
y2 = [avg_profit_list[i] + std_profit_list[i] for i in range(len(avg_profit_list))]
plt.fill_between(avg.index, y1, y2, color='blue', alpha=0.2)
plt.show()

# Shaded plot
plt.plot(avg.index, avg.revenue)
std_revenue_list = list(data.loc[:, ["year", "revenue", "profit"]].groupby("year").std()["revenue"])
avg_revenue_list = list(avg.revenue)

y1 = [avg_revenue_list[i] - std_revenue_list[i] for i in range(len(avg_revenue_list))]
y2 = [avg_revenue_list[i] + std_revenue_list[i] for i in range(len(avg_revenue_list))]
plt.fill_between(avg.index, y1, y2, color='blue', alpha=0.2)
plt.show()
