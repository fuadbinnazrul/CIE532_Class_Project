#!/usr/bin/env python
# coding: utf-8

# In[82]:


get_ipython().run_line_magic('matplotlib', 'inline')
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys,os,glob
sys.path.append('/projects/academic/kangsun/fuadbinn/Oversampling_matlab')
from popy import Level3_Data, Level3_List
from scipy import stats
from itertools import combinations


# In[83]:


dts = pd.date_range('2020-1-1','2020-12-31',freq='1d')
l3_path_pattern = '/vscratch/grp-kangsun/zolalaya/s5p_cornbelt/grid_size_0.02_block_reduce/%Y/%m/CONUS_%Y_%m_%d.nc'
l3s = Level3_List(dt_array=dts)


# In[84]:


l3s.read_nc_pattern(
    l3_path_pattern=l3_path_pattern,
    fields_name=['column_amount','column_amount_DD','surface_altitude_DD']
)


# In[85]:


dates = pd.to_datetime(dts)
if len(dates) != len(l3s):
    dates = dates[:len(l3s)]  


daily_mean = np.array([np.nanmean(day['column_amount']) for day in l3s])


s_daily = pd.Series(daily_mean, index=dates)


s_weekly = s_daily.resample('W-SUN').mean()
s_roll7  = s_daily.rolling(window=7, min_periods=1, center=False).mean()


# In[86]:


fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

axes[0].plot(s_daily.index, s_daily.values, marker='o', lw=1, label='Daily')
axes[0].set_title('Daily time series of NO$_2$ column amount (Year 2020)')
axes[0].set_ylabel('Column Amount')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(s_weekly.index, s_weekly.values, color='green', marker='s', lw=1.5, label='Weekly mean')
axes[1].set_title('Weekly mean of NO$_2$ column amount (Year 2020)')
axes[1].set_ylabel('Column Amount')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()


# In[87]:


#Weekdays average

fields    = ['column_amount', 'column_amount_DD', 'surface_altitude_DD']
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

rows = []

for dow, name in enumerate(day_names):
    mask = (l3s.df.index.dayofweek == dow)

    l3s_day = l3s.trim(time_mask=mask)
    l3s_day = l3s_day.aggregate()

    row = {"weekday": name}
    for f in fields:
        row[f] = np.nanmean(l3s_day[f])
    rows.append(row)

weekday_means = pd.DataFrame(rows).set_index("weekday")


weekend_mask = l3s.df.index.dayofweek.isin([5, 6])

l3s_weekend = l3s.trim(time_mask=weekend_mask).aggregate()
weekend_avg = {f: np.nanmean(l3s_weekend[f]) for f in fields}


weekday_mask = l3s.df.index.dayofweek.isin([0, 1, 2, 3, 4])

l3s_weekdays = l3s.trim(time_mask=weekday_mask).aggregate()
weekday_avg_all = {f: np.nanmean(l3s_weekdays[f]) for f in fields}


print()
display(weekday_means)


# In[88]:


plt.figure(figsize=(9,5))

plt.plot(
    weekday_means.index,
    weekday_means['column_amount'],
    marker='o',
    linewidth=2,
    color='royalblue',
    label='Column Amount'
)

plt.title("Column Amount – Mean by Weekday", fontsize=14)
plt.xlabel("Weekday")
plt.ylabel("Mean Column Amount")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[89]:


weekday_val = weekday_avg_all['column_amount']
weekend_val = weekend_avg['column_amount']

percent_decrease = ((weekday_val - weekend_val) / weekday_val) * 100

print(f"Weekend NO₂ is {percent_decrease:.2f}% lower than weekdays.")


# In[98]:


#6 days rolling average

beta_total_vals = []
for dow in range(7):
    mask = (l3s.df.index.dayofweek == dow)
    l3s_day = l3s.trim(time_mask=mask).aggregate()
    beta_total_vals.append(np.nanmean(l3s_day[field]))
beta_total = pd.Series(beta_total_vals, index=day_names, name="beta_total_direct")



y6_vals = []
for dow in range(7):
    mask = (l3s.df.index.dayofweek != dow) 
    l3s_6day = l3s.trim(time_mask=mask).aggregate()
    y6_vals.append(np.nanmean(l3s_6day[field]))
y6 = pd.Series(y6_vals, index=day_names, name="y_6day_leave1out")

S6 = np.nansum(y6.values)
beta_6day = pd.Series(S6 - 6.0 * y6.values, index=day_names, name="beta_from_6day")


# In[91]:


#5 days rolling average

y5_vals = []
A_rows = []
y5_labels = []

k = 1
for ex1, ex2 in combinations(range(7), 2): 
    included = [d for d in range(7) if d not in (ex1, ex2)]

    mask = l3s.df.index.dayofweek.isin(included)
    l3s_5day = l3s.trim(time_mask=mask).aggregate()
    yk = np.nanmean(l3s_5day[field])
    y5_vals.append(yk)

    row = np.zeros(7, dtype=float)
    row[included] = 1.0
    A_rows.append(row)

    y5_labels.append(f"y{k}: exclude {day_names[ex1]} & {day_names[ex2]}")
    k += 1

y5 = pd.Series(y5_vals, index=y5_labels, name="y_5day")
A = np.vstack(A_rows)                 # shape (21, 7)
b = 5.0 * y5.to_numpy(dtype=float)    # A beta = 5y

beta_5day_vec, *_ = np.linalg.lstsq(A, b, rcond=None)   # robust LS solution
beta_5day = pd.Series(beta_5day_vec, index=day_names, name="beta_from_5day")


# In[92]:


def per_day_metrics(est: pd.Series, truth: pd.Series):
    err = est - truth
    abs_err = err.abs()
    sq_err = err**2
    mae = abs_err.mean()
    rmse = np.sqrt(sq_err.mean())
    return abs_err, sq_err, mae, rmse


abs6, sq6, mae6, rmse6 = per_day_metrics(beta_6day, beta_total)
abs5, sq5, mae5, rmse5 = per_day_metrics(beta_5day, beta_total)


table = pd.DataFrame({
    "Day": day_names,
    "beta_observed": beta_total.values,
    "beta_6day": beta_6day.values,
    "beta_5day": beta_5day.values,
    "abs_err_6day": abs6.values,
    "abs_err_5day": abs5.values,
    "sq_err_6day": sq6.values,
    "sq_err_5day": sq5.values,
}).set_index("Day")


table["MAE_6day_overall"] = mae6
table["RMSE_6day_overall"] = rmse6
table["MAE_5day_overall"] = mae5
table["RMSE_5day_overall"] = rmse5

print(table)

summary = pd.DataFrame({
    "MAE": [mae6, mae5],
    "RMSE": [rmse6, rmse5]
}, index=["6-day reconstructed vs observed", "5-day reconstructed vs observed"])
print("\nSummary:")
print(summary)


# In[93]:


#Monthly weekdays and Weekends analysis

month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

weekday_means_month = []
weekend_means_month = []

for m in range(1, 13):

    mask_month = (l3s.df.index.month == m)

    mask_weekday = mask_month & l3s.df.index.dayofweek.isin([0,1,2,3,4])
    l3s_weekday = l3s.trim(time_mask=mask_weekday)

    if len(l3s_weekday.df) == 0:
        weekday_means_month.append(np.nan)
    else:
        weekday_means_month.append(np.nanmean(l3s_weekday.aggregate()[field]))

    mask_weekend = mask_month & l3s.df.index.dayofweek.isin([5,6])
    l3s_weekend = l3s.trim(time_mask=mask_weekend)

    if len(l3s_weekend.df) == 0:
        weekend_means_month.append(np.nan)
    else:
        weekend_means_month.append(np.nanmean(l3s_weekend.aggregate()[field]))


x = np.arange(12)
width = 0.35

plt.figure(figsize=(14,6))

plt.bar(x - width/2, weekday_means_month, width, label="Weekday Mean", color="royalblue")
plt.bar(x + width/2, weekend_means_month, width, label="Weekend Mean", color="orange")

plt.xticks(x, month_names)
plt.ylabel("Mean Column Amount")
plt.title("Monthly Weekday vs Weekend Mean (Column Amount)")
plt.grid(axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# In[94]:


#short-term trend analysis

weeks = l3s.df.index.isocalendar().week.to_numpy()
unique_weeks = np.unique(weeks)

weekly_vals = []
week_ids = []

for w in unique_weeks:
    mask_week = (weeks == w)
    l3_week = l3s.trim(time_mask=mask_week)

    if len(l3_week.dt_array) == 0:
        continue

    l3_week = l3_week.aggregate()
    mean_val = np.nanmean(l3_week[field])

    if np.isfinite(mean_val):
        weekly_vals.append(mean_val)
        week_ids.append(w)

weekly_vals = np.array(weekly_vals)
week_ids = np.array(week_ids)


n_last = 26
weekly_vals_26 = weekly_vals[-n_last:]
week_ids_26 = week_ids[-n_last:]


X = np.arange(1, len(weekly_vals_26) + 1)
y = weekly_vals_26


X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()

intercept = model.params[0]
slope = model.params[1]
p_slope = model.pvalues[1]
r2 = model.rsquared

ci = model.conf_int(alpha=0.05)   
ci_slope_low  = ci[1, 0]
ci_slope_high = ci[1, 1]

print(f"y = {intercept:.6e} + {slope:.6e} * week_index")
print(f"Slope p-value (H0: slope = 0) = {p_slope:.4e}")
print(f"R² = {r2:.4f}")
print(f"95% CI for slope = [{ci_slope_low:.6e}, {ci_slope_high:.6e}]")


pred = model.get_prediction(X_with_const)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int(alpha=0.05) 


plt.figure(figsize=(10,5))

plt.plot(X, y, "o", label="Weekly mean (last 26 weeks)")

plt.plot(X, pred_mean, "r-", linewidth=2, label="Trend line")

plt.fill_between(X, pred_ci[:,0], pred_ci[:,1], 
                 color="red", alpha=0.2, label="95% CI")

plt.xlabel("Week")
plt.ylabel("Weekly mean Column Amount")
plt.title("Trend of NO$_2$ Column Amount Over Last 26 Weeks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# In[95]:


#Model comparison

class WeekAverager():
    def __init__(self, days_per_week=7):
        self.days_per_week = days_per_week

    def get_combinations(self, days_to_skip):
        return list(combinations(range(self.days_per_week), days_to_skip))

week_averager = WeekAverager(7)


weeks = l3s.df.index.isocalendar().week.to_numpy()
unique_weeks = np.unique(weeks)
last_26_weeks = unique_weeks[-26:]

model1_weekly = []      # 7-day mean (reference)
model2_weekly = []      # best 6-day avg (min error)
model3_weekly = []      # best 5-day avg (min error)

for w in last_26_weeks:
    week_mask = (weeks == w)
    l3_week = l3s.trim(time_mask=week_mask)

    if len(l3_week.dt_array) < 5:
        continue
  
    daily_vals = []
    for d in range(len(l3_week.dt_array)):
        daily_vals.append(np.nanmean(l3_week[d][field]))

    daily_vals = np.array(daily_vals)

    if len(daily_vals) != 7:
        continue


    true_week_mean = np.nanmean(daily_vals)
    model1_weekly.append(true_week_mean)

    skip1_sets = week_averager.get_combinations(days_to_skip=1)

    model2_candidates = []
    for skip in skip1_sets:
        keep = [i for i in range(7) if i not in skip]
        avg_val = np.nanmean(daily_vals[keep])
        model2_candidates.append(avg_val)

    model2_best = min(model2_candidates, key=lambda x: abs(x - true_week_mean))
    model2_weekly.append(model2_best)

    skip2_sets = week_averager.get_combinations(days_to_skip=2)

    model3_candidates = []
    for skip in skip2_sets:
        keep = [i for i in range(7) if i not in skip]
        avg_val = np.nanmean(daily_vals[keep])
        model3_candidates.append(avg_val)

    model3_best = min(model3_candidates, key=lambda x: abs(x - true_week_mean))
    model3_weekly.append(model3_best)


model1 = np.array(model1_weekly)
model2 = np.array(model2_weekly)
model3 = np.array(model3_weekly)


mae2 = np.mean(np.abs(model2 - model1))
mae3 = np.mean(np.abs(model3 - model1))

rmse2 = np.sqrt(np.mean((model2 - model1)**2))
rmse3 = np.sqrt(np.mean((model3 - model1)**2))


print("\nModel 2: Best 6-day average")
print(f"MAE  = {mae2:.6e}")
print(f"RMSE = {rmse2:.6e}")

print("\nModel 3: Best 5-day average")
print(f"MAE  = {mae3:.6e}")
print(f"RMSE = {rmse3:.6e}")
  


# In[96]:


error_df = pd.DataFrame({
    "week_id": used_weeks,
    "week_index_last26": np.arange(1, len(used_weeks) + 1),
    "mean_7day": model1,
    "best_6day": model2,
    "best_5day": model3,
    "abs_err_6day": err2_abs,
    "abs_err_5day": err3_abs,
    "sq_err_6day": err2_sq,
    "sq_err_5day": err3_sq,
})

plt.figure(figsize=(10,5))

x = error_df["week_index_last26"].values

plt.plot(x, error_df["mean_7day"], marker="o", label="7-day mean (reference)")
plt.plot(x, error_df["best_6day"], marker="s", label="Best 6-day avg")
plt.plot(x, error_df["best_5day"], marker="^", label="Best 5-day avg")

plt.xlabel("Week")
plt.ylabel("Weekly mean Column Amount")
plt.title("Model comparison over last 26 weeks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# In[97]:


#Study Area Plotting
mask_mon = (l3s.df.index.dayofweek == 0) 

l3s_mon = l3s.trim(time_mask=mask_mon).aggregate()

l3s_mon.plot('column_amount')            

