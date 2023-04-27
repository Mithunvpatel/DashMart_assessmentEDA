#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px


# In[2]:


# Load data from CSV file
data = pd.read_excel(r'Sample deliveries data - 1 month (1) (2).xlsx')

# In[3]:


# Create scatter plot of order amount and tip amount
plt.scatter(data['Order total'], data['Amount of tip'])
plt.xlabel('Order amount')
plt.ylabel('Tip amount')

# Save the figure as a PNG image file
plt.savefig('scatterplot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Calculate correlation coefficient between order amount and tip amount
corr = data['Order total'].corr(data['Amount of tip'])
print('Correlation coefficient:', corr)


# In[74]:


# Calculate total order amount for each customer
total_order_amount = data.groupby('Consumer ID')['Order total'].sum().reset_index()

# Calculate total tip amount for each customer
total_tip_amount = data.groupby('Consumer ID')['Amount of tip'].sum().reset_index()

# Calculate number of orders for each customer
order_count = data.groupby('Consumer ID').size().reset_index(name='Order count')

# Combine data into a single dataframe
customer_data = pd.merge(total_order_amount, total_tip_amount, on='Consumer ID')
customer_data = pd.merge(customer_data, order_count, on='Consumer ID')

# Normalize data using Min-Max scaling
customer_data_norm = (customer_data - customer_data.min()) / (customer_data.max() - customer_data.min())

# Apply k-means clustering algorithm
kmeans = KMeans(n_clusters=4)
kmeans.fit(customer_data_norm)

# Add cluster labels to customer data
customer_data['Cluster'] = kmeans.labels_

# Analyze each customer segment
for i in range(kmeans.n_clusters):
    print('Cluster', i)
    cluster_data = customer_data[customer_data['Cluster'] == i]
    print('Number of customers:', len(cluster_data))
    print('Average order amount: $', cluster_data['Order total'].mean())
    print('Average tip amount: $', cluster_data['Amount of tip'].mean())
    print('Average order count:', cluster_data['Order count'].mean())
    print()

# Create scatter plot of order amount and tip amount for each customer
scatter = plt.scatter(customer_data['Order total'], customer_data['Amount of tip'], c=kmeans.labels_, cmap='rainbow')
plt.xlabel('Order amount')
plt.ylabel('Tip amount')

# # Create legend handles and labels
# handles, labels = scatter.legend_elements()

# Add legend to the scatter plot
plt.legend(handles, labels, loc='upper right')


# Alternative way to add legend
plt.legend(handles=scatter.legend_elements()[0], labels=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'], loc='upper right')

# # Save the figure as a PNG image file
# plt.savefig('Segment.png', dpi=300, bbox_inches='tight')

plt.show()


# In[5]:


# Calculate the tip percentage for each order
data['tip_percentage'] = data['Amount of tip'] / data['Order total']

# Calculate the average tip percentage for different order total ranges
order_total_bins = [0, 10, 20, 30, 40, 50, 100, 200]
data['order_total_range'] = pd.cut(data['Order total'], order_total_bins)
avg_tip_percentage = data.groupby('order_total_range')['tip_percentage'].mean()

# Create a bar chart to visualize the results
colors = ['tab:blue'] * len(avg_tip_percentage)
max_tip_idx = avg_tip_percentage.reset_index()['tip_percentage'].idxmax()
colors[max_tip_idx] = 'tab:red'

plt.figure(figsize=(10, 5))

# Convert the y-axis values to percentage
y_axis_values = avg_tip_percentage * 100

plt.bar(avg_tip_percentage.index.astype(str), y_axis_values, color=colors)
# plt.title('Average tip percentage by total order range')
plt.xlabel('Order total range')
plt.ylabel('Average tip percentage (%)')
plt.ylim(6, 7.5)

# Add labels to the bars
for i, v in enumerate(avg_tip_percentage):
    y_pos = v * 100 + 0.5
    if i == max_tip_idx:
        plt.text(i, y_pos - 0.54, f'{v:.2%}', ha='center', va='top', color='white', fontweight='bold')
    else:
        plt.text(i, y_pos, f'{v:.2%}', ha='center', va='top', color='white')

# Add a dotted line for the average tip percentage
plt.axhline(y=avg_tip_percentage.mean() * 100, linestyle='--', color='grey')

# add average line
avg_tip = avg_tip_percentage.mean()
plt.axhline(avg_tip * 100, color='grey', linestyle='--')

# add average label
plt.text(0.35, avg_tip * 100 + 0.08, f'Average: {avg_tip:.2%}', color='grey', ha='right')

# plt.savefig('AVGbytotOrder2.png', dpi=300, bbox_inches='tight')

plt.show()


# In[6]:


count = 0
for order_total in data["Order total"]:
    if 40 <= order_total <= 50:
        count += 1

print(f"Number of orders between 40 and 50 dollars: {count}")


# In[73]:


# Group the data by Restaurant ID and sum the refunded amount and tip amount for each group
refund_totals = data.groupby('Restaurant ID')[['Refunded amount', 'Amount of tip']].sum()

# Sort the groups in descending order based on the total refunded amount
sorted_refunds = refund_totals.sort_values(by='Refunded amount', ascending=False)

# Select the top 10 restaurants with the most refunds
top_10 = sorted_refunds.head(10)
top_314 = sorted_refunds.head(314)
# Create a scatter plot of the top 10 restaurants
fig, ax = plt.subplots(figsize=(8, 6))

# Set the x and y axis labels
ax.set_xlabel('Total Tips Received')
ax.set_ylabel('Total Refunds Received')

# Set the title of the plot
# ax.set_title('Top 10 Restaurants with the Most Refunds')

# Set the x-axis limits to 0 to 900
ax.set_ylim([0, 900])

# Create a scatter plot with different colors for each point
colors = range(len(top_10))
scatter = ax.scatter(top_10['Amount of tip'], top_10['Refunded amount'], s=top_10['Refunded amount'], c=colors, cmap='Purples')

# Add labels to the data points, showing the restaurant ID
for i, row in top_10.iterrows():
    ax.text(row['Amount of tip'] + 0.5, row['Refunded amount'] + 0.5, str(i))

# Add a legend showing the total refunds received by each restaurant
legend = ax.legend(labels=top_10.index, title='Restaurant ID', loc='upper left', fontsize='small', title_fontsize='small')

# Set the size of the markers in the legend
for handle in legend.legendHandles:
    handle.set_sizes([30])

# plt.savefig('T10Restrefund.png', dpi=300, bbox_inches='tight')    
    
# Show the plot
plt.show()
print(top_314.mean())


# In[58]:


data['Customer placed order date'] = pd.to_datetime(data['Customer placed order date'])
# assuming that the 'Customer placed order date' column is in datetime format
data['order_day'] = data['Customer placed order date'].dt.strftime('%A')
df_dates = data
df = df_dates

# assuming your dataframe is named "df"
# convert the "Customer placed order datetime" column to datetime if it's not already
df["Customer placed order datetime"] = pd.to_datetime(df["Customer placed order datetime"])

# define the ordered weekdays
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# convert the "order_day" column to a categorical type with the ordered weekdays
df["order_day"] = pd.Categorical(df["order_day"], categories=weekdays, ordered=True)

# rename the weekday categories to a shorter form
short_weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
df["order_day"].cat.rename_categories(short_weekdays, inplace=True)


# group by the ordered "order_day" column and count the number of rows in each group
demand_by_day = df.groupby("order_day").size()

# print the demand for each day of the week
print(demand_by_day)

# plot the demand by day as a bar chart
plt.bar(demand_by_day.index, demand_by_day.values)
plt.title("Demand by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Demand")
plt.ylim(2100, 3000)
plt.savefig('DOW.png', dpi=300, bbox_inches='tight') 
plt.show()


# In[ ]:




