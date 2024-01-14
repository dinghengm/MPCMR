#%%
import matplotlib.pyplot as plt
import numpy as np

# Data from the image
groups = ['AROM Pre MRI', 'AROM Post MRI', 'Combination Pre MRI', 'Combination Post MRI']
means = [6.7, 0.2, 12.9, 3.4]
medians = [3.5, 1.3, 6, 2]

# Converting means and medians to numpy arrays for easy plotting
means = np.array(means)
medians = np.array(medians)

# Indices of the groups
ind = np.arange(len(groups))

# Width of the bars
width = 0.35

fig, ax = plt.subplots()

# Plotting means and medians
rects1 = ax.bar(ind - width/2, means, width, label='Mean')
rects2 = ax.bar(ind + width/2, medians, width, label='Median')

# Adding some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time in minutes')
ax.set_title('Duration of nursing time in minutes')
ax.set_xticks(ind)
ax.set_xticklabels(groups)
ax.legend()

# Function to attach a text label above each bar displaying its height
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# Rotating x-axis labels to fit and prevent overlap
plt.xticks(rotation=45)

# Adjust layout to make room for the rotated x-axis labels
plt.tight_layout()

# Display the plot
plt.show()
# %%

# Data from the image
groups = ['Pre MRI', 'Post MRI']
means_arom = np.array([6.7, 0.2])
means_combination=np.array([12.9,3.4])
std_arom=np.array([1.1,0.7])
std_combination= np.array([2.5,1.3])

medians_arom = np.array([3.5, 1.3])
medians_combination=np.array([6,2])

# Indices of the groups
ind = np.arange(len(groups))

# Width of the bars
width = 0.35

fig, ax = plt.subplots()

# Plotting means with error bars representing positive standard deviation only
error1_bars = ax.bar(ind - width/2, means_arom, width, yerr=std_arom, label='Aromatherapy N=99', alpha=1, capsize=5)
error2_bars = ax.bar(ind + width/2, means_combination, width, yerr= std_combination, label='Combination N=20', alpha=1, capsize=5)

#median_1points = ax.plot(ind - width/2, medians_arom, '*', color='black')
#median_2points = ax.plot(ind + width/2, medians_combination, '*', color='black', label='Median')

# Plotting means and medians
#rects1 = ax.bar(ind - width/2, means, width, label='Mean')
#rects2 = ax.bar(ind + width/2, medians, width, label='Median')

# Adding some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time in minutes')
ax.set_title('Duration of nursing time in minutes')
ax.set_xticks(ind)
ax.set_xticklabels(groups)
ax.legend()

# Function to attach a text label above each bar displaying its height
def autolabel(rects,text=None):
    if text==None:
        text='{}'.format(height)
    for rect in rects:
        height = rect.get_height()
        ax.annotate(text,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

#autolabel(error1_bars)
#autolabel(error2_bars)


# Rotating x-axis labels to fit and prevent overlap
plt.xticks(rotation=45)

# Adjust layout to make room for the rotated x-axis labels
#plt.tight_layout()

# Display the plot
plt.show()
# %%
