#%%
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import scipy
#%%
def get_p_value(data):
    #In the order of 0-1, 1-2, 0-2
    try:
        pvalues=[]
        stat,pvalue=scipy.stats.ttest_ind(data[0],data[1])
        pvalues.append(pvalue)
        stat,pvalue=scipy.stats.ttest_ind(data[1],data[2])
        pvalues.append(pvalue)
        stat,pvalue=scipy.stats.ttest_ind(data[0],data[2])
        pvalues.append(pvalue)
    except:
        pvalues=[]
        stat,pvalue=scipy.stats.ttest_ind(data[0],data[1])
        pvalues.append(pvalue)
    return pvalues
def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

def bonferroni_correction(pvalue,n):
    if pvalue/n <= 0.0001:
        return "****"
    elif pvalue/n <= 0.001:
        return "***"
    elif pvalue/n <= 0.01:
        return "**"
    elif pvalue/n <= 0.05:
        return "*"
    return "ns"



# %% Claim Bland Altman plot
def bland_altman_plot(data1, data2, Print_title,*args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    CI_low    = md - 1.96*sd
    CI_high   = md + 1.96*sd

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='black', linestyle='-')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.title(r"$\mathbf{Bland-Altman}$" + " " + r"$\mathbf{Plot}$"+f"\n {Print_title}")
    plt.xlabel("Means")
    plt.ylabel("Difference")
    plt.ylim(md - 3.5*sd, md + 3.5*sd)

    xOutPlot = np.min(mean) + (np.max(mean)-np.min(mean))*1.14

    plt.text(xOutPlot, md - 1.96*sd, 
        r'-1.96SD:' + "\n" + "%.2f" % CI_low, 
        ha = "center",
        va = "center",
        )
    plt.text(xOutPlot, md + 1.96*sd, 
        r'+1.96SD:' + "\n" + "%.2f" % CI_high, 
        ha = "center",
        va = "center",
        )
    plt.text(xOutPlot, md, 
        r'Mean:' + "\n" + "%.2f" % md, 
        ha = "center",
        va = "center",
        )
    plt.subplots_adjust(right=0.85)

    return md, sd, mean, CI_low, CI_high

#%%
dirname='C:\Research\MRI\MP_EPI'
df=pd.read_csv(os.path.join(dirname,r'mapping_Jan.csv'))
#Read one only

#CIRC_ID_list=['CIRC_00373','CIRC_00381','CIRC_00382','CIRC_00398','CIRC_00405','CIRC_00393','CIRC_00407']    
CIRC_ID_list=['CIRC_00373','CIRC_00381','CIRC_00382','CIRC_00398','CIRC_00405','CIRC_00407','CIRC_00419', 'CIRC_00429','CIRC_00452','CIRC_00446']    
#CIRC_ID_list=['CIRC_00291','CIRC_00292','CIRC_00296','CIRC_00302','CIRC_00303']              
#ID_list=['MP01','MP02','MP03','T1-MOLLI','T2-FLASH']
ID_list=['MP01','MP02','MP03','T1_MOLLI','T1_MOLLI_FB','T2_FLASH','T2_FLASH_FB']

keys=['CIRC_ID']
#value=[CIRC_ID_list[1]]
#df_CIRD=df[df['CIRC_ID'].str.contains('|'.join(CIRC_ID_list))]
df_CIRD=df
#%%
df_t1=df_CIRD.copy()
searchfor_T1=[ID_list[i] for i in [0,3,4]]
df_t1=df_t1[df_t1['ID'].str.contains('|'.join(searchfor_T1),case=False)]
df_t2=df_CIRD.copy()
searchfor_T2=[ID_list[i] for i in [1,5,6]]
df_t2=df_t2[df_t2['ID'].str.contains('|'.join(searchfor_T2),case=False)]
#%%
df_t1=df_CIRD.copy()
searchfor_T1=[ID_list[i] for i in [0,3]]
df_t1=df_t1[df_t1['ID'].str.contains('|'.join(searchfor_T1),case=False)]
df_t2=df_CIRD.copy()
searchfor_T2=[ID_list[i] for i in [1,5]]
df_t2=df_t2[df_t2['ID'].str.contains('|'.join(searchfor_T2),case=False)]
#%%

for seg in ['global','septal','lateral']:
    df_t1=df_CIRD.copy()
    searchfor=[ID_list[i] for i in [0,3]]
    df_t1=df_t1[df_t1['ID'].str.contains('|'.join(searchfor))]
    #
    searchfor_slice=[f'Slice 0 {seg}',f'Slice 1 {seg}',f'Slice 2 {seg}']
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 9))
    for y_ind,str_read in enumerate(searchfor_slice):
        column=['ID']
        column.append(str_read)
        df_slice0=df_t1[column]
        mean=[]
        var=[]
        all_data=[]
        for sample,search_read in enumerate(searchfor_T1):
            if sample==0:
                df_plot=df_slice0[df_slice0['ID'].str.contains(search_read)]
                df_plot = df_plot[~df_plot.isnull()]
            elif sample>0:
                df_plot=df_slice0[df_slice0['ID']==search_read]
                df_plot = df_plot[~df_plot.isnull()]
            all_data.append(df_plot[str_read].tolist())
            mean.append(np.mean(df_plot[str_read]))
            var.append(np.std(df_plot[str_read]))
        bplot1 = axs[0,y_ind].boxplot(all_data,
                            patch_artist=True,  # fill with color
                            labels=searchfor_T1)  # will be used to label x-ticks
        axs[0,y_ind].set_title(f'{searchfor_slice[y_ind]}')

    searchfor_slice=[f'Slice 0 {seg}',f'Slice 1 {seg}',f'Slice 2 {seg}']
    for y_ind,str_read in enumerate(searchfor_slice):
        column=['ID']
        column.append(str_read)
        df_slice0=df_t2[column]
        mean=[]
        var=[]
        all_data=[]
        for sample,search_read in enumerate(searchfor_T2):
            if sample==0:
                df_plot=df_slice0[df_slice0['ID'].str.contains(search_read)]
                df_plot = df_plot[~df_plot.isnull()]
            elif sample>0:
                df_plot=df_slice0[df_slice0['ID']==search_read]
                df_plot = df_plot[~df_plot.isnull()]
            all_data.append(df_plot[str_read].tolist())
            mean.append(np.mean(df_plot[str_read]))
            var.append(np.std(df_plot[str_read]))
        bplot1 = axs[1,y_ind].boxplot(all_data,
                            patch_artist=True,  # fill with color
                            labels=searchfor_T2)  # will be used to label x-ticks
        axs[1,y_ind].set_title(f'{searchfor_slice[y_ind]}')
    print('')
    plt.show()


# %%
#PLOT GLOBAL
#

searchfor_slice=['global','septal','lateral']
plt.close()
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 9))
for y_ind,str_read in enumerate(searchfor_slice):
    column=['ID']
    column.append(str_read)
    df_slice0=df_t1[column]
    mean=[]
    var=[]
    all_data=[]
    for sample,search_read in enumerate(searchfor_T1):
        if sample==0:
            df_plot=df_slice0[df_slice0['ID'].str.contains(search_read)]
            df_plot = df_plot[~df_plot.isnull()]
        elif sample>0:
            df_plot=df_slice0[df_slice0['ID']==search_read]
            df_plot = df_plot[~df_plot.isnull()]
        all_data.append(df_plot[str_read].tolist())
        mean.append(np.mean(df_plot[str_read]))
        var.append(np.std(df_plot[str_read]))
    bplot1 = axs[0,y_ind].boxplot(all_data,
                        patch_artist=True,  # fill with color
                        labels=searchfor_T1)  # will be used to label x-ticks
    axs[0,y_ind].set_title(f'{searchfor_slice[y_ind]}')
for y_ind,str_read in enumerate(searchfor_slice):
    column=['ID']
    column.append(str_read)
    df_slice0=df_t2[column]
    mean=[]
    var=[]
    all_data=[]
    for sample,search_read in enumerate(searchfor_T2):
        if sample==0:
            df_plot=df_slice0[df_slice0['ID'].str.contains(search_read)]
            df_plot = df_plot[~df_plot.isnull()]
        elif sample>0:
            df_plot=df_slice0[df_slice0['ID']==search_read]
            df_plot = df_plot[~df_plot.isnull()]
        all_data.append(df_plot[str_read].tolist())
        mean.append(np.mean(df_plot[str_read]))
        var.append(np.std(df_plot[str_read]))
    bplot1 = axs[1,y_ind].boxplot(all_data,
                        patch_artist=True,  # fill with color
                        labels=searchfor_T2)  # will be used to label x-ticks
    axs[1,y_ind].set_title(f'{searchfor_slice[y_ind]}')
print('')
plt.show()
#%%
#Plot the mid slice
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 9))
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 9))

# for y_ind,seg in enumerate(['global','septal','lateral']):
for y_ind,seg in enumerate(['global']):
    searchfor_slice=[f'Slice 0 {seg}',f'Slice 1 {seg}',f'Slice 2 {seg}']

    str_read=searchfor_slice[1]
    column=['ID']
    column.append(str_read)
    df_slice0=df_t1[column]
    mean=[]
    var=[]
    all_data=[]
    for sample,search_read in enumerate(searchfor):
        if sample==0:
            df_plot=df_slice0[df_slice0['ID'].str.contains(search_read)]
            df_plot = df_plot[~df_plot.isnull()]
        elif sample>0:
            df_plot=df_slice0[df_slice0['ID']==search_read]
            df_plot = df_plot[~df_plot.isnull()]
        all_data.append(df_plot[str_read].tolist())
        mean.append(np.mean(df_plot[str_read]))
        var.append(np.std(df_plot[str_read]))
    p_values=get_p_value(all_data)
    pvalue_asterisks=[convert_pvalue_to_asterisks(p) for p in p_values ]
    bplot1 = axs[0,y_ind].boxplot(all_data,
                        patch_artist=True,  # fill with color
                        labels=searchfor)  # will be used to label x-ticks
    y_position =   max(max(all_data)) 
    for idx, pval in enumerate(pvalue_asterisks):
        axs[1,y_ind].text(x=idx, y=y_position, s=pval)
    axs[0,y_ind].set_title(f'{seg}')
    df_t2=df_CIRD.copy()
    searchfor=[ID_list[i] for i in [1,5,6]]
    df_t2=df_t2[df_t2['ID'].str.contains('|'.join(searchfor))]
    searchfor_slice=[f'Slice 0 {seg}',f'Slice 1 {seg}',f'Slice 2 {seg}']
    str_read=searchfor_slice[1]
    column=['ID']
    column.append(str_read)
    df_slice0=df_t2[column]
    mean=[]
    var=[]
    all_data=[]
    for sample,search_read in enumerate(searchfor):
        if sample==0:
            df_plot=df_slice0[df_slice0['ID'].str.contains(search_read)]
            df_plot = df_plot[~df_plot.isnull()]
        elif sample>0:
            df_plot=df_slice0[df_slice0['ID']==search_read]
            df_plot = df_plot[~df_plot.isnull()]
        all_data.append(df_plot[str_read].tolist())
        mean.append(np.mean(df_plot[str_read]))
        var.append(np.std(df_plot[str_read]))
    p_values=get_p_value(all_data)
    pvalue_asterisks=[convert_pvalue_to_asterisks(p) for p in p_values ]
    bplot1 = axs[1,y_ind].boxplot(all_data,
                        patch_artist=True,  # fill with color
                        labels=searchfor)  # will be used to label x-ticks
    y_position =   max(max(all_data)) 
    for idx, pval in enumerate(pvalue_asterisks):
        axs[1,y_ind].text(x=idx, y=y_position, s=pval)
    axs[1,y_ind].set_title(f'{seg}')
    print('')
plt.show()

#%%
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
#The global value
# for y_ind,seg in enumerate(['global','septal','lateral']):
for y_ind,seg in enumerate(['global']):
    searchfor_slice=[f'{seg}',f'{seg}',f'{seg}']

    str_read=seg
    column=['ID']
    column.append(str_read)
    df_slice0=df_t1[column]
    mean=[]
    var=[]
    all_data_T1=[]
    for sample,search_read in enumerate(searchfor_T1):
        if sample==0:
            df_plot=df_slice0[df_slice0['ID'].str.contains(search_read)]
            df_plot = df_plot[~df_plot.isnull()]
        elif sample>0:
            df_plot=df_slice0[df_slice0['ID']==search_read]
            df_plot = df_plot[~df_plot.isnull()]
        all_data_T1.append(df_plot[str_read].tolist())
        mean.append(np.mean(df_plot[str_read]))
        var.append(np.std(df_plot[str_read]))
    p_values=get_p_value(all_data_T1)
    pvalue_asterisks=[convert_pvalue_to_asterisks(p) for p in p_values ]
    print('T1', p_values)
    bplot1 = axs[0].boxplot(all_data_T1,
                        patch_artist=True,
                        labels=searchfor_T1)  # will be used to label x-ticks
    axs[0].set_ylabel('T1_value')
    y_position =   max(max(all_data_T1)) 
    x1, x2,x3 = 1,2, 3

    y, h, col = max(map(max, all_data_T1))*1.1, max(map(max, all_data_T1))*0.05, 'k'

    axs[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    axs[0].plot([x1, x1, x3, x3], [y, y+2*h, y+2*h, y], lw=1.5, c=col)
    axs[0].plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
    axs[0].text((x1+x2)*.5, y+h, pvalue_asterisks[0], ha='center', va='bottom', color=col)
    axs[0].text((x2+x3)*.5, y+h, pvalue_asterisks[1], ha='center', va='bottom', color=col)
    axs[0].text((x1+x3)*.5, y+2*h, pvalue_asterisks[2], ha='center', va='bottom', color=col)
    print('T1:'+f'{pvalue_asterisks}')
    axs[0].set_title('Comparison of MP-EPI and T1-MOLLI')
    axs[0].set_ylim([950,1900])
    #axs[0].set_ylim=(0,max(all_data_T1[0])*1.1)
    searchfor_slice=[f'{seg}',f'{seg}',f'{seg}']
    str_read=seg
    column=['ID']
    column.append(str_read)
    df_slice0=df_t2[column]
    mean=[]
    var=[]
    all_data_T2=[]
    for sample,search_read in enumerate(searchfor_T2):
        if sample==0:
            df_plot=df_slice0[df_slice0['ID'].str.contains(search_read)]
            df_plot = df_plot[~df_plot.isnull()]
        elif sample>0:
            df_plot=df_slice0[df_slice0['ID']==search_read]
            df_plot = df_plot[~df_plot.isnull()]
        all_data_T2.append(df_plot[str_read].tolist())
        mean.append(np.mean(df_plot[str_read]))
        var.append(np.std(df_plot[str_read]))
    p_values=get_p_value(all_data_T2)
    print('T2', p_values)
    pvalue_asterisks=[convert_pvalue_to_asterisks(p) for p in p_values ]
    bplot1 = axs[1].boxplot(all_data_T2,
                        patch_artist=True,  # fill with color
                        labels=searchfor_T2)  # will be used to label x-ticks
    y_position =   max(max(all_data_T2)) 
    x1, x2,x3 = 1,2, 3
    print('T2:'+f'{pvalue_asterisks}')
    y, h, col = y_position*1.1, y_position*0.05, 'k'
    axs[1].set_ylim([30,65])
    axs[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    axs[1].plot([x1, x1, x3, x3], [y, y+2*h, y+2*h, y], lw=1.5, c=col)
    axs[1].plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
    axs[1].set_ylabel('T2_value')
    axs[1].set_title('Comparison of MP-EPI and T2-FLASH')
    axs[1].plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
    axs[1].text((x1+x2)*.5, y+h, pvalue_asterisks[0], ha='center', va='bottom', color=col)
    axs[1].text((x2+x3)*.5, y+h, pvalue_asterisks[1], ha='center', va='bottom', color=col)
    axs[1].text((x1+x3)*.5, y+2*h, pvalue_asterisks[2], ha='center', va='bottom', color=col)
    #axs[1].set_ylim=((0,max(all_data_T2[0])*1.1))
    #axs[1].text((x1+x3)*.5, y+h, f'p={p_values[2]:.2f}', ha='center', va='bottom', color=col)
    #axs[1].set_title(f'{seg}')

plt.show()
#%%
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
#The global value
# for y_ind,seg in enumerate(['global','septal','lateral']):
for y_ind,seg in enumerate(['global']):
    searchfor_slice=[f'{seg}',f'{seg}',f'{seg}']

    str_read=seg
    column=['ID']
    column.append(str_read)
    df_slice0=df_t1[column]
    mean=[]
    var=[]
    all_data_T1=[]
    for sample,search_read in enumerate(searchfor_T1):
        if sample==0:
            df_plot=df_slice0[df_slice0['ID'].str.contains(search_read)]
            df_plot = df_plot[~df_plot.isnull()]
        elif sample>0:
            df_plot=df_slice0[df_slice0['ID']==search_read]
            df_plot = df_plot[~df_plot.isnull()]
        all_data_T1.append(df_plot[str_read].tolist())
        mean.append(np.mean(df_plot[str_read]))
        var.append(np.std(df_plot[str_read]))
    p_values=get_p_value(all_data_T1)
    pvalue_asterisks=[convert_pvalue_to_asterisks(p) for p in p_values ]
    print('T1', p_values)
    bplot1 = axs[0].boxplot(all_data_T1,
                        patch_artist=True,
                        labels=searchfor_T1)  # will be used to label x-ticks
    axs[0].set_ylabel('T1_value')
    y_position =   max(max(all_data_T1)) 
    x1, x2,x3 = 1,2, 3

    y, h, col = max(map(max, all_data_T1))*1.1, max(map(max, all_data_T1))*0.05, 'k'

    axs[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    #axs[0].plot([x1, x1, x3, x3], [y, y+2*h, y+2*h, y], lw=1.5, c=col)
    #axs[0].plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
    axs[0].text((x1+x2)*.5, y+h, pvalue_asterisks[0], ha='center', va='bottom', color=col)
    #axs[0].text((x2+x3)*.5, y+h, pvalue_asterisks[1], ha='center', va='bottom', color=col)
    #axs[0].text((x1+x3)*.5, y+2*h, pvalue_asterisks[2], ha='center', va='bottom', color=col)
    print('T1:'+f'{pvalue_asterisks}')
    axs[0].set_title('Comparison of MP-EPI and T1-MOLLI')
    axs[0].set_ylim([950,1900])
    #axs[0].set_ylim=(0,max(all_data_T1[0])*1.1)
    searchfor_slice=[f'{seg}',f'{seg}',f'{seg}']
    str_read=seg
    column=['ID']
    column.append(str_read)
    df_slice0=df_t2[column]
    mean=[]
    var=[]
    all_data_T2=[]
    for sample,search_read in enumerate(searchfor_T2):
        if sample==0:
            df_plot=df_slice0[df_slice0['ID'].str.contains(search_read)]
            df_plot = df_plot[~df_plot.isnull()]
        elif sample>0:
            df_plot=df_slice0[df_slice0['ID']==search_read]
            df_plot = df_plot[~df_plot.isnull()]
        all_data_T2.append(df_plot[str_read].tolist())
        mean.append(np.mean(df_plot[str_read]))
        var.append(np.std(df_plot[str_read]))
    p_values=get_p_value(all_data_T2)
    print('T2', p_values)
    pvalue_asterisks=[convert_pvalue_to_asterisks(p) for p in p_values ]
    bplot1 = axs[1].boxplot(all_data_T2,
                        patch_artist=True,  # fill with color
                        labels=searchfor_T2)  # will be used to label x-ticks
    y_position =   max(max(all_data_T2)) 
    x1, x2,x3 = 1,2, 3
    print('T2:'+f'{pvalue_asterisks}')
    y, h, col = y_position*1.1, y_position*0.05, 'k'
    axs[1].set_ylim([30,65])
    axs[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    #axs[1].plot([x1, x1, x3, x3], [y, y+2*h, y+2*h, y], lw=1.5, c=col)
    #axs[1].plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
    axs[1].set_ylabel('T2_value')
    axs[1].set_title('Comparison of MP-EPI and T2-FLASH')
    #axs[1].plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
    axs[1].text((x1+x2)*.5, y+h, pvalue_asterisks[0], ha='center', va='bottom', color=col)
    #axs[1].text((x2+x3)*.5, y+h, pvalue_asterisks[1], ha='center', va='bottom', color=col)
    #axs[1].text((x1+x3)*.5, y+2*h, pvalue_asterisks[2], ha='center', va='bottom', color=col)
    #axs[1].set_ylim=((0,max(all_data_T2[0])*1.1))
    #axs[1].text((x1+x3)*.5, y+h, f'p={p_values[2]:.2f}', ha='center', va='bottom', color=col)
    #axs[1].set_title(f'{seg}')

plt.show()
#%%
#bonferroni_correction for p value
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
#The global value
# for y_ind,seg in enumerate(['global','septal','lateral']):
for y_ind,seg in enumerate(['global']):
    searchfor_slice=[f'{seg}',f'{seg}',f'{seg}']

    str_read=seg
    column=['ID']
    column.append(str_read)
    df_slice0=df_t1[column]
    mean=[]
    var=[]
    all_data_T1=[]
    for sample,search_read in enumerate(searchfor_T1):
        if sample==0:
            df_plot=df_slice0[df_slice0['ID'].str.contains(search_read)]
            df_plot = df_plot[~df_plot.isnull()]
        elif sample>0:
            df_plot=df_slice0[df_slice0['ID']==search_read]
            df_plot = df_plot[~df_plot.isnull()]
        all_data_T1.append(df_plot[str_read].tolist())
        mean.append(np.mean(df_plot[str_read]))
        var.append(np.std(df_plot[str_read]))
    p_values=get_p_value(all_data_T1)
    pvalue_asterisks=[bonferroni_correction(p,len(all_data_T1[0])) for p in p_values ]
    print('T1', p_values)
    bplot1 = axs[0].boxplot(all_data_T1,
                        patch_artist=True,
                        labels=searchfor_T1)  # will be used to label x-ticks
    axs[0].set_ylabel('T1_value')
    y_position =   max(max(all_data_T1)) 
    x1, x2,x3 = 1,2, 3

    y, h, col = max(map(max, all_data_T1))*1.1, max(map(max, all_data_T1))*0.05, 'k'

    axs[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    #axs[0].plot([x1, x1, x3, x3], [y, y+2*h, y+2*h, y], lw=1.5, c=col)
    #axs[0].plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
    axs[0].text((x1+x2)*.5, y+h, pvalue_asterisks[0], ha='center', va='bottom', color=col)
    #axs[0].text((x2+x3)*.5, y+h, pvalue_asterisks[1], ha='center', va='bottom', color=col)
    #axs[0].text((x1+x3)*.5, y+2*h, pvalue_asterisks[2], ha='center', va='bottom', color=col)
    print('T1:'+f'{pvalue_asterisks}')
    axs[0].set_title('Comparison of MP-EPI and T1-MOLLI')
    axs[0].set_ylim([950,1900])
    #axs[0].set_ylim=(0,max(all_data_T1[0])*1.1)
    searchfor_slice=[f'{seg}',f'{seg}',f'{seg}']
    str_read=seg
    column=['ID']
    column.append(str_read)
    df_slice0=df_t2[column]
    mean=[]
    var=[]
    all_data_T2=[]
    for sample,search_read in enumerate(searchfor_T2):
        if sample==0:
            df_plot=df_slice0[df_slice0['ID'].str.contains(search_read)]
            df_plot = df_plot[~df_plot.isnull()]
        elif sample>0:
            df_plot=df_slice0[df_slice0['ID']==search_read]
            df_plot = df_plot[~df_plot.isnull()]
        all_data_T2.append(df_plot[str_read].tolist())
        mean.append(np.mean(df_plot[str_read]))
        var.append(np.std(df_plot[str_read]))
    p_values=get_p_value(all_data_T2)
    print('T2', p_values)
    pvalue_asterisks=[bonferroni_correction(p,len(all_data_T2[0])) for p in p_values ]
    bplot1 = axs[1].boxplot(all_data_T2,
                        patch_artist=True,  # fill with color
                        labels=searchfor_T2)  # will be used to label x-ticks
    y_position =   max(max(all_data_T2)) 
    x1, x2,x3 = 1,2, 3
    print('T2:'+f'{pvalue_asterisks}')
    y, h, col = y_position*1.1, y_position*0.05, 'k'
    axs[1].set_ylim([30,65])
    axs[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    #axs[1].plot([x1, x1, x3, x3], [y, y+2*h, y+2*h, y], lw=1.5, c=col)
    #axs[1].plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
    axs[1].set_ylabel('T2_value')
    axs[1].set_title('Comparison of MP-EPI and T2-FLASH')
    #axs[1].plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
    axs[1].text((x1+x2)*.5, y+h, pvalue_asterisks[0], ha='center', va='bottom', color=col)
    #axs[1].text((x2+x3)*.5, y+h, pvalue_asterisks[1], ha='center', va='bottom', color=col)
    #axs[1].text((x1+x3)*.5, y+2*h, pvalue_asterisks[2], ha='center', va='bottom', color=col)
    #axs[1].set_ylim=((0,max(all_data_T2[0])*1.1))
    #axs[1].text((x1+x3)*.5, y+h, f'p={p_values[2]:.2f}', ha='center', va='bottom', color=col)
    #axs[1].set_title(f'{seg}')

plt.show()
#%%
readDiff='MP03'
mean=[]
var=[]
all_data_DWI=[]
df_03=df_CIRD.copy()
df_03=df_03[df_03['ID'].str.contains(readDiff)]
for y_ind,seg in enumerate(['global']):
    searchfor_slice=[f'{seg}',f'{seg}',f'{seg}']

    str_read=seg
    column=['ID']
    column.append(str_read)
    df_slice0=df_03[column]

    all_data_DWI.append(df_slice0[str_read].tolist())
    mean.append(np.mean(df_slice0[str_read]))
    var.append(np.std(df_slice0[str_read]))
    #bplot1 = plt.boxplot(all_data_DWI,
                        #patch_artist=True,
                        #labels='DWI')  # will be used to label x-ticks
    #bplot1.set_ylabel('ADC_value')



#%%

#%%
md, sd, mean, CI_low, CI_high = bland_altman_plot(all_data_T1[0], all_data_T1[1],Print_title='MP_EPI-MOLLI')

plt.show()
#%%
md, sd, mean, CI_low, CI_high = bland_altman_plot(all_data_T1[0], all_data_T1[2],Print_title='MP_EPI-MOLLI-FB')
plt.show()

# %%
md, sd, mean, CI_low, CI_high = bland_altman_plot(all_data_T2[0], all_data_T2[1],Print_title='MP_EPI-FLASH')

plt.show()
#%%
md, sd, mean, CI_low, CI_high = bland_altman_plot(all_data_T2[0], all_data_T2[2],Print_title='MP_EPI-FLASH-FB')
plt.show()
# %%
#MOSAIC FOR THE 292
#%%
#303: 
from sklearn.linear_model import LinearRegression
Y=np.array(all_data_T1[0]).reshape((-1,1))
X=np.array(all_data_T1[1]).reshape((-1,1))

model = LinearRegression()
model=model.fit(X, Y)
#plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
r_sq = model.score(X, Y)
a=model.coef_[0]
b=model.intercept_
x=np.arange(min(X),max(X),3)
y=a*x + b
print(a,b,r_sq)

plt.plot(x,y,linestyle='dashed',label='MP-EPI')
plt.text(1000,1300,f'y={round(a[0],3)}x+{round(b[0],2)}\nR={round(r_sq,3)}')
plt.scatter(all_data_T1[1],all_data_T1[0])
#plt.errorbar(all_data_T1[0],all_data_T1[1],np.std(all_data_T1[1]), ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
plt.plot(x,x,linestyle='solid',label='Molli')
plt.xlim=((-5,x[-1]+50))
plt.xlabel('T1-MOLLI T1 (ms)')
plt.ylabel('MP-EPI T1 (ms)')
plt.title('Corrlation of T1')
plt.legend()
# %%
from sklearn.linear_model import LinearRegression
Y=np.array(all_data_T2[0]).reshape((-1,1))
X=np.array(all_data_T2[1]).reshape((-1,1))
model = LinearRegression()
model=model.fit(X, Y)
#plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
r_sq = model.score(X, Y)
a=model.coef_[0]
b=model.intercept_
x=np.arange(min(X),max(X),0.3)
y=a*x + b
print(a,b,r_sq)
plt.plot(x,y,linestyle='dashed',label='MP-EPI')
plt.text(46,44,f'y={a[0]:.3f}x+{b[0]:.3f}\nR={r_sq:.3f}')
plt.scatter(all_data_T2[1],all_data_T2[0])
#plt.errorbar(all_data_T2[0],all_data_T2[1],np.std(all_data_T2[1]), ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
plt.plot(x,x,linestyle='solid',label='Reference')
#plt.xlim=((-5,x[-1]+50))
plt.xlabel('T2-FLASH T2 (ms)')
plt.ylabel('MP-EPI T2 (ms)')
plt.title('Corrlation of T2')
plt.legend()

#%%




# %%
#
#MEAN and STD

print('MP01:',np.mean(all_data_T1[0]),'std:',np.std(all_data_T1[0]))
print('T1:',np.mean(all_data_T1[1]),'std:',np.std(all_data_T1[1]))
print('T1-FB:',np.mean(all_data_T1[2]),'std:',np.std(all_data_T1[2]))
print('MP02:',np.mean(all_data_T2[0]),'std:',np.std(all_data_T2[0]))
print('T2:',np.mean(all_data_T2[1]),'std:',np.std(all_data_T2[1]))
print('T2-FB:',np.mean(all_data_T2[2]),'std:',np.std(all_data_T2[2]))
print('DWI:',np.mean(all_data_DWI),'std:',np.std(all_data_DWI))

# %%
print('MP01:',np.mean(all_data_T1[0]),'std:',np.std(all_data_T1[0]))
print('T1:',np.mean(all_data_T1[1]),'std:',np.std(all_data_T1[1]))
print('MP02:',np.mean(all_data_T2[0]),'std:',np.std(all_data_T2[0]))
print('T2:',np.mean(all_data_T2[1]),'std:',np.std(all_data_T2[1]))
#%%
print('DWI:',np.mean(all_data_DWI),'std:',np.std(all_data_DWI))

# %%
