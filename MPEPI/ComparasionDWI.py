#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom
from scipy.optimize import curve_fit
import scipy
import scipy.io 

# %%
#information of the plot:
plt.close()

#Ref
DWI_Ref_ave_unsort=np.array([1.41,1.34,1.33,1.29,1.20,1.01,0.99])
DWI_Ref_std_unsort=np.array([0.01,0.02,0.01,0.01,0.03,0.02,0.05])

#1

DWI_ave_unsort=np.array([1.41,1.34,1.33,1.29,1.19,1.01,0.98])
DWI_std_unsort=np.array([0.01,0.02,0.01,0.01,0.03,0.02,0.05])

#DWI_ave_unsort=np.array([1.43,1.36,1.36,1.31,1.21,1.03,0.98])
#DWI_std_unsort=np.array([0.02,0.01,0.01,0.01,0.02,0.02,0.04])


arr1inds=DWI_Ref_ave_unsort.argsort()
DWI_Ref_ave=DWI_Ref_ave_unsort[arr1inds]
DWI_Ref_std=DWI_Ref_std_unsort[arr1inds]

arr1inds=DWI_ave_unsort.argsort()
DWI_ave=DWI_ave_unsort[arr1inds]
DWI_std=DWI_std_unsort[arr1inds]

x=np.arange(len(DWI_Ref_ave_unsort))
plt.errorbar(x,DWI_ave,DWI_std,marker='*',label='ADC')
plt.errorbar(x,DWI_Ref_ave,DWI_Ref_std,label='Identity')
plt.xlabel('Tubes Number')
plt.legend()
plt.title('ADC Correlation Plot')

#%%
#Plot the correlation plot
%matplotlib inline
from sklearn.linear_model import LinearRegression
X=DWI_Ref_ave.reshape((-1, 1))

model = LinearRegression()
model=model.fit(X, DWI_ave)
#plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
r_sq = model.score(X, DWI_ave)
a=model.coef_[0]
b=model.intercept_
x=np.arange(0.9,DWI_Ref_ave[-1]+0.2,0.01)
y=a*x + b
plt.plot(x,y,linestyle='dashed',label='MP-EPI')
plt.text(1.3,1,f'y={a:.3f}x+{b:.3f}\nR={r_sq:.3f}')
plt.errorbar(DWI_Ref_ave,DWI_ave,DWI_std, ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
plt.plot(x,x,linestyle='solid',label='Reference')
#plt.xlim(0.9,1.6)
#plt.ylim(0.9,1.6)
plt.xlabel('Reference ADC (ms)')
plt.ylabel('MPEPI-ADC (ms)')
plt.legend()
#plt.title('T1 Correlation Plot')



# %%

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
    plt.title(f"{Print_title}")
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

plt.figure()
md, sd, mean, CI_low, CI_high = bland_altman_plot(DWI_Ref_ave, DWI_ave,Print_title='Reference ADC, MP-EPI ADC')

plt.show()
#%%
md, sd, mean, CI_low, CI_high = bland_altman_plot(T1_Molli_ave, T1_EPI_ave,Print_title='MOLLI T1, MP-EPI T1')
plt.show()
#%%
md, sd, mean, CI_low, CI_high = bland_altman_plot(T1_SE_ave, T1_EPI_ave,Print_title='Reference T1, MP-EPI T1')

plt.show()

#%%
plt.figure()
from sklearn.linear_model import LinearRegression
X=T2_SE_ave.reshape((-1, 1))

model = LinearRegression()
model=model.fit(X, T2_EPI_ave)
#plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
r_sq = model.score(X, T2_EPI_ave)
a=model.coef_[0]
b=model.intercept_
x=np.arange(0,T2_SE_ave[-1],3)
y=a*x + b
plt.plot(x,y,linestyle='dashed',label='MP-EPI')
plt.text(115,30,f'y={a:.3f}x+{b:.3f}\nR={r_sq:.3f}')
plt.errorbar(T2_SE_ave,T2_EPI_ave,T2_EPI_std, ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
plt.plot(x,x,linestyle='solid',label='Reference')
plt.xlim=((-5,x[-1]+50))
plt.xlabel('Reference T2 (ms)')
plt.ylabel('MP-EPI T2 (ms)')
plt.legend()
#%%
plt.figure()
from sklearn.linear_model import LinearRegression
X=T2_Flash_ave.reshape((-1, 1))

model = LinearRegression()
model=model.fit(X, T2_EPI_ave)
#plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
r_sq = model.score(X, T2_EPI_ave)
a=model.coef_[0]
b=model.intercept_
x=np.arange(0,T2_Flash_ave[-1],3)
y=a*x + b
plt.plot(x,y,linestyle='dashed',label='MP-EPI')
plt.text(115,30,f'y={a:.3f}x+{b:.3f}\nR={r_sq:.3f}')
plt.errorbar(T2_Flash_ave,T2_EPI_ave,T2_EPI_std, ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
plt.plot(x,x,linestyle='solid',label='FLASH T2')
plt.xlim=((-5,x[-1]+50))
plt.xlabel('FLASH T2 (ms)')
plt.ylabel('MP-EPI T2 (ms)')
plt.legend()
#%%
#Plot the correlation plot --T2
plt.figure()
from sklearn.linear_model import LinearRegression
X=T2_SE_ave.reshape((-1, 1))

model = LinearRegression()
model=model.fit(X, T2_Flash_ave)
#plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
r_sq = model.score(X, T2_Flash_ave)
a=model.coef_[0]
b=model.intercept_
x=np.arange(0,T2_SE_ave[-1],3)
y=a*x + b
plt.plot(x,y,linestyle='dashed',label='FLASH')
plt.text(115,30,f'y={a:.3f}x+{b:.3f}\nR={r_sq:.3f}')
plt.errorbar(T2_SE_ave,T2_Flash_ave,T2_Flash_std, ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
plt.plot(x,x,linestyle='solid',label='Reference')
plt.xlim=((-5,x[-1]+50))
plt.xlabel('Reference T2 (ms)')
plt.ylabel('FLASH T2 (ms)')
plt.legend()
# %%

md, sd, mean, CI_low, CI_high = bland_altman_plot(T2_Flash_ave, T2_EPI_ave,Print_title='FLASH T2, MP-EPI T2')
plt.show()
md, sd, mean, CI_low, CI_high = bland_altman_plot(T2_SE_ave, T2_EPI_ave,Print_title='Reference T2, MP-EPI T2')
plt.show()
md, sd, mean, CI_low, CI_high = bland_altman_plot(T2_SE_ave, T2_Flash_ave,Print_title='Reference T2, FLASH T2')
plt.show()
# %%
from scipy.stats import linregress
import numpy as np
import plotly.graph_objects as go
def bland_altman_plot(data1, data2, data1_name='A', data2_name='B', subgroups=None, plotly_template='none', annotation_offset=0.05, plot_trendline=True, n_sd=1.96,*args, **kwargs):
    data1 = np.asarray( data1 )
    data2 = np.asarray( data2 )
    mean = np.mean( [data1, data2], axis=0 )
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean( diff )  # Mean of the difference
    sd = np.std( diff, axis=0 )  # Standard deviation of the difference


    fig = go.Figure()

    if plot_trendline:
        slope, intercept, r_value, p_value, std_err = linregress(mean, diff)
        trendline_x = np.linspace(mean.min(), mean.max(), 10)
        fig.add_trace(go.Scatter(x=trendline_x, y=slope*trendline_x + intercept,
                                 name='Trendline',
                                 mode='lines',
                                 line=dict(
                                        width=4,
                                        dash='dot')))
    if subgroups is None:
        fig.add_trace( go.Scatter( x=mean, y=diff, mode='markers', **kwargs))
    else:
        for group_name in np.unique(subgroups):
            group_mask = np.where(np.array(subgroups) == group_name)
            fig.add_trace( go.Scatter(x=mean[group_mask], y=diff[group_mask], mode='markers', name=str(group_name), **kwargs))



    fig.add_shape(
        # Line Horizontal
        type="line",
        xref="paper",
        x0=0,
        y0=md,
        x1=1,
        y1=md,
        line=dict(
            # color="Black",
            width=6,
            dash="dashdot",
        ),
        name=f'Mean {round( md, 2 )}',
    )
    fig.add_shape(
        # borderless Rectangle
        type="rect",
        xref="paper",
        x0=0,
        y0=md - n_sd * sd,
        x1=1,
        y1=md + n_sd * sd,
        line=dict(
            color="SeaGreen",
            width=2,
        ),
        fillcolor="LightSkyBlue",
        opacity=0.4,
        name=f'±{n_sd} Standard Deviations'
    )

    # Edit the layout
    fig.update_layout( title=f'Bland-Altman Plot for {data1_name} and {data2_name}',
                       xaxis_title=f'Average of {data1_name} and {data2_name}',
                       yaxis_title=f'{data1_name} Minus {data2_name}',
                       template=plotly_template,
                       annotations=[dict(
                                        x=1,
                                        y=md,
                                        xref="paper",
                                        yref="y",
                                        text=f"Mean {round(md,2)}",
                                        showarrow=True,
                                        arrowhead=7,
                                        ax=50,
                                        ay=0
                                    ),
                                   dict(
                                       x=1,
                                       y=n_sd*sd + md + annotation_offset,
                                       xref="paper",
                                       yref="y",
                                       text=f"+{n_sd} SD",
                                       showarrow=False,
                                       arrowhead=0,
                                       ax=0,
                                       ay=-20
                                   ),
                                   dict(
                                       x=1,
                                       y=md - n_sd *sd + annotation_offset,
                                       xref="paper",
                                       yref="y",
                                       text=f"-{n_sd} SD",
                                       showarrow=False,
                                       arrowhead=0,
                                       ax=0,
                                       ay=20
                                   ),
                                   dict(
                                       x=1,
                                       y=md + n_sd * sd - annotation_offset,
                                       xref="paper",
                                       yref="y",
                                       text=f"{round(md + n_sd*sd, 2)}",
                                       showarrow=False,
                                       arrowhead=0,
                                       ax=0,
                                       ay=20
                                   ),
                                   dict(
                                       x=1,
                                       y=md - n_sd * sd - annotation_offset,
                                       xref="paper",
                                       yref="y",
                                       text=f"{round(md - n_sd*sd, 2)}",
                                       showarrow=False,
                                       arrowhead=0,
                                       ax=0,
                                       ay=20
                                   )
                               ])
    return fig

#%%