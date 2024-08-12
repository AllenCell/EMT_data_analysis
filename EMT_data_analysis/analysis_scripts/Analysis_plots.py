#%%[markdown]
## Script to save plots in the Figure directory
### Importing the required libraries

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
# %% [markdown]
### creating a directory for figures
cwd=os.getcwd()
figdir=os.path.join(cwd, r'Figures_v3')
if not os.path.exists(figdir):
    os.makedirs(figdir)
# %% [markdown]
### importing the feature manifest - ask user to input the path to feature manifest
#### currently- /allen/aics/assay-dev/users/Nivedita/EMT/EMT_deliverable/BF_colony_mask/Manifests/Entire_compiled_feature_manifest_382_movies_BF_colony_mask_v7.csv

file_path = input('Enter the file path to feature manifest: ')

# e.g. C:\Users\Bob\Desktop\example.txt
# or /home/Bob/Desktop/example.txt
print(file_path)

if os.path.exists(file_path):
    print('The file exists')

else:
    raise FileNotFoundError('No such file or directory.')

df=pd.read_csv(file_path)


print(f'Total number of movies in the manifest={df.fms_id.nunique()}')

#%%[markdown]
### Filtering the dataframe to remove movies with extra cells or extra colonies in the FOV

df_f=df[df.Extracells_or_extra_colony_inFOV==0]
df_f['gene']=df_f['gene'].apply(lambda x: 'EOMES' if 'EOMES' in x else x)
#for plotting the conditions int he order- 2D PLF EMT, 2D EMT, 3D EMT
df_f['Condition_order']=df_f['Condition'].apply(lambda x: 'a.2D PLF EMT' if 'PLF' in x else 'b.2D EMT' if '2D MG' in x else 'c.3D EMT')
color_map= {'2D MG EMT 1:60 MG':"deepskyblue", '2D PLF EMT 1:60 MG':"darkmagenta",'3D MG EMT 1:60 MG':"orange",'3D MG EMT no MG' :"darkgreen"}
color_map_order= {'a.2D PLF EMT':"darkmagenta",'b.2D EMT':"deepskyblue", 'c.3D EMT':"orange"}
print(f'No. of movies for analysis post filtering ={df_f.fms_id.nunique()} ')
# %% [markdown]
### Plotting mean intensity plots for each gene
intdir=os.path.join(figdir, r'Intensity_plots')
if not os.path.exists(intdir):
    os.makedirs(intdir)

#filtering to 10 z-slices
df_z=df_f[(df_f.z_norm>=0) & (df_f.z_norm<10)]

df_int=df_z.groupby(['Condition_order','gene','fms_id','Timepoint']).agg({'total_intensity':'sum','area_pixels':'sum'}).reset_index()
df_int['mean_intensity']=df_int['total_intensity']/df_int['area_pixels']
df_int['time_hr']=df_int['Timepoint']*(30/60)

#Plotting mean intensity
for g, d_g in df_int.groupby('gene'):
    ###----------------------mean intensity---------------------###
    n=d_g.fms_id.nunique()
    
    fig,ax=plt.subplots(1,1)
    sns.lineplot(d_g, x='time_hr', y='mean_intensity', hue='Condition_order', palette=color_map_order, errorbar=('pi', 50), estimator=np.nanmean)
    plt.ylabel('Mean intensity (a.u.)', fontsize=14)
    plt.xlabel('Time (hr)', fontsize=14)
    plt.title(f'Gene: {g}', fontsize=14)
    
    plt.rcParams.update({'font.size':14})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
    plt.savefig(fr'{intdir}/Mean_intensity_plot_{g}_n{n}_mean_line.pdf', dpi=600, transparent=True) 

#plotting median intensity
df_norm=pd.DataFrame()
for id,df_fms in df_int.groupby('fms_id'):
    df_fms=df_fms.sort_values('Timepoint')
    int_min=df_fms.mean_intensity.min()
    int_max=df_fms.mean_intensity.max()
    df_fms['norm_intensity']=(df_fms.mean_intensity-int_min)/(int_max-int_min)
    df_norm=pd.concat([df_norm,df_fms])

for g, d_g in df_norm.groupby('gene'):
    
    n=d_g.fms_id.nunique()
    
    fig,ax=plt.subplots(1,1)
    sns.lineplot(d_g, x='time_hr', y='norm_intensity', hue='Condition_order', palette=color_map_order, errorbar=('pi', 50), estimator=np.nanmean)
    plt.ylabel('Normalized mean intensity (a.u.)', fontsize=14)
    plt.xlabel('Time (hr)', fontsize=14)
    plt.title(f'Gene: {g}', fontsize=14)
    
    plt.rcParams.update({'font.size':14})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
    plt.savefig(fr'{intdir}/Normalized_Mean_intensity_plot_{g}_n{n}_mean_line.pdf', dpi=600, transparent=True) 


# %%[markdown]
### Plotting migration timing box plots - [compiled for all genes in the manifest]
df_f=df_f.sort_values('Timepoint')
df_summary=df_f.groupby('fms_id').agg('first').reset_index()

# Plotting migration box plots 
#%%
n_m=df_summary.fms_id.nunique()

df_summary=df_summary.sort_values('Condition_order')
fig_mig=px.box(df_summary, x='Condition_order', y='Migration_hr', color='Condition_order', color_discrete_map=color_map_order, points='all', template='simple_white',range_y=(15,40), width=800, height=600)
fig_mig.update_layout(yaxis_title='Migration time (hr)',font=dict(size=18))
#fig_mig.show()
fig_mig.write_image(rf'{figdir}/Migration_box_plot_n{n_m}_eomes_h2b_sox_cdh1_TJP1.pdf', scale=2 )
#pio.write_html(fig_mig2, r'\\allen\aics\assay-dev\users\Nivedita\EMT\EMT_deliverable\Deliverable_plots\migration_plots\Migration_box_plot_n343_eomes_h2b_sox_cdh1_tjp1_MYH10_.html')


# %%[markdown]
### Plotting area at the glass over time
df_a=df_f.groupby(['Condition_order','gene','fms_id','Timepoint']).agg({'Bottom_z_mip':'first'}).reset_index()
df_a['time_hr']=df_a['Timepoint']*(30/60)
n_a=df_a.fms_id.nunique()
fig,ax=plt.subplots(1,1)
sns.lineplot(df_a, x='time_hr', y='Bottom_z_mip', hue='Condition_order', palette=color_map_order, errorbar=('pi', 50), estimator=np.median)
plt.ylabel('Colony area over bottom 2 Z ( $\ um^2$)', fontsize=14)
plt.xlabel('Time (hr)', fontsize=14)

plt.ylim(0,170000)
    
plt.rcParams.update({'font.size':14})
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
plt.savefig(rf'{figdir}/Bottom_area_over_time_MIP_n{n_a}_CDH_H2b_sox2_eomes_TJP1.pdf', transparent=True, dpi=600)
# %% [markdown]
##Plotting example of inflection point detection for Migration timing on Area at glass over time 

def plot_example(df_fms):
    df_fms=df_fms.reset_index()
    df_fms=df_fms.sort_values('Timepoint')
    raw=df_fms['Bottom_z_mip']
    raw_values=raw.values
    df_fms['smooth']=savgol_filter(raw_values,polyorder=2, window_length=40) 
    df_fms['dy2']=savgol_filter(raw_values,polyorder=2, window_length=40, deriv=2)
    d_filt=df_fms[(df_fms.Timepoint>=35)&(df_fms.Timepoint<=80)]
    index_infl=d_filt['dy2'].idxmax()
    x_p=df_fms['time_hr'][index_infl]
    y_p=df_fms['Bottom_z_mip'][index_infl]
    return df_fms['time_hr'],df_fms['Bottom_z_mip'], x_p,y_p

example_2d=df_a[df_a.fms_id=='1aa62a434b694e8bbbac34b75e6e324a']
example_3d=df_a[df_a.fms_id=='3d3935e6150e49abbaaa4632228b46da']
example_plf=df_a[df_a.fms_id=='45b7d421a4434045bc7e2b9ec679809a']

fig,ax=plt.subplots(1,1,figsize=(8,6))
x_2d,y_2d, x_p2d, y_p2d=plot_example(example_2d)
ax.plot(x_2d,y_2d, c='deepskyblue')
ax.scatter(x_p2d,y_p2d,c='black', marker='D', s=100)

x_plf,y_plf, x_pplf, y_pplf=plot_example(example_plf)
ax.plot(x_plf,y_plf, c='darkmagenta')
ax.scatter(x_pplf,y_pplf,c='black', marker='D', s=100)

x_3d,y_3d, x_p3d, y_p3d=plot_example(example_3d)
ax.plot(x_3d,y_3d, c='orange')
ax.scatter(x_p3d,y_p3d,c='black', marker='D', s=100)

plt.ylabel('Colony area- bottom 2 Z ( $\ um^2$)', fontsize=16)
plt.xlabel('Time (hr)', fontsize=16)
plt.ylim(0,170000)
plt.xlim(-1,50)
plt.tight_layout()
plt.savefig(rf'{figdir}/Example_area_colony_2dmg_plf_and3d_combined_v2.pdf', dpi=600, transparent=True)


#%% [markdown]
### Integrated time line plots
def plot_connected_box_plot(df, Gene, column_1_name, column_2_name, ylim=[10,40], ylabel='Time (hr)'):
    df_gene=df[df.gene==Gene]
    fig, axs = plt.subplots(figsize=(10, 5))
    conditions=['2D MG EMT 1:60 MG', '2D PLF EMT 1:60 MG','3D MG EMT 1:60 MG']
    for i in range(len(conditions)):
        df_con = df_gene[df_gene["Condition"]==conditions[i]]
        conditionXindex = i*4

        colors = color_map[conditions[i]]
        plt.boxplot(df_con[column_1_name], positions=[conditionXindex],patch_artist=True,widths=0.5, boxprops=dict(facecolor=colors, linewidth=1, color='k'), medianprops=dict(linestyle='-', linewidth=1, color='k'))
        plt.scatter(np.zeros(len(df_con))+conditionXindex+0.5,df_con[column_1_name],c=colors, edgecolor='white',alpha = 1, s=80)
        plt.scatter(np.zeros(len(df_con))+conditionXindex+1.5,df_con[column_2_name],c=colors,edgecolor='white',alpha = 0.7, s=80)
   
        plt.plot( [conditionXindex+0.5,conditionXindex+1.5], [df_con[column_1_name],df_con[column_2_name]], c='k',alpha = 0.3, linewidth=0.5)


        plt.boxplot(df_con[column_2_name], positions=[conditionXindex+2],patch_artist=True, widths=0.5, boxprops=dict(facecolor=colors, alpha=0.5, linewidth=1, color='k'), medianprops=dict(linestyle='-', linewidth=1, color='k'))
#plt.xlim(-0.5,7.5)
    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.title(f'{column_1_name}vs{column_2_name}')
    return fig,axs
# %%
fig1,ax1=plot_connected_box_plot(df_summary, 'EOMES', 'Eomes_time_at_max_intensity','Migration_hr')
fig1.savefig(rf'{figdir}/Connected_box_plot_Eomes_time_at_max_intensity_vs_Migration_hr.pdf', dpi=600)
# %%
fig2,ax2=plot_connected_box_plot(df_summary, 'CDH1','CDH_inflection_hr','Migration_hr')
fig2.savefig(rf'{figdir}/Connected_box_plot_CDH_inflection_hr_Migration_hr.pdf', dpi=600)
# %%
fig3,ax3=plot_connected_box_plot(df_summary, 'SOX2','Sox_t_50_decrease','Migration_hr')
fig3.savefig(rf'{figdir}/Connected_box_plot_Sox_t_50_decrease_vs_Migration_hr.pdf', dpi=600)


#%%[markdown]
### integrated timeline- difference from migration time

df_eomes=df_summary[df_summary.gene=='EOMES']
df_eomes['Difference']=df_eomes['Eomes_time_at_max_intensity']-df_eomes['Migration_hr']
df_eomes.rename(columns={'Eomes_time_at_max_intensity':'gene_metric'}, inplace=True)

df_cdh=df_summary[df_summary.gene=='CDH1']
df_cdh['Difference']=df_cdh['CDH_inflection_hr']-df_cdh['Migration_hr']
df_cdh.rename(columns={'CDH_inflection_hr':'gene_metric'}, inplace=True)

df_sox=df_summary[df_summary.gene=='SOX2']
df_sox['Difference']=df_sox['Sox_t_50_decrease']-df_sox['Migration_hr']
df_sox.rename(columns={'Sox_t_50_decrease':'gene_metric'},inplace=True)

df_comb=pd.concat([df_cdh,df_eomes,df_sox])
df_comb['Condition_order']=df_comb['Condition'].apply(lambda x: 'a.2D PLF EMT' if 'PLF' in x else 'b.2D EMT' if '2D MG' in x else 'c.3D EMT')

df_comb=df_comb.sort_values(['gene','Condition_order'], ascending=[True,False])
color_map_order= {'a.2D PLF EMT':"darkmagenta",'b.2D EMT':"deepskyblue", 'c.3D EMT':"orange"}

#Plotting migration time for different genes
#%%
df_summary=df_summary.sort_values(['gene','Condition_order'], ascending=[True,False])
fig_mig_g=px.box(df_summary, x='Migration_hr', y='gene', color='Condition_order', color_discrete_map=color_map_order, points='all', template='simple_white',range_x=(10,40),width=1000, height=1300)
fig_mig_g.update_layout(yaxis_title='Gene_Migration time', xaxis_title='Real time (hr)', font=dict(size=18))
fig_mig_g.update_layout(boxgroupgap=0.45, boxgap=0.2)
#fig_mig_g.show()
fig_mig_g.write_image(rf'{figdir}/Migration_box_plot_per_gene_all_genes.pdf', scale=2 )

#Plotting Metric for each gene
#%%
fig_metric=px.box(df_comb, x='gene_metric', y='gene', color='Condition_order', color_discrete_map=color_map_order, points='all', template='simple_white',range_x=(10,40),width=1000, height=800)
fig_metric.update_layout(yaxis_title='Metrics for each gene', xaxis_title='Real time (hr)', font=dict(size=18))
fig_metric.update_layout(boxgroupgap=0.45, boxgap=0.2)
#fig_metric.show()
fig_metric.write_image(rf'{figdir}/Metric for each gene_box_plot.pdf', scale=2 )

# Plotting the difference between metric and the migration time
fig_diff=px.box(df_comb, x='Difference', y='gene', color='Condition_order', color_discrete_map=color_map_order, points='all', template='simple_white',range_x=(-20,10), width=1000, height=800)
fig_diff.update_layout(boxgroupgap=0.45, boxgap=0.2)
fig_diff.update_layout(yaxis_title='Difference between metric and migration time',font=dict(size=18))
#fig_diff.show()
fig_diff.write_image(rf'{figdir}/Difference between metric and migration for each gene_box_plot.pdf', scale=2 )
# %%
