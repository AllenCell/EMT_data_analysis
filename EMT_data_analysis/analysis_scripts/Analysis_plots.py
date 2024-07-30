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

# %% [markdown]
### creating a directory for figures
cwd=os.getcwd()
figdir=os.path.join(cwd, r'Figures_v2')
if not os.path.exists(figdir):
    os.makedirs(figdir)
# %% [markdown]
### importing the feature manifest and defining the color map
color_map= {'2D MG EMT 1:60 MG':"deepskyblue", '2D PLF EMT 1:60 MG':"darkmagenta",'3D MG EMT 1:60 MG':"orange"}

df=pd.read_csv(r'/allen/aics/assay-dev/users/Nivedita/EMT/EMT_deliverable/BF_colony_mask/Manifests/Entire_compiled_feature_manifest_382_movies_BF_colony_mask_v7.csv')

print(f'Total number of movies in the manifest={df.fms_id.nunique()}')

#%%[markdown]
### Filtering the dataframe to remove movies with extra cells or extra colonies in the FOV

df_f=df[df.Extracells_or_extra_colony_inFOV==0]
df_f['gene']=df_f['gene'].apply(lambda x: 'EOMES' if 'EOMES' in x else x)
print(f'No. of movies for analysis post filtering ={df_f.fms_id.nunique()} ')
# %%
### Plotting mean intensity plots for each gene
intdir=os.path.join(figdir, r'Intensity_plots')
if not os.path.exists(intdir):
    os.makedirs(intdir)

#filtering to 10 z-slices
df_z=df_f[(df_f.z_norm>=0) & (df_f.z_norm<10)]

df_int=df_z.groupby(['Condition','gene','fms_id','Timepoint']).agg({'total_intensity':'sum','area_pixels':'sum'}).reset_index()
df_int['mean_intensity']=df_int['total_intensity']/df_int['area_pixels']
df_int['time_hr']=df_int['Timepoint']*(30/60)

#Plotting mean intensity
for g, d_g in df_int.groupby('gene'):
    ###----------------------mean intensity---------------------###
    n=d_g.fms_id.nunique()
    
    fig,ax=plt.subplots(1,1)
    sns.lineplot(d_g, x='time_hr', y='mean_intensity', hue='Condition', palette=color_map, errorbar=('pi', 50), estimator=np.nanmean)
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
    ###----------------------mean intensity---------------------###
    n=d_g.fms_id.nunique()
    
    fig,ax=plt.subplots(1,1)
    sns.lineplot(d_g, x='time_hr', y='norm_intensity', hue='Condition', palette=color_map, errorbar=('pi', 50), estimator=np.nanmean)
    plt.ylabel('Normalized mean intensity (a.u.)', fontsize=14)
    plt.xlabel('Time (hr)', fontsize=14)
    plt.title(f'Gene: {g}', fontsize=14)
    
    plt.rcParams.update({'font.size':14})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
    plt.savefig(fr'{intdir}/Normalized_Mean_intensity_plot_{g}_n{n}_mean_line.pdf', dpi=600, transparent=True) 


# %%[markdown]
### Plotting migration timing box plots - [compiled for genes- EOMES_h2b_sox_CDH1]
gene_list=['CDH1','HIST1H2BJ','SOX2','EOMES']
df_f=df_f.sort_values('Timepoint')
df_summary=df_f.groupby('fms_id').agg('first').reset_index()

# Plotting migration box plots 
df_g=df_summary[df_summary.gene.isin(gene_list)]
n_m=df_g.fms_id.nunique()
df_g=df_g.sort_values('Condition')
fig_mig=px.box(df_g, x='Condition', y='Migration_hr', color='Condition', color_discrete_map=color_map, points='all', template='simple_white',range_y=(15,40), width=800, height=600)
fig_mig.update_layout(yaxis_title='Migration time (hr)',font=dict(size=18))
fig_mig.show()
fig_mig.write_image(rf'{figdir}/Migration_box_plot_n{n_m}_eomes_h2b_sox_cdh1.pdf', scale=2 )
#pio.write_html(fig_mig2, r'\\allen\aics\assay-dev\users\Nivedita\EMT\EMT_deliverable\Deliverable_plots\migration_plots\Migration_box_plot_n343_eomes_h2b_sox_cdh1_tjp1_MYH10_.html')


# %%[markdown]
### Plotting area at the glass over time
df_a=df_f.groupby(['Condition','gene','fms_id','Timepoint']).agg({'Bottom_z_mip':'first'}).reset_index()
df_a['time_hr']=df_a['Timepoint']*(30/60)
gene_list=['CDH1','HIST1H2BJ','SOX2','EOMES']
df_ag=df_a[df_a.gene.isin(gene_list)]
n_a=df_ag.fms_id.nunique()

fig,ax=plt.subplots(1,1)
sns.lineplot(df_a, x='time_hr', y='Bottom_z_mip', hue='Condition', palette=color_map, errorbar=('pi', 50), estimator=np.median)
plt.ylabel('Colony area over bottom 2 Z ( $\ um^2$)', fontsize=14)
plt.xlabel('Time (hr)', fontsize=14)

plt.ylim(0,170000)
    
plt.rcParams.update({'font.size':14})
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
plt.savefig(rf'{figdir}/Bottom_area_over_time_MIP_n{n_a}_CDH_H2b_sox2_eomes_v2.pdf', transparent=True, dpi=600)
# %%

