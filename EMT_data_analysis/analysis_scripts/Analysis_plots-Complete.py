import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import kaleido
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from EMT_data_analysis.tools import io, const
from EMT_data_analysis.analysis_scripts import plot_tools
from pathlib import Path
import scikit_posthocs as sp
from scipy.stats import f_oneway, pearsonr
import statsmodels.api as sm

print("test")

warnings.filterwarnings("ignore")

# figs_dir = io.setup_base_directory_name("figures")

# df = io.load_image_analysis_extracted_features(load_from_aws=True)
df = pd.read_csv('/allen/aics/users/filip.sluzewski/Public_Repos/emt-data-analysis/resubmission_scripts/Complete EMT Data - Segmentation Data.csv', index_col=None)
figs_dir = '/allen/aics/emt/data_analysis_plots/Colony_Metrics/full_dataset_figures/'
Path(figs_dir).mkdir(parents=True, exist_ok=True)
out_type = 'svg'

# check if/why we have nan values here
df['Experimental Condition'].replace('',np.nan, inplace=True)
df.dropna(subset=['Experimental Condition'], inplace=True)

df['Experimental Condition'] = df['Experimental Condition'].apply(lambda x: x.replace('2D MG EMT 1:60 MG','2D colony EMT').replace('2D PLF EMT 1:60 MG', '2D PLF colony EMT').replace('3D MG EMT 1:60 MG', '3D lumenoid EMT'))

# drop EOMES|TBR2 data
df = df[(df['Gene']!='EOMES|TBR2')&(df['Gene']!='TBR2|EOMES')]
df['Gene']=df['Gene'].apply(lambda x: 'H2B' if 'H2B' in x else x)

df.rename(columns={'Inflection Point':'Migration Time (h)', 'Inflection Point InOut':'Migration Time InOut (h)'}, inplace=True)

df_f = df[(df['Experimental Condition']=='2D PLF colony EMT') | (df['Experimental Condition']=='2D colony EMT') | (df['Experimental Condition']=='3D lumenoid EMT')]
df_f = df_f[
    (df_f['Single Colony Or Lumenoid At Time of Migration']==True)& \
    (df_f['Absence Of Migrating Cells Coming From Colony Out Of FOV At Time Of Migration']==True)& \
    (df_f['Perturbation']=='No perturbation')& \
    (df_f['Absence Of Excessive Cell Death']==True)& \
    (df_f['Image Size Z']==30)& \
    (df_f['Fixation Status']=='Live Cells')
]

# Adding a Timepoint (h) column which converts frames into hours using  the Timelapse Interval column value
time_interval=30 #int(''.join(filter(lambda i: i.isdigit(),df_f['Timelapse Interval'].unique()[0] )))
df_f['Timepoint (h)']=df_f['Timepoint']*(time_interval/60)

# For plotting the conditions in the order- 2D PLF EMT, 2D EMT, 3D EMT
df_f['Condition order for plots']=df_f['Experimental Condition'].apply(lambda x: 'a.2D PLF EMT' if '2D PLF colony EMT' in x else 'b.2D EMT' if '2D colony EMT' in x else 'c.3D EMT')

n_filtered_movies=df_f['Movie ID'].nunique()
# print(f'No. of movies for analysis post filtering ={n_filtered_movies} ')


### -----------------------------------------------------------------------------------
# Area at Glass for entire dataset
### -----------------------------------------------------------------------------------


# print('Generating plots for Area at the glass for all three conditions and corresponding migration time estimated from the inflection of area at glass over time (Fig.5 C, D , E)')

df_a = df_f.groupby(['Condition order for plots','Gene','Movie ID','Timepoint (h)']).agg({'Area at the glass(square micrometer)':'first', 'Migration Time (h)':'first'}).reset_index()
n_a = df_a['Movie ID'].nunique()
fig,ax = plt.subplots(1,1)
# for scn, df_scn in df_a[df_a['Gene']=='TBXT'].groupby('Movie ID'):
sns.lineplot(df_a, x='Timepoint (h)', y='Area at the glass(square micrometer)', hue='Condition order for plots', palette=const.COLOR_MAP, errorbar=('pi', 50), estimator=np.median)
plt.ylabel('Colony area over bottom 2 Z ( $\ um^2$)', fontsize=14)
plt.xlabel('Time (hr)', fontsize=14)
plt.ylim(0,170000)
plt.rcParams.update({'font.size':14})
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
plt.savefig(rf'{figs_dir}/Area_at_the_glass_over_time_MIP_n{n_a}.{out_type}', transparent=True, dpi=600)

### -----------------------------------------------------------------------------------
# Area at Glass for Only H2B
### -----------------------------------------------------------------------------------

#graphing only H2B volonies area at glass
df_a_h2b = df_f[df_f['Gene']=='H2B'].groupby(['Condition order for plots','Gene','Movie ID','Timepoint (h)']).agg({'Area at the glass(square micrometer)':'first', 'Migration Time (h)':'first'}).reset_index()
n_a = df_a['Movie ID'].nunique()
fig,ax = plt.subplots(1,1)
# for scn, df_scn in df_a[df_a['Gene']=='TBXT'].groupby('Movie ID'):
sns.lineplot(df_a_h2b, x='Timepoint (h)', y='Area at the glass(square micrometer)', hue='Condition order for plots', palette=const.COLOR_MAP, errorbar=('pi', 50), estimator=np.median)
plt.ylabel('Colony area over bottom 2 Z ( $\ um^2$)', fontsize=14)
plt.xlabel('Time (hr)', fontsize=14)
plt.ylim(0,170000)
plt.rcParams.update({'font.size':14})
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
plt.savefig(rf'{figs_dir}/Area_at_the_glass_over_time_MIP_H2B_n{n_a}.{out_type}', transparent=True, dpi=600)

# plot_tools.plot_examples(
#     df_int = df_a,
#     id_plf = const.EXAMPLE_PLF,
#     id_2d = const.EXAMPLE_2D,
#     id_3d = const.EXAMPLE_3D,
#     gene = "Example",
#     metric = 'Migration Time (h)',
#     variable = 'Area at the glass(square micrometer)',
#     figs_dir = figs_dir)

### -----------------------------------------------------------------------------------
# Migration Timing For Entire Data
### -----------------------------------------------------------------------------------

df_f = df_f.sort_values('Timepoint (h)')
# Summarizing the dataframe/manifest to have one line/metric per movie
df_summary = df_f.groupby('Movie ID').agg('first').reset_index()

n_m = df_summary['Movie ID'].nunique()
df_summary = df_summary.sort_values(['Gene','Condition order for plots'])
fig_mig = px.box(df_summary, x='Condition order for plots', y='Migration Time (h)', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(15,35), width=800, height=600)
fig_mig.update_layout(yaxis_title='Migration Time (h)',font=dict(size=18))
fig_mig.write_image(rf'{figs_dir}/Migration_box_plot_n{n_m}.{out_type}', scale=2 )

### -----------------------------------------------------------------------------------
# Migration Timing For Only H@B
### -----------------------------------------------------------------------------------

n_m = df_summary[df_summary['Gene']=='H2B']['Movie ID'].nunique()
df_summary = df_summary.sort_values(by='Condition order for plots')
fig_mig = px.box(df_summary[df_summary['Gene']=='H2B'], x='Condition order for plots', y='Migration Time (h)', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(15,35), width=800, height=600)
fig_mig.update_layout(yaxis_title='Migration Time (h)',font=dict(size=18))
fig_mig.write_image(rf'{figs_dir}/Migration_box_plot_H2B_n{n_m}.{out_type}', scale=2 )

# n_m = df_summary[df_summary['Gene']=='H2B']['Movie ID'].nunique()
# df_summary = df_summary.sort_values(by='Condition order for plots')
# fig_mig = px.box(df_summary[df_summary['Gene']=='H2B'], x='Condition order for plots', y='Migration Time (h) Manual', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(15,35), width=800, height=600)
# fig_mig.update_layout(yaxis_title='Migration Time (h)',font=dict(size=18))
# fig_mig.write_image(rf'{figs_dir}/Migration_box_plot_H2B_n{n_m}_(manual).{out_type}', scale=2 )

# print('\n\n\n...statitsitcal analysis of overall migriation timing between the conditions for H2B...')
# x_mig = df_summary['Migration Time (h)'][['2D PLF' in val for val in df_summary['Experimental Condition'].values]].dropna()
# y_mig = df_summary['Migration Time (h)'][['2D colony EMT' in val for val in df_summary['Experimental Condition'].values]].dropna()
# z_mig = df_summary['Migration Time (h)'][['3D lumenoid EMT' in val for val in df_summary['Experimental Condition'].values]].dropna()

# print('2D PLF: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(x_mig), np.median(x_mig), np.std(x_mig), np.min(x_mig), np.max(x_mig)))
# print('2D EMT: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(y_mig), np.median(y_mig), np.std(y_mig), np.min(y_mig), np.max(y_mig)))
# print('3D EMT: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(z_mig), np.median(z_mig), np.std(z_mig), np.min(z_mig), np.max(z_mig)))

# plot_tools.run_statistics(x_mig,y_mig,z_mig)


### -----------------------------------------------------------------------------------
# Migration Timing Per Gene
### -----------------------------------------------------------------------------------


# print('Generating Box plots for migration timing for each gene in the dataset and fo each condition within that gene (Fig.S3)')
df_summary['gene_m'] = pd.Categorical(df_summary['Gene'], df_summary['Gene'].unique())
df_summary = df_summary.sort_values(['gene_m','Condition order for plots'])
fig_mig_g = px.box(df_summary, y='Migration Time (h)', x='gene_m', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(10,35),width=1800, height=600)
fig_mig_g.update_layout(showlegend=False)
fig_mig_g.update_layout(xaxis_title='Cell lines', yaxis_title='Migration in real time (h)', font=dict(size=18))
fig_mig_g.update_layout(boxgroupgap=0.5, boxgap=0.5)
fig_mig_g.write_image(rf'{figs_dir}/Migration_box_plot_per_gene_all_conditions.{out_type}', scale=2 )

for g, df_g in df_summary.groupby('Gene'):
    print(f'\n\n\n...statistical analysis for the migration timing per condition for gene={g}')
    x_mig = df_g['Migration Time (h)'][['2D PLF' in val for val in df_g['Experimental Condition'].values]].dropna()
    y_mig = df_g['Migration Time (h)'][['2D colony EMT' in val for val in df_g['Experimental Condition'].values]].dropna()
    z_mig = df_g['Migration Time (h)'][['3D lumenoid EMT' in val for val in df_g['Experimental Condition'].values]].dropna()

    print('2D PLF: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(x_mig), np.median(x_mig), np.std(x_mig), np.min(x_mig), np.max(x_mig)))
    print('2D EMT: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(y_mig), np.median(y_mig), np.std(y_mig), np.min(y_mig), np.max(y_mig)))
    print('3D EMT: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(z_mig), np.median(z_mig), np.std(z_mig), np.min(z_mig), np.max(z_mig)))
        
    plot_tools.run_statistics(x_mig, y_mig, z_mig)


### -----------------------------------------------------------------------------------
# MEAN INTESITITY GRAPHS
### -----------------------------------------------------------------------------------


# print('Generating plots for  mean intensity plots for each gene - Fig.6B, Fig. 6F and Fig 6 J')

# Filtering to 10 z-slices over which the mean intensity is calculated
df_z = df_f[(df_f['Normalized Z plane']>=0) & (df_f['Normalized Z plane']<10)]

# Grouping by condition and gene and each movie to get mean itnensity over time for each movie
df_int = df_z.groupby(['Experimental Condition','Condition order for plots','Gene','Movie ID','Timepoint (h)']).agg({'Total intensity per Z':'sum','Area of all cells mask per Z (pixels)':'sum'}).reset_index()
df_int['Mean Intensity']=df_int['Total intensity per Z']/df_int['Area of all cells mask per Z (pixels)']
df_int['Mean Intensity'] = df_int['Mean Intensity'].replace(0,np.nan)

# Plotting mean intensity
for g, d_g in df_int.groupby('Gene'):
    n = d_g['Movie ID'].nunique()
    
    fig,ax = plt.subplots(1,1)
    # for scn, df_scn in d_g.groupby('Movie ID'):
    sns.lineplot(d_g, x='Timepoint (h)', y='Mean Intensity', hue='Condition order for plots', palette=const.COLOR_MAP, errorbar=('pi', 50), estimator=np.nanmean)
    plt.ylabel('Mean intensity (a.u.)', fontsize=14)
    plt.xlabel('Time (h)', fontsize=14)
    plt.title(f'Gene: {g}', fontsize=14)
    
    plt.rcParams.update({'font.size':14})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
    plt.savefig(fr'{figs_dir}/Mean_intensity_plot_{g}_n{n}_mean_line.{out_type}', dpi=600, transparent=True) 


### -----------------------------------------------------------------------------------
# GENE EXPRESSION CONNECTED GRAPHS
### -----------------------------------------------------------------------------------


# print('Generating plots for Fig.6C., Fig.6 G, Fig.6 K  ')
# Filtering to 10 z-slices over which the mean intensity is calculated
df_z=df_f[(df_f['Normalized Z plane']>=0) & (df_f['Normalized Z plane']<10)]

# Grouping by condition and gene and each movie to get mean itnensity over time for each movie
df_int=df_z.groupby([
    'Experimental Condition',
    'Condition order for plots',
    'Gene',
    'Data ID',
    'Timepoint (h)'
]).agg(
    {
        'Total intensity per Z': 'sum',
        'Area of all cells mask per Z (pixels)': 'sum',
        'Time of max EOMES expression (h)': 'first',
        'Time of max TBXT expression (h)': 'first',
        'Time of inflection of E-cad expression (h)': 'first',
        'Time of half-maximal SOX2 expression (h)': 'first'
    }
).reset_index()
df_int['Mean Intensity']=df_int['Total intensity per Z']/df_int['Area of all cells mask per Z (pixels)']

Path(rf'{figs_dir}/Individual_Examples').mkdir(exist_ok=True, parents=True)
# Time of max EOMES expression (h) examples
# import pdb; pdb.set_trace()
plot_tools.plot_examples(
    df_int = df_int,
    id_plf = const.EOMES_PLF,
    id_2d = const.EOMES_2D,
    id_3d = const.EOMES_3D,
    gene = "EOMES",
    figs_dir = figs_dir+'/Individual_Examples',
    metric='Time of max EOMES expression (h)')

# Time of max TBXT expression (h) examples
plot_tools.plot_examples(
    df_int = df_int,
    id_plf = const.TBXT_PLF,
    id_2d = const.TBXT_2D,
    id_3d = const.TBXT_3D,
    gene = "TBXT",
    figs_dir = figs_dir+'/Individual_Examples',
    metric='Time of max TBXT expression (h)')

# Time of inflection of E-cad expression (h) examples-
plot_tools.plot_examples(
    df_int = df_int,
    id_plf = const.CDH_PLF,
    id_2d = const.CDH_2D,
    id_3d = const.CDH_3D,
    gene = "CDH1",
    figs_dir = figs_dir+'/Individual_Examples',
    metric='Time of inflection of E-cad expression (h)')

# Time of inflection of SOX expression (h) examples-
plot_tools.plot_examples(
    df_int = df_int,
    id_plf = const.SOX_PLF,
    id_2d = const.SOX_2D,
    id_3d = const.SOX_3D,
    gene = "SOX2",
    figs_dir = figs_dir+'/Individual_Examples',
    metric = 'Time of half-maximal SOX2 expression (h)')




# print('Generating plots for connected scatter and box plots- Fig.6 D,H, L')
df_exp = df_summary.dropna(subset=['Migration Time (h)'])
# Plotting pair-wise connected plots for Time of max EOMES expression (h) and Migration Time for EOMES (Fig.6D)
fig1,ax1 = plot_tools.plot_connected_box_plot(df_exp.dropna(subset=['Time of max EOMES expression (h)']), 'EOMES', 'Time of max EOMES expression (h)','Migration Time (h)')
fig1.savefig(rf'{figs_dir}/Connected_box_plot_Eomes_time_at_max_EOMES_expression_vs_Migration_hr.{out_type}', dpi=600)

# Plotting pair-wise connected plots for Time of max EOMES expression (h) and Migration Time for EOMES (Fig.6D)
fig1,ax1 = plot_tools.plot_connected_box_plot(df_exp.dropna(subset=['Time of max TBXT expression (h)']), 'TBXT', 'Time of max TBXT expression (h)','Migration Time (h)', ylim=[10,50])
fig1.savefig(rf'{figs_dir}/Connected_box_plot_TBXT_time_at_max_TBXT_expression_vs_Migration_hr.{out_type}', dpi=600)

# Plotting pair-wise connected plots for Time of inflection of E-cad expression (h) and Migration Time for E-Cad (Fig.6H)
fig2,ax2 = plot_tools.plot_connected_box_plot(df_exp.dropna(subset=['Time of inflection of E-cad expression (h)']), 'CDH1','Time of inflection of E-cad expression (h)','Migration Time (h)')
fig2.savefig(rf'{figs_dir}/Connected_box_plot_Ecad_time_of_inflection_E_Cad_expression_Migration_hr.{out_type}', dpi=600)

# Plotting pair-wise connected plots for Time of half-maximal SOX2 expression (h) and Migration Time for SOX2 (Fig.6L)
fig3,ax3 = plot_tools.plot_connected_box_plot(df_exp.dropna(subset=['Time of half-maximal SOX2 expression (h)']), 'SOX2','Time of half-maximal SOX2 expression (h)','Migration Time (h)')
fig3.savefig(rf'{figs_dir}/Connected_box_plot_Sox_time_of_half_maximal_vs_Migration_hr.{out_type}', dpi=600)



### -----------------------------------------------------------------------------------
# GENE EXPRESSION EXPERIMENTS
### -----------------------------------------------------------------------------------


'''
Plotting box plots for Supplementary figure S5 a. timing of expression change relative
to time of induction of EMT (time 0), b. Difference between gene metrics and migration
time, c. timing of expression change normalized between the time of induction and
migration (0-time of EMT induction and 1- time of migration)
''' 
# print('Generating plots for supplementary Fig.5. Timing of expression change')

# Compiling data for the plots
df_eomes = df_summary[df_summary.Gene=='EOMES']
df_eomes['Difference']=df_eomes['Time of max EOMES expression (h)']-df_eomes['Migration Time (h)']
df_eomes['Normalized_metric']=df_eomes['Time of max EOMES expression (h)']/df_eomes['Migration Time (h)']
df_eomes.rename(columns={'Time of max EOMES expression (h)':'gene_metric'}, inplace=True)

df_tbxt = df_summary[df_summary.Gene=='TBXT']
df_tbxt['Difference']=df_tbxt['Time of max TBXT expression (h)']-df_tbxt['Migration Time (h)']
df_tbxt['Normalized_metric']=df_tbxt['Time of max TBXT expression (h)']/df_tbxt['Migration Time (h)']
df_tbxt.rename(columns={'Time of max TBXT expression (h)':'gene_metric'}, inplace=True)

df_cdh = df_summary[df_summary.Gene=='CDH1']
df_cdh['Difference']=df_cdh['Time of inflection of E-cad expression (h)']-df_cdh['Migration Time (h)']
df_cdh['Normalized_metric']=df_cdh['Time of inflection of E-cad expression (h)']/df_cdh['Migration Time (h)']
df_cdh.rename(columns={'Time of inflection of E-cad expression (h)':'gene_metric'}, inplace=True)

df_sox = df_summary[df_summary.Gene=='SOX2']
df_sox['Difference']=df_sox['Time of half-maximal SOX2 expression (h)']-df_sox['Migration Time (h)']
df_sox['Normalized_metric']=df_sox['Time of half-maximal SOX2 expression (h)']/df_sox['Migration Time (h)']
df_sox.rename(columns={'Time of half-maximal SOX2 expression (h)':'gene_metric'},inplace=True)

df_comb=pd.concat([df_cdh,df_eomes,df_sox,df_tbxt])
df_comb=df_comb.sort_values(['Gene','Condition order for plots'], ascending=[False,True])

# Only plotting for 2D colony EMT and 3D colony EMT conditions
conditions_colony=['a.2D PLF EMT','b.2D EMT','c.3D EMT']
df_comb_f = df_comb[df_comb['Condition order for plots'].isin(conditions_colony)]
df_comb_f['sort_value'] = df_summary['Gene'].apply(lambda gene: 0 if gene=='SOX2' else 1 if gene=='TBXT' else 2 if gene=='EOMES' else 3)
df_comb_f.sort_values(by=['sort_value', 'Condition order for plots'], axis=0, inplace=True)

# Fig.S5 a.timing of expression change relative to time of induction of EMT (time 0)
fig_metric = px.box(df_comb_f, y='gene_metric', x='Gene', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(10,50),width=1200, height=600)
fig_metric.update_layout(showlegend=False)

fig_metric.update_layout(xaxis_title='Cell lines', yaxis_title='Time of expression change (h)', font=dict(size=18))
fig_metric.update_layout(boxgroupgap=0.5, boxgap=0.25)
fig_metric.write_image(rf'{figs_dir}/Timing_of_expression_change_relative_to_EMT_induction_T_0_FigS5a.{out_type}', scale=2 )

# Fig.S5 b.Difference between gene metrics and migration time- time of expression change relative to migration time
fig_difference = px.box(df_comb_f, y='Difference', x='Gene', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(-30,20),width=1200, height=600)
fig_difference.update_layout(showlegend=False)

fig_difference.update_layout(xaxis_title='Cell lines', yaxis_title='Time of expression change relative to migiration time (h)', font=dict(size=18))
fig_difference.update_layout(boxgroupgap=0.5, boxgap=0.25)
fig_difference.write_image(rf'{figs_dir}/Timing_of_expression_change_relative_to_migration_time_FigS5b.{out_type}', scale=2 )

# Fig.S5 c.timing of expression change normalized between the time of induction and migration (0-time of EMT induction and 1- time of migration)
fig_difference = px.box(df_comb_f, y='Normalized_metric', x='Gene', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(0,2.2),width=1200, height=600)
fig_difference.update_layout(showlegend=False)

fig_difference.update_layout(xaxis_title='Cell lines', yaxis_title='Time of expression change divided by migration time (h)', font=dict(size=18))
fig_difference.update_layout(boxgroupgap=0.5, boxgap=0.25)
fig_difference.write_image(rf'{figs_dir}/Timing_of_expression_change_divided_by_migration_time_FigS5c.{out_type}', scale=2 )

print('\n\n\n.......Statistical comparison for gene metric:')
for g, df_g in df_comb.groupby('Gene'):
    print(f'gene={g}')
    x = df_g['gene_metric'][['2D PLF' in val for val in df_g['Experimental Condition'].values]].dropna()
    y = df_g['gene_metric'][['2D colony EMT' in val for val in df_g['Experimental Condition'].values]].dropna()
    z = df_g['gene_metric'][['3D lumenoid EMT' in val for val in df_g['Experimental Condition'].values]].dropna()

    print('2D PLF: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(x), np.median(x), np.std(x), np.min(x), np.max(x)))
    print('2D EMT: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(y), np.median(y), np.std(y), np.min(y), np.max(y)))
    print('3D EMT: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(z), np.median(z), np.std(z), np.min(z), np.max(z)))
     
    plot_tools.run_statistics(x,y,z)

print('\n\n\n.......Statistical comparison for gene expression change relative to migration time:')
for g, df_g in df_comb.groupby('Gene'):
    print(f'gene={g}')
    x = df_g['Difference'][['2D PLF' in val for val in df_g['Experimental Condition'].values]].dropna()
    y = df_g['Difference'][['2D colony EMT' in val for val in df_g['Experimental Condition'].values]].dropna()
    z = df_g['Difference'][['3D lumenoid EMT' in val for val in df_g['Experimental Condition'].values]].dropna()

    print('2D PLF: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(x), np.median(x), np.std(x), np.min(x), np.max(x)))
    print('2D EMT: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(y), np.median(y), np.std(y), np.min(y), np.max(y)))
    print('3D EMT: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(z), np.median(z), np.std(z), np.min(z), np.max(z)))
     
    plot_tools.run_statistics(x,y,z)

print('\n\n\n.......Statistical comparison for gene expression change divided by migration time:')
for g, df_g in df_comb.groupby('Gene'):
    print(f'gene={g}')
    x = df_g['Normalized_metric'][['2D PLF' in val for val in df_g['Experimental Condition'].values]].dropna()
    y = df_g['Normalized_metric'][['2D colony EMT' in val for val in df_g['Experimental Condition'].values]].dropna()
    z = df_g['Normalized_metric'][['3D lumenoid EMT' in val for val in df_g['Experimental Condition'].values]].dropna()

    print('2D PLF: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(x), np.median(x), np.std(x), np.min(x), np.max(x)))
    print('2D EMT: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(y), np.median(y), np.std(y), np.min(y), np.max(y)))
    print('3D EMT: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(z), np.median(z), np.std(z), np.min(z), np.max(z)))
     
    plot_tools.run_statistics(x,y,z)

### -----------------------------------------------------------------------------------
# COLLAGENASE EXPERIMENTS
### -----------------------------------------------------------------------------------



# Collagenase Experiment Graphs
Path(rf'{figs_dir}/Collagenase').mkdir(exist_ok=True, parents=True)
df_coll = df[df['Perturbation']=='Collagenase']

df_summary = df_coll.drop_duplicates(subset=['Movie ID'])

n_m = df_summary['Movie ID'].nunique()
# print(n_m, 'movies found')
df_summary['sort_value'] = df_summary['Drug Concentration'].apply(lambda c: float(c.split()[0]) if 'HBSS' not in c else -1)
df_summary['Collagenease concentration (ug/mL)'] = [float(c.split(' ')[0])  if 'HBSS' not in c else 0 for c in df_summary['Drug Concentration'].values]
df_summary = df_summary.sort_values(by=['sort_value'], axis=0)
# df_summary['gene_m'] = pd.Categorical(df_summary['Gene'], df_summary['Gene'].unique())
# df_summary = df_summary.sort_values(['gene_m','Drug Concentration'])

# print('Generating Box plots for migration timing for each gene in the dataset collagenase')
print('\n\n\n.......Statistical comparison for migration time with collagenase treatment:')
for gene, df_gene in df_summary.groupby('Gene'):
    color_map={tgt:'orange' for tgt in df_gene['Drug Concentration'].unique()}
    fig_mig_g = px.box(df_gene, y='Migration Time (h)', x='Drug Concentration', color='Drug Concentration', color_discrete_map=color_map, points='all', template='simple_white',range_y=(10,40),width=800, height=600)
    fig_mig_g.update_layout(showlegend=False)
    fig_mig_g.update_layout(xaxis_title='Cell lines', yaxis_title='Migration in real time (h)', font=dict(size=18))
    fig_mig_g.update_traces(width=0.6)
    fig_mig_g.update_layout(boxgroupgap=0.4, boxgap=0.4)
    fig_mig_g.write_image(rf'{figs_dir}/Collagenase/Migration_box_plot_{gene}_per_conditions.{out_type}', scale=2 )

    print('Gene: ', gene)
    key = {g:i for i, g in enumerate(df_gene['Drug Concentration'].unique())}
    concentrations = list(key.keys())
    migration = [[]]*len(concentrations)

    for g, d_g in df_gene.groupby('Drug Concentration'):
        migration[key[g]] = d_g['Migration Time (h)'].values

    for c, v in zip(concentrations, migration):
        print('{5}: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(v), np.median(v), np.std(v), np.min(v), np.max(v), c))

    print('Concentrations:')
    print(' '.join(['{0} n:{1} |'.format(t, len(v)) for t, v in zip(concentrations, migration)]))
    print('Significant difference between the distributions: applying post hoc with Holm-Bonferroni adjustment of p-value')    
    
    print(sp.posthoc_mannwhitney(migration, p_adjust = 'holm'))


    # Calculating statistics for downward trend in collagenase concentrations vs migration time

    X = df_gene['Collagenease concentration (ug/mL)']
    Y = df_gene['Migration Time (h)']

    # It's important to add a constant (intercept) to the model
    X = sm.add_constant(X)
    
    # import pdb; pdb.set_trace()

    # Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(Y, X)
    results = model.fit()

    # Print the full summary of the regression results
    print(results.summary())

    # Interpretation of key results from the linear regression
    print("\n--- Interpretation of Linear Regression Results ---")
    slope_p_value = results.pvalues['Collagenease concentration (ug/mL)']
    r_squared = results.rsquared
    slope_coeff = results.params['Collagenease concentration (ug/mL)']

    print(f"R-squared: {r_squared:.4f}")
    print(f"Slope (Coefficient for concentration): {slope_coeff:.4f}")
    print(f"P-value for the slope: {slope_p_value:.4g}") # Using 'g' for scientific notation if needed

    alpha = 0.05
    if slope_p_value < alpha:
        print(f"\nConclusion: The p-value for the slope is less than {alpha}, indicating a statistically significant linear relationship between drug concentration and migration time.")
        print(f"On average, for each 1 ug/mL increase in drug concentration, the migration time changes by {slope_coeff:.2f} hours.")
    else:
        print("\nConclusion: The p-value for the slope is not less than 0.05, so we cannot conclude there is a significant linear relationship.")




### -----------------------------------------------------------------------------------
# CRISPRi EXPERIMENTS
### -----------------------------------------------------------------------------------


# CRISPR knockdown experiments
(Path(figs_dir) / 'CRISPRi').mkdir(parents=True, exist_ok=True)
df_f = df[df['Gene']=='CLYBL'].drop_duplicates(subset=['Movie ID'])

df_f = df_f[
    (df_f['Single Colony Or Lumenoid At Time of Migration']==True)& \
    (df_f['Absence Of Migrating Cells Coming From Colony Out Of FOV At Time Of Migration']==True)& \
    (df_f['Absence Of Excessive Cell Death']==True)& \
    (df_f['Image Size Z']==30)& \
    (df_f['Fixation Status']=='Live Cells')
]

df_summary = df_f.groupby('Movie ID').agg('first').reset_index()
df_summary['Colony'] = df_summary['Experimental Condition'].apply(lambda s: s.split(' CRISPRi ')[0] if 'CRISPRi' in s else s)
df_summary = df_summary[df_summary['Colony']=='3D lumenoid EMT']
df_summary['Knockdown'] = df_summary['Experimental Condition'].apply(lambda s: s.split(' CRISPRi ')[-1] if 'CRISPRi' in s else 'Control')
df_summary['Target'] = df_summary['Knockdown'].apply(lambda s: s.replace(' g1','').replace(' g2','').replace(' g3',''))
# import pdb; pdb.set_trace()

print('Generating Box plots for migration timing for CLYBL gene knockdown experiment')
df_summary['Condition order for plots']=df_summary['Knockdown'].apply(lambda x: 'a.Control' if 'Control' in x else 'b.Seq Scr' if 'seq' in x else f'c.{x}' if 'g1' in x else f'd.{x}' if 'g2' in x else f'e.{x}')
df_summary = df_summary.sort_values(['Condition order for plots'])
# import pdb; pdb.set_trace()

print('\n\n\n.......Statistical comparison for migration time with gene knockdowns:')

for gene in ['TBXT', 'Snail']:
    df_gene = df_summary[(df_summary['Target']=='Control')|(df_summary['Target']=='seq scr')|(df_summary['Target']==gene)]

    color_map={tgt:'orange' for tgt in df_gene['Condition order for plots'].unique()}

    fig_mig_g = px.box(df_gene, y='Migration Time (h)', x='Knockdown', points='all', color='Condition order for plots', color_discrete_map=color_map, template='simple_white',range_y=(10,35),width=800, height=600)
    fig_mig_g.update_layout(showlegend=False)
    fig_mig_g.update_layout(xaxis_title='Cell lines', yaxis_title='Migration in real time (h)', font=dict(size=18))
    fig_mig_g.update_traces(width=0.5)
    fig_mig_g.update_layout(boxgroupgap=0.5, boxgap=0.5)
    fig_mig_g.write_image(rf'{figs_dir}/CRISPRi/Migration_box_plot_CRISPRi_for_{gene}_3D_lumenoid.{out_type}', scale=2 )

    print('Gene: ', gene)
    key = {g:i for i, g in enumerate(df_gene['Knockdown'].unique())}
    targets = list(key.keys())
    values = [[]]*len(targets)

    for g, d_g in df_gene.groupby('Knockdown'):
        values[key[g]] = d_g['Migration Time (h)'].values

    for c, v in zip(targets, values):
        print('{5}: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(v), np.median(v), np.std(v), np.min(v), np.max(v), c))

    print('Targets:')
    print(' '.join(['{0} n:{1} |'.format(t, len(v)) for t, v in zip(targets, values)]))
    print('Significant difference between the distributions: applying post hoc with Holm-Bonferroni adjustment of p-value')
    print(sp.posthoc_mannwhitney(values, p_adjust = 'holm'))


### -----------------------------------------------------------------------------------
# ZO1 Heatmaps
### -----------------------------------------------------------------------------------


# print('Generating Heatmaps for ZO1 - Fig.7 and Fig. S6 ')
# Filtering the dataset to only ZO1 data
# df_zo = df_f[df_f.Gene=='TJP1']

# df_zo_examples = df_zo[df_zo['Movie ID'].isin(const.EXAMPLE_ZO1_IDS)]

# Generating and saving the heatmaps
# plot_tools.Intensity_over_z(df_zo, figs_dir=figs_dir)


### -----------------------------------------------------------------------------------
# Inside Outside Migration
### -----------------------------------------------------------------------------------

# print('Generating plots for inside-outside classification and migration time (Fig.5 G, H ,I)')


df_f = df[(df['Gene']=='H2B') & (df['Experimental Condition']=='3D lumenoid EMT')]
df_f = df_f[
    (df_f['Single Colony Or Lumenoid At Time of Migration']==True)& \
    (df_f['Absence Of Migrating Cells Coming From Colony Out Of FOV At Time Of Migration']==True)& \
    (df_f['Perturbation']=='No perturbation')& \
    (df_f['Absence Of Excessive Cell Death']==True)& \
    (df_f['Image Size Z']==30)& \
    (df_f['Fixation Status']=='Live Cells')
]
df_f['Migration Time InOut (h)'].replace('',np.nan, inplace=True)
df_f = df_f.dropna(subset=['Migration Time InOut (h)'])

# Adding a Timepoint (h) column which converts frames into hours using  the Timelapse Interval column value
time_interval=30 #int(''.join(filter(lambda i: i.isdigit(),df_f['Timelapse Interval'].unique()[0] )))
df_f['Timepoint (h)']=df_f['Timepoint']*(time_interval/60)

# For plotting the conditions in the order- 2D PLF EMT, 2D EMT, 3D EMT
df_f['Condition order for plots']=df_f['Experimental Condition'].apply(lambda x: 'a.2D PLF EMT' if '2D PLF colony EMT' in x else 'b.2D EMT' if '2D colony EMT' in x else 'c.3D EMT')

df_summary = df_f.groupby(['Movie ID']).agg('first').reset_index()

# Filtering out the movie with additional colony or cells in the FOV and merging with feature manifest for plots

dir_io = Path('/allen/aics/emt/basement_membrane_segmentation/Resubmission/localization')

df_io = []
df_io.append(io.load_inside_outside_classification(load_from_aws = True))
for fn in dir_io.glob('*.csv'):
    df_io.append(pd.read_csv(fn, index_col=None))

df_io = pd.concat(df_io, ignore_index=True)


df_info = df_summary[[
    'Condition order for plots',
    'Movie ID',
    'Gene',
    'Migration Time (h)',
    'Migration Time InOut (h)', 
    'Timepoint (h)',
    'Bottom Z plane', 
    'Dataset',
    'Plate Barcode',
    'Scene Index',
    'Position Index',
    'Well Label'
]]

dfio_merge=pd.merge(df_io, df_info, on='Movie ID')

n_movies_io=dfio_merge['Movie ID'].nunique()
# print(f'Total number of movies for which plots are generated={n_movies_io}')

# Grouping the data and getting the fraction of 'True' values in 'Inside' column to get fraction of nuclei inside the basement membrane for each movie
dfio_grouped=dfio_merge.groupby([
    'Condition order for plots',
    'Gene',
    'Movie ID',
    'Time hr'
]).agg({
    'Inside':'mean', 
    'Migration Time (h)':'first', 
    'Migration Time InOut (h)':'first'
}).reset_index()
dfio_grouped['Fraction_outside']=1-dfio_grouped['Inside'] #fraction of nuclei outside the basement membrane

(Path(figs_dir) / 'Inside-Outside').mkdir(parents=True, exist_ok=True)
# Plotting the fraction of nuclei over time (Fig.5G)
fig,ax=plt.subplots(1,1)
sns.lineplot(dfio_grouped, x='Time hr', y='Fraction_outside', hue='Condition order for plots', palette=const.COLOR_MAP, errorbar=('pi',50), estimator=np.median)
plt.xlabel('Time (h)', fontsize=16)
plt.ylim(-0.1,1)
plt.xlim(left=10)
plt.ylabel('Fraction of nuclei outside lumen', fontsize=16)
plt.legend(loc='upper left')
plt.savefig(fr'{figs_dir}/Inside-Outside/Fraction_of_nuclei_outside_lumen.{out_type}', dpi=600, transparent=True)


dfio_grouped=dfio_merge.groupby([
    'Condition order for plots',
    'Movie ID',
]).agg({
    'Migration Time (h)':'first', 
    'Migration Time InOut (h)':'first'
})

# Plotting migration time estimated from inside and outside classification of nuclei w.r.t basement memebrane vs migration time estimated from area at the glass (Fig. 5I)
fig_scatter, ax = plt.subplots(1,1, figsize=(10,10))
fig_scatter = sns.scatterplot(dfio_grouped, x='Migration Time (h)', y='Migration Time InOut (h)', hue='Condition order for plots', palette=const.COLOR_MAP, s=100, alpha=0.7, linewidth=2, edgecolor='coral', legend=False)
plt.xlim(20,36)
plt.ylim(20,36)

plt.xlabel('Migration Time from area at glass (h)', fontsize=16)
plt.ylabel('Migration Time fraction of nuclei outside basement membrane (h)', fontsize=16)
plt.rcParams.update({'font.size':16})
plt.savefig(fr'{figs_dir}/Inside-Outside/Scatter_plot_between_computer_migration_area_on_glass_vs_inside_outside.{out_type}', dpi=600, transparent=True)

print('\n\n\n.......Statistical comparison for migration time using Area-at-Mask vs Inside-Outside:')
X = dfio_grouped['Migration Time (h)'].values
Y = dfio_grouped['Migration Time InOut (h)'].values

results = pearsonr(X, Y)
print('n: {0:d}'.format(n_movies_io))
print('Pearson Correlation: {0:.4f} | p-Value: {1}'.format(results.statistic, results.pvalue))

# Plotting example to show how migration time is estimated from fraction of nuclei outside the basement membrane over time (Fig. 5H )
# df_io_id = dfio_grouped[dfio_grouped['Movie ID']==const.EXAMPLE_IO_ID]

# fig,ax = plt.subplots(1,1,figsize=(8,6))

# x_io = df_migration_io['Migration Time IO (h)'][df_migration_io['Movie ID']==const.EXAMPLE_IO_ID].values[0]
# y_io = df_io_id['Fraction_outside'][df_io_id['Time hr']==x_io].values[0]
# ax.plot(df_io_id['Time hr'],df_io_id['Fraction_outside'], c='orange', linewidth=3)
# ax.scatter(x_io,y_io,c='black', marker='D', s=100) 
# plt.ylabel(f'Fraction of nuclei outside the lumen', fontsize=16)
# plt.xlabel('Time (hr)', fontsize=16)
# plt.xlim(left=10)
# plt.tight_layout()
# plt.savefig(fr'{figs_dir}/Example_migration_estimation_fraction_nuclei_outside_basement_membrane.{out_type}', dpi=600, transparent=True)