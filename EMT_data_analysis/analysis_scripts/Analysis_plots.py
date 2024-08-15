import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from EMT_data_analysis.tools import io, const
from EMT_data_analysis.analysis_scripts import plot_tools

figs_dir = io.setup_base_directory_name("figures")

df = io.load_image_analysis_extracted_features(load_from_aws=True)

df_f=df[(df['Single Colony Or Lumenoid At Time of Migration']==True)&(df['Absence Of Migrating Cells Coming From Colony Out Of FOV At Time Of Migration'])]
df_f['Gene']=df_f['Gene'].apply(lambda x: 'EOMES' if 'EOMES' in x else x)
# Adding a Timepoint (h) column which converts frames into hours using  the Timelapse Interval column value
time_interval=int(''.join(filter(lambda i: i.isdigit(),df_f['Timelapse Interval'].unique()[0] )))
df_f['Timepoint (h)']=df_f['Timepoint']*(time_interval/60)
#for plotting the conditions in the order- 2D PLF EMT, 2D EMT, 3D EMT
df_f['Condition order for plots']=df_f['Experimental Condition'].apply(lambda x: 'a.2D PLF EMT' if 'PLF' in x else 'b.2D EMT' if '2D colony' in x else 'c.3D EMT')

n_filtered_movies=df_f['Movie ID'].nunique()
print(f'No. of movies for analysis post filtering ={n_filtered_movies} ')

print('Generating plots for Area at the glass for all three conditions and corresponding migration time estimated from the inflection of area at glass over time (Fig.5 C, D , E)')

df_a=df_f.groupby(['Condition order for plots','Gene','Movie ID','Timepoint (h)']).agg({'Area at the glass(square micrometer)':'first', 'Migration time (h)':'first'}).reset_index()
n_a=df_a['Movie ID'].nunique()
fig,ax=plt.subplots(1,1)
sns.lineplot(df_a, x='Timepoint (h)', y='Area at the glass(square micrometer)', hue='Condition order for plots', palette=const.COLOR_MAP, errorbar=('pi', 50), estimator=np.median)
plt.ylabel('Colony area over bottom 2 Z ( $\ um^2$)', fontsize=14)
plt.xlabel('Time (hr)', fontsize=14)
plt.ylim(0,170000)
plt.rcParams.update({'font.size':14})
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
plt.savefig(rf'{figs_dir}/Area_at_the_glass_over_time_MIP_n{n_a}.pdf', transparent=True, dpi=600)

example_2d='3500005551_8'
example_3d='3500005551_46'
example_plf='3500005551_4'

plot_tools.plot_examples(df_a,id_plf=example_plf, id_2d=example_2d, id_3d=example_3d, gene="Example", metric='Migration time (h)', variable='Area at the glass(square micrometer)', figs_dir=figs_dir)

df_f=df_f.sort_values('Timepoint (h)')
df_summary=df_f.groupby('Movie ID').agg('first').reset_index() # summarizing the dataframe/manifest to have one line/metric per movie

n_m=df_summary['Movie ID'].nunique()

df_summary=df_summary.sort_values('Condition order for plots')
fig_mig=px.box(df_summary, x='Condition order for plots', y='Migration time (h)', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(15,40), width=800, height=600)
fig_mig.update_layout(yaxis_title='Migration time (h)',font=dict(size=18))
fig_mig.write_image(rf'{figs_dir}/Migration_box_plot_n{n_m}.pdf', scale=2 )

print('...statitsitcal analysis of overall migriation timing between the conditions...')
x_mig=df_summary['Migration time (h)'][df_summary['Experimental Condition']=='2D PLF colony EMT']
y_mig=df_summary['Migration time (h)'][df_summary['Experimental Condition']=='2D colony EMT']
z_mig=df_summary['Migration time (h)'][df_summary['Experimental Condition']=='3D lumenoid EMT']

plot_tools.run_statistics(x_mig,y_mig,z_mig)

print('Generating Box plots for migration timing for each gene in the dataset and fo each condition within that gene (Fig.S3)')
df_summary['gene_m'] = pd.Categorical(df_summary['Gene'], ['HIST1H2BJ','SOX2','EOMES','CDH1','TJP1'])
df_summary=df_summary.sort_values(['gene_m','Condition order for plots'])
fig_mig_g=px.box(df_summary, y='Migration time (h)', x='gene_m', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(10,40),width=1800, height=600)
fig_mig_g.update_layout(showlegend=False)
fig_mig_g.update_layout(xaxis_title='Cell lines', yaxis_title='Migration in real time (h)', font=dict(size=18))
fig_mig_g.update_layout(boxgroupgap=0.5, boxgap=0.5)
fig_mig_g.write_image(rf'{figs_dir}/Migration_box_plot_per_gene_all_conditions.pdf', scale=2 )

for g, df_g in df_summary.groupby('Gene'):
    print(f'..statistical analysis for the migration timing per condition for gene={g}')
    x_mig=df_g['Migration time (h)'][df_g['Experimental Condition']=='2D PLF colony EMT']
    y_mig=df_g['Migration time (h)'][df_g['Experimental Condition']=='2D colony EMT']
    z_mig=df_g['Migration time (h)'][df_g['Experimental Condition']=='3D lumenoid EMT']

    plot_tools.run_statistics(x_mig,y_mig,z_mig)

print('Generating plots for  mean intensity plots for each gene - Fig.6B, Fig. 6F and Fig 6 J')

# filtering to 10 z-slices over which the mean intensity is calculated
df_z=df_f[(df_f['Normalized Z plane']>=0) & (df_f['Normalized Z plane']<10)]

#Grouping by condition and gene and each movie to get mean itnensity over time for each movie
df_int=df_z.groupby(['Experimental Condition','Condition order for plots','Gene','Movie ID','Timepoint (h)']).agg({'Total intensity per Z':'sum','Area of all cells mask per Z (pixels)':'sum'}).reset_index()
df_int['Mean Intensity']=df_int['Total intensity per Z']/df_int['Area of all cells mask per Z (pixels)']

#Plotting mean intensity
for g, d_g in df_int.groupby('Gene'):
    n=d_g['Movie ID'].nunique()
    
    fig,ax=plt.subplots(1,1)
    sns.lineplot(d_g, x='Timepoint (h)', y='Mean Intensity', hue='Condition order for plots', palette=const.COLOR_MAP, errorbar=('pi', 50), estimator=np.nanmean)
    plt.ylabel('Mean intensity (a.u.)', fontsize=14)
    plt.xlabel('Time (h)', fontsize=14)
    plt.title(f'Gene: {g}', fontsize=14)
    
    plt.rcParams.update({'font.size':14})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
    plt.savefig(fr'{figs_dir}/Mean_intensity_plot_{g}_n{n}_mean_line.pdf', dpi=600, transparent=True) 

df_norm=pd.DataFrame()
for id, df_id in df_int.groupby('Movie ID'):
    df_id=df_id.sort_values('Timepoint (h)')
    int_min=df_id['Mean Intensity'].min()
    int_max=df_id['Mean Intensity'].max()
    df_id['Normalized intensity']=(df_id['Mean Intensity']-int_min)/(int_max-int_min)
    df_norm=pd.concat([df_norm,df_id])

for g, d_g in df_norm.groupby('Gene'):    
    n=d_g['Movie ID'].nunique()
    
    fig,ax=plt.subplots(1,1)
    sns.lineplot(d_g, x='Timepoint (h)', y='Normalized intensity', hue='Condition order for plots', palette=const.COLOR_MAP, errorbar=('pi', 50), estimator=np.nanmean)
    plt.ylabel('Normalized mean intensity (a.u.)', fontsize=14)
    plt.xlabel('Time (h)', fontsize=14)
    plt.title(f'Gene: {g}', fontsize=14)
    
    plt.rcParams.update({'font.size':14})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
    plt.savefig(fr'{figs_dir}/Normalized_Mean_intensity_plot_{g}_n{n}_mean_line.pdf', dpi=600, transparent=True) 


print('Generating plots for Fig.6C., Fig.6 G, Fig.6 K  ')

# Filtering to 10 z-slices over which the mean intensity is calculated
df_z=df_f[(df_f['Normalized Z plane']>=0) & (df_f['Normalized Z plane']<10)]

# Grouping by condition and gene and each movie to get mean itnensity over time for each movie
df_int=df_z.groupby([
    'Experimental Condition',
    'Condition order for plots',
    'Gene',
    'Movie ID',
    'Timepoint (h)'
]).agg(
    {
        'Total intensity per Z': 'sum',
        'Area of all cells mask per Z (pixels)': 'sum',
        'Time of max EOMES expression (h)': 'first',
        'Time of inflection of E-cad expression (h)': 'first',
        'Time of half-maximal SOX2 expression (h)': 'first'
    }
).reset_index()
df_int['Mean Intensity']=df_int['Total intensity per Z']/df_int['Area of all cells mask per Z (pixels)']

# Time of max EOMES expression (h) examples-
Eomes_id_plf='3500005551_6'
Eomes_id_2d='3500005551_11'
Eomes_id_3d='3500005551_19'

plot_tools.plot_examples(df_int,id_plf=Eomes_id_plf, id_2d=Eomes_id_2d, id_3d=Eomes_id_3d, gene="EOMES", figs_dir=figs_dir, metric='Time of max EOMES expression (h)')

## Time of inflection of E-cad expression (h) examples-
CDH_id_plf='3500006073_35'
CDH_id_2d='3500006212_19'
CDH_id_3d='3500006102_7'

plot_tools.plot_examples(df_int,id_plf=CDH_id_plf, id_2d=CDH_id_2d, id_3d=CDH_id_3d, gene="CDH1", figs_dir=figs_dir, metric='Time of inflection of E-cad expression (h)')

##plotting SOX metric examples
SOX_id_plf='3500005548_49'
SOX_id_2d='3500005827_34'
SOX_id_3d='3500005548_67'

plot_tools.plot_examples(df_int,id_plf=SOX_id_plf, id_2d=SOX_id_2d, id_3d=SOX_id_3d, gene="SOX2", figs_dir=figs_dir, metric='Time of half-maximal SOX2 expression (h)')

print('Generating plots for connected scatter and box plots- Fig.6 D,H, L')

##### Plotting pair-wise connected plots for Time of max EOMES expression (h) and Migration time for EOMES (Fig.6D)
fig1,ax1=plot_tools.plot_connected_box_plot(df_summary, 'EOMES', 'Time of max EOMES expression (h)','Migration time (h)')
fig1.savefig(rf'{figs_dir}/Connected_box_plot_Eomes_time_at_max_EOMES_expression_vs_Migration_hr.pdf', dpi=600)

##### Plotting pair-wise connected plots for Time of inflection of E-cad expression (h) and Migration time for E-Cad (Fig.6H)
fig2,ax2=plot_tools.plot_connected_box_plot(df_summary, 'CDH1','Time of inflection of E-cad expression (h)','Migration time (h)')
fig2.savefig(rf'{figs_dir}/Connected_box_plot_Ecad_time_of_inflection_E_Cad_expression_Migration_hr.pdf', dpi=600)

##### Plotting pair-wise connected plots for Time of half-maximal SOX2 expression (h) and Migration time for SOX2 (Fig.6L)
fig3,ax3=plot_tools.plot_connected_box_plot(df_summary, 'SOX2','Time of half-maximal SOX2 expression (h)','Migration time (h)')
fig3.savefig(rf'{figs_dir}/Connected_box_plot_Sox_time_of_half_maximal_vs_Migration_hr.pdf', dpi=600)

'''
Plotting box plots for Supplementary figure S5 a. timing of expression change relative
to time of induction of EMT (time 0), b. Difference between gene metrics and migration
time, c. timing of expression change normalized between the time of induction and
migration (0-time of EMT induction and 1- time of migration)
''' 
print('Generating plots for supplementary Fig.5. Timing of expression change')

## Compiling data for the plots
df_eomes=df_summary[df_summary.Gene=='EOMES']
df_eomes['Difference']=df_eomes['Time of max EOMES expression (h)']-df_eomes['Migration time (h)']
df_eomes['Normalized_metric']=df_eomes['Time of max EOMES expression (h)']/df_eomes['Migration time (h)']
df_eomes.rename(columns={'Time of max EOMES expression (h)':'gene_metric'}, inplace=True)

df_cdh=df_summary[df_summary.Gene=='CDH1']
df_cdh['Difference']=df_cdh['Time of inflection of E-cad expression (h)']-df_cdh['Migration time (h)']
df_cdh['Normalized_metric']=df_cdh['Time of inflection of E-cad expression (h)']/df_cdh['Migration time (h)']
df_cdh.rename(columns={'Time of inflection of E-cad expression (h)':'gene_metric'}, inplace=True)

df_sox=df_summary[df_summary.Gene=='SOX2']
df_sox['Difference']=df_sox['Time of half-maximal SOX2 expression (h)']-df_sox['Migration time (h)']
df_sox['Normalized_metric']=df_sox['Time of half-maximal SOX2 expression (h)']/df_sox['Migration time (h)']
df_sox.rename(columns={'Time of half-maximal SOX2 expression (h)':'gene_metric'},inplace=True)

df_comb=pd.concat([df_cdh,df_eomes,df_sox])
df_comb=df_comb.sort_values(['Gene','Condition order for plots'], ascending=[False,True])

# Only plotting for 2D colony EMT and 3D colony EMT conditions
conditions_colony=['b.2D EMT','c.3D EMT']
df_comb_f=df_comb[df_comb['Condition order for plots'].isin(conditions_colony)]

# Fig.S5 a.timing of expression change relative to time of induction of EMT (time 0)
fig_metric=px.box(df_comb_f, y='gene_metric', x='Gene', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(10,40),width=1200, height=600)
fig_metric.update_layout(showlegend=False)

fig_metric.update_layout(xaxis_title='Cell lines', yaxis_title='Time of expression change (h)', font=dict(size=18))
fig_metric.update_layout(boxgroupgap=0.5, boxgap=0.25)
fig_metric.write_image(rf'{figs_dir}/Timing_of_expression_change_relative_to_EMT_induction_T_0_FigS5a.pdf', scale=2 )

# Fig.S5 b.Difference between gene metrics and migration time- time of expression change relative to migration time
fig_difference=px.box(df_comb_f, y='Difference', x='Gene', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(-30,10),width=1200, height=600)
fig_difference.update_layout(showlegend=False)

fig_difference.update_layout(xaxis_title='Cell lines', yaxis_title='Time of expression change relative to migiration time (h)', font=dict(size=18))
fig_difference.update_layout(boxgroupgap=0.5, boxgap=0.25)
fig_difference.write_image(rf'{figs_dir}/Timing_of_expression_change_relative_to_migration_time_FigS5b.pdf', scale=2 )

# Fig.S5 c.timing of expression change normalized between the time of induction and migration (0-time of EMT induction and 1- time of migration)
fig_difference=px.box(df_comb_f, y='Normalized_metric', x='Gene', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(0,1.5),width=1200, height=600)
fig_difference.update_layout(showlegend=False)

fig_difference.update_layout(xaxis_title='Cell lines', yaxis_title='Time of expression change divided by migration time (h)', font=dict(size=18))
fig_difference.update_layout(boxgroupgap=0.5, boxgap=0.25)
fig_difference.write_image(rf'{figs_dir}/Timing_of_expression_change_divided_by_migration_time_FigS5c.pdf', scale=2 )


print('.......Statistical comparison for gene metric:')
for g, df_g in df_comb.groupby('Gene'):
    print(f'gene={g}')
    x=df_g['gene_metric'][df_g['Experimental Condition']=='2D PLF colony EMT']
    y=df_g['gene_metric'][df_g['Experimental Condition']=='2D colony EMT']
    z=df_g['gene_metric'][df_g['Experimental Condition']=='3D lumenoid EMT']

    plot_tools.run_statistics(x,y,z)

print('Generating Heatmaps for ZO1 - Fig.7 and Fig. S6 ')
# Filtering the dataset to only ZO1 data
df_zo=df_f[df_f.Gene=='TJP1']

#example ZO1 Movie IDs
Example_zo1_ids = [
    '3500005829_12',
    '3500005698_13',
    '3500005834_69',
    '3500005834_68',
    '3500005834_65',
    '3500005834_50',
    '3500005829_7',
    '3500005829_41',
    '3500005698_5',
    '3500005829_44',
    '3500005829_46',
    '3500006071_7',
    '3500006071_3',
    '3500005829_48',
    '3500005829_60',
    '3500005834_55']

df_zo_examples=df_zo[df_zo['Movie ID'].isin(Example_zo1_ids)]

#generating and saving the heatmaps
plot_tools.Intensity_over_z(df_zo_examples, figs_dir=figs_dir)

print('Generating plots for inside-outside classification and migration time (Fig.5 G, H ,I)')

df_io = io.load_inside_outside_classification(load_from_aws = True)

# Filtering out the movie with additional colony or cells in the FOV and merging with feature manifest for plots

df_info=df_summary[['Condition order for plots','Movie ID','Gene','Migration time (h)','Bottom Z plane']]

dfio_merge=pd.merge(df_io, df_info, on='Movie ID')

n_movies_io=dfio_merge['Movie ID'].nunique()
print(f'Total number of movies for which plots are generated={n_movies_io}')

# Grouping the data and getting the fraction of 'True' values in 'Inside' column to get fraction of nuclei inside the basement membrane for each movie
dfio_grouped=dfio_merge.groupby(['Condition order for plots','Gene','Movie ID','Time hr']).agg({'Inside':'mean', 'Migration time (h)':'first'}).reset_index()
dfio_grouped['Fraction_outside']=1-dfio_grouped['Inside'] #fraction of nuclei outside the basement membrane


# Plotting the fraction of nuclei over time (Fig.5G)
fig,ax=plt.subplots(1,1)
sns.lineplot(dfio_grouped, x='Time hr', y='Fraction_outside', hue='Condition order for plots', palette=const.COLOR_MAP, errorbar=('pi',50), estimator=np.median)
plt.xlabel('Time (h)', fontsize=16)
plt.ylim(-0.1,1)
plt.xlim(left=10)
plt.ylabel('Fraction of nuclei outside lumen', fontsize=16)
plt.legend(loc='upper left')
plt.savefig(fr'{figs_dir}/Fraction_of_nuclei_outside_lumen.pdf', dpi=600, transparent=True)

# Estimating migration time from inside outside classification
movie_ids, io_migration_time,migration_time, condition, gene=[],[],[],[],[]
for id, df_id in dfio_grouped.groupby('Movie ID'):
   
    d_filt=df_id[(df_id['Time hr']>=12)&(df_id['Time hr']<=40)]
    raw=d_filt['Fraction_outside']
    raw_values=raw.values
    d_filt['dy2']=savgol_filter(raw_values,polyorder=2, window_length=30, deriv=2)
    index_infl=d_filt['dy2'].idxmax()

    migration_time.append(df_id['Migration time (h)'].unique()[0])
    movie_ids.append(id)
    io_migration_time.append(d_filt['Time hr'][index_infl])
    condition.append(df_id['Condition order for plots'].unique()[0])
    gene.append(df_id.Gene.unique()[0])

df_migration_io=pd.DataFrame(zip(movie_ids, io_migration_time,migration_time, condition, gene), columns=['Movie ID','Migration time IO (h)', 'Migration time (h)','Condition order for plots','Gene'])

# Plotting migration time estimated from inside and outside classification of nuclei w.r.t basement memebrane vs migration time estimated from area at the glass (Fig. 5I)
fig_scatter,ax=plt.subplots(1,1, figsize=(10,10))
fig_scatter=sns.scatterplot(df_migration_io, x='Migration time (h)', y='Migration time IO (h)', hue='Condition order for plots', palette=const.COLOR_MAP, s=100, alpha=0.7, linewidth=2, edgecolor='coral', legend=False)
plt.xlim(20,36)
plt.ylim(20,36)

plt.xlabel('Migration time from area at glass (h)', fontsize=16)
plt.ylabel('Migration time fraction of nuclei outside basement membrane (h)', fontsize=16)
plt.rcParams.update({'font.size':16})
plt.savefig(fr'{figs_dir}/Scatter_plot_between_computer_migration_area_on_glass_vs_inside_outside.pdf', dpi=600, transparent=True)

# Plotting example to show how migration time is estimated from fraction of nuclei outside the basement membrane over time (Fig. 5H )

example_io_id='3500005834_21'
df_io_id=dfio_grouped[dfio_grouped['Movie ID']==example_io_id]

fig,ax=plt.subplots(1,1,figsize=(8,6))

x_io=df_migration_io['Migration time IO (h)'][df_migration_io['Movie ID']==example_io_id].values[0]
y_io=df_io_id['Fraction_outside'][df_io_id['Time hr']==x_io].values[0]
ax.plot(df_io_id['Time hr'],df_io_id['Fraction_outside'], c='orange', linewidth=3)
ax.scatter(x_io,y_io,c='black', marker='D', s=100) 
plt.ylabel(f'Fraction of nuclei outside the lumen', fontsize=16)
plt.xlabel('Time (hr)', fontsize=16)
plt.xlim(left=10)
plt.tight_layout()
plt.savefig(fr'{figs_dir}/Example_migration_estimation_fraction_nuclei_outside_basement_membrane.pdf', dpi=600, transparent=True)