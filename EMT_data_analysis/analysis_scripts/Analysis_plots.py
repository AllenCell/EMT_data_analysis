#%%[markdown]
## Script to generate plots and perform statistical analysis in the manuscript and save them in the Figure directory
####### input_link_1: (Feature manifest):/allen/aics/assay-dev/users/Nivedita/EMT/EMT_deliverable/BF_colony_mask/Manifests/Feature_manifest_V8_for_upload.csv
####### input_link2 (inside-outside nuclei classification):/allen/aics/assay-dev/users/Nivedita/EMT/EMT_deliverable/BF_colony_mask/Manifests/inside_outside_compiled_manifest_release.csv
#### Importing the required libraries

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy import stats
import scikit_posthocs as sp

# %% [markdown]
#### -------------- importing the feature manifest - ask user to input the path to feature manifest---------------


file_path = input('Enter the file path to feature manifest: ')

# e.g. C:\Users\abc\Desktop\example.csv or /home/abc/Desktop/example.csv


if os.path.exists(file_path):
    print('Loading the file')
    df=pd.read_csv(file_path)
    n_movies=df['Movie Unique ID'].nunique()
    print(f'Total number of movies in the manifest={n_movies}')

else:
    raise FileNotFoundError('No such file or directory.')


#%%
#%%[markdown]
#### ------------- Filtering the dataframe to remove movies with extra cells or extra colonies in the FOV at the time of migration--------

df_f=df[(df['Single Colony Or Lumenoid At Time of Migration']==True)&(df['Absence Of Migrating Cells Coming From Colony Out Of FOV At Time Of Migration'])]
df_f['Gene']=df_f['Gene'].apply(lambda x: 'EOMES' if 'EOMES' in x else x)
# Adding a Timepoint (h) column which converts frames into hours using  the Timelapse Interval column value
time_interval=int(''.join(filter(lambda i: i.isdigit(),df_f['Timelapse Interval'].unique()[0] )))
df_f['Timepoint (h)']=df_f['Timepoint']*(time_interval/60)
#for plotting the conditions in the order- 2D PLF EMT, 2D EMT, 3D EMT
df_f['Condition order for plots']=df_f['Experimental Condition'].apply(lambda x: 'a.2D PLF EMT' if 'PLF' in x else 'b.2D EMT' if '2D colony' in x else 'c.3D EMT')
color_map= {'a.2D PLF EMT':"darkmagenta",'b.2D EMT':"deepskyblue", 'c.3D EMT':"orange"}
n_filtered_movies=df_f['Movie Unique ID'].nunique()
print(f'No. of movies for analysis post filtering ={n_filtered_movies} ')




# %% [markdown]
#### ------------- creating a directory for figures-------------
cwd=os.getcwd()
figdir=os.path.join(cwd, r'Figures_manuscript')
if not os.path.exists(figdir):
    os.makedirs(figdir)


#%%[markdown]
#### -------------defining function to plots examples (used in the script below)------------

def plot_examples(df_int, id_plf, id_2d, id_3d, gene, metric,variable='Mean Intensity', figuredir=figdir):
    '''
    This function  plots one example for individual trajectories of mean intensity over time for each condition to represent how the gene metrics 
    (time at max EOMES expression, Time at inflection of E-Cad loss and Time at half maximal loss of SOX2 expression) were estimated.
    It is also used to plot migration time estimation example for area at glass over time.
    Parameters
    ----------
    df_int: DataFrame
        Dataframe with mean intensity over time information for each movie in the dataset along with the respective gene metrics.
    
    id_plf: String
        Movie Unique ID to plot the mean intensity trajectory for a movie with 2D PLF colony EMT condition
    
    id_2d: String
        Movie Unique ID to plot the mean intensity trajectory for a movie with 2D colony EMT condition

    id_3d: String
        Movie Unique ID to plot the mean intensity trajectory for a movie with 3D lumenoid EMT condition

    gene : String
        Value from the column 'Gene' to represent the gene for which the example plots are being generated

    metric: String
        Name of the column of gene metric : one of these three values : 'Time of max EOMES expression (h)','Time of inflection of E-cad expression (h)','Time of half-maximal SOX2 expression (h)'

    figuredir: directory path
        Directory to save the example plots

    Returns
    -------
    saves plots in the figuredir'''


    df_plf=df_int[df_int['Movie Unique ID']==id_plf]
    df_2d=df_int[df_int['Movie Unique ID']==id_2d]
    df_3d=df_int[df_int['Movie Unique ID']==id_3d]

    fig,ax=plt.subplots(1,1,figsize=(8,6))

    x_metric_2d=df_2d[metric].values[0]
    y_metric_2d=df_2d[variable][df_2d['Timepoint (h)']==x_metric_2d].values[0]
    ax.plot(df_2d['Timepoint (h)'],df_2d[variable], c='deepskyblue', linewidth=3)
    ax.scatter(x_metric_2d,y_metric_2d,c='black', marker='D', s=100)


    x_metric_plf=df_plf[metric].values[0]
    y_metric_plf=df_plf[variable][df_plf['Timepoint (h)']==x_metric_plf].values[0]
    ax.plot(df_plf['Timepoint (h)'],df_plf[variable], c='darkmagenta', linewidth=3)
    ax.scatter(x_metric_plf,y_metric_plf,c='black', marker='D', s=100)

    x_metric_3d=df_3d[metric].values[0]
    y_metric_3d=df_3d[variable][df_3d['Timepoint (h)']==x_metric_3d].values[0]
    ax.plot(df_3d['Timepoint (h)'],df_3d[variable], c='orange', linewidth=3)
    ax.scatter(x_metric_3d,y_metric_3d,c='black', marker='D', s=100)


    plt.ylabel(f'{variable}', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)

    plt.xlim(-1,50)
    plt.tight_layout()
    plt.savefig(rf'{figuredir}/Example_{gene}_{metric}.pdf', dpi=600, transparent=True)

#### function for statistical analysis
def run_statistics (x,y,z):
    '''
    This function runs statistical analysis on three distributions and runs a pariwise comparison (post-hoc) if there is a significant difference (alpha=0.05).

    Parameters
    ----------
    x: Array or list
    Values for first distribution
    
    y: Array or list
    Values for second distribution

    z: Array or list
    Values for third distribution
  

    Returns
    -------
    prints out the statistical results
    '''
    print('Number of movies:')
    print(f'n_plf:{len(x)}, n_2d:{len(y)}, n_3d:{len(z)}')

    st, p=stats.kruskal(x,y,z)
    print(f'Kruskal wallis:statistic: {st}, p-value: {p}')

    if p<=0.05:

        print('Significant difference between the distributions: applying post hoc with Holm-Bonferroni adjustment of p-value')
        variables=[x, y, z]
        print(sp.posthoc_mannwhitney(variables, p_adjust = 'holm'))

    else:
        print('No significant differnce between the distributions, no posthoc needed')
#%%[markdown]
#### ---------- Plots for Area at the glass for all three conditions and corresponding migration time estimated from the inflection of area at glass over time----------
print('Generating plots for Area at the glass for all three conditions and corresponding migration time estimated from the inflection of area at glass over time (Fig.5 C, D , E)')
##### Creating figure directory for migration timing plots (Fig.5 C, D , E)
migdir=os.path.join(figdir, r'Migration_time_plots')
if not os.path.exists(migdir):
    os.makedirs(migdir)

##### Plotting area at glass over time (Fig.5C)

df_a=df_f.groupby(['Condition order for plots','Gene','Movie Unique ID','Timepoint (h)']).agg({'Area at the glass(square micrometer)':'first', 'Migration time (h)':'first'}).reset_index()
n_a=df_a['Movie Unique ID'].nunique()
fig,ax=plt.subplots(1,1)
sns.lineplot(df_a, x='Timepoint (h)', y='Area at the glass(square micrometer)', hue='Condition order for plots', palette=color_map, errorbar=('pi', 50), estimator=np.median)
plt.ylabel('Colony area over bottom 2 Z ( $\ um^2$)', fontsize=14)
plt.xlabel('Time (hr)', fontsize=14)

plt.ylim(0,170000)
    
plt.rcParams.update({'font.size':14})
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
plt.savefig(rf'{migdir}/Area_at_the_glass_over_time_MIP_n{n_a}.pdf', transparent=True, dpi=600)



#####  Plotting migration detection example (Fig.5D)

example_2d='3500005551_8'
example_3d='3500005551_46'
example_plf='3500005551_4'

plot_examples(df_a,id_plf=example_plf, id_2d=example_2d, id_3d=example_3d, gene="Example", metric='Migration time (h)', variable='Area at the glass(square micrometer)', figuredir=migdir)



#%% [markdown]
#### ----------Box plots of overall migration time for each condition (Fig.5E) ---------

df_f=df_f.sort_values('Timepoint (h)')
df_summary=df_f.groupby('Movie Unique ID').agg('first').reset_index() # summarizing the dataframe/manifest to have one line/metric per movie

n_m=df_summary['Movie Unique ID'].nunique()

df_summary=df_summary.sort_values('Condition order for plots')
fig_mig=px.box(df_summary, x='Condition order for plots', y='Migration time (h)', color='Condition order for plots', color_discrete_map=color_map, points='all', template='simple_white',range_y=(15,40), width=800, height=600)
fig_mig.update_layout(yaxis_title='Migration time (h)',font=dict(size=18))
fig_mig.show()
fig_mig.write_image(rf'{migdir}/Migration_box_plot_n{n_m}.pdf', scale=2 )

# ####-----statitsitcal analysis of overall migriation timing between the conditions------
print('...statitsitcal analysis of overall migriation timing between the conditions...')
x_mig=df_summary['Migration time (h)'][df_summary['Experimental Condition']=='2D PLF colony EMT']
y_mig=df_summary['Migration time (h)'][df_summary['Experimental Condition']=='2D colony EMT']
z_mig=df_summary['Migration time (h)'][df_summary['Experimental Condition']=='3D lumenoid EMT']

run_statistics(x_mig,y_mig,z_mig)

####----------Box plots for migration timing for each gene in the dataset and fo each condition within that gene (Fig.S3)
print('Generating Box plots for migration timing for each gene in the dataset and fo each condition within that gene (Fig.S3)')
df_summary['gene_m'] = pd.Categorical(df_summary['Gene'], ['HIST1H2BJ','SOX2','EOMES','CDH1','TJP1'])
df_summary=df_summary.sort_values(['gene_m','Condition order for plots'])
fig_mig_g=px.box(df_summary, y='Migration time (h)', x='gene_m', color='Condition order for plots', color_discrete_map=color_map, points='all', template='simple_white',range_y=(10,40),width=1800, height=600)
fig_mig_g.update_layout(showlegend=False)
fig_mig_g.update_layout(xaxis_title='Cell lines', yaxis_title='Migration in real time (h)', font=dict(size=18))
fig_mig_g.update_layout(boxgroupgap=0.5, boxgap=0.5)
fig_mig_g.write_image(rf'{migdir}/Migration_box_plot_per_gene_all_conditions.pdf', scale=2 )

for g, df_g in df_summary.groupby('Gene'):
    print(f'..statistical analysis for the migration timing per condition for gene={g}')
    x_mig=df_g['Migration time (h)'][df_g['Experimental Condition']=='2D PLF colony EMT']
    y_mig=df_g['Migration time (h)'][df_g['Experimental Condition']=='2D colony EMT']
    z_mig=df_g['Migration time (h)'][df_g['Experimental Condition']=='3D lumenoid EMT']

    run_statistics(x_mig,y_mig,z_mig)

# %% [markdown]
#### ---------- Plotting mean intensity plots for each gene - Fig.6B, Fig. 6F and Fig 6 J----------
print('Generating plots for  mean intensity plots for each gene - Fig.6B, Fig. 6F and Fig 6 J')

intdir=os.path.join(figdir, r'Intensity_plots')
if not os.path.exists(intdir):
    os.makedirs(intdir)



#filtering to 10 z-slices over which the mean intensity is calculated
df_z=df_f[(df_f['Normalized Z plane']>=0) & (df_f['Normalized Z plane']<10)]

#Grouping by condition and gene and each movie to get mean itnensity over time for each movie
df_int=df_z.groupby(['Experimental Condition','Condition order for plots','Gene','Movie Unique ID','Timepoint (h)']).agg({'Total intensity per Z':'sum','Area of all cells mask per Z (pixels)':'sum'}).reset_index()
df_int['Mean Intensity']=df_int['Total intensity per Z']/df_int['Area of all cells mask per Z (pixels)']


#Plotting mean intensity
for g, d_g in df_int.groupby('Gene'):
    ###----------------------mean intensity---------------------###
    n=d_g['Movie Unique ID'].nunique()
    
    fig,ax=plt.subplots(1,1)
    sns.lineplot(d_g, x='Timepoint (h)', y='Mean Intensity', hue='Condition order for plots', palette=color_map, errorbar=('pi', 50), estimator=np.nanmean)
    plt.ylabel('Mean intensity (a.u.)', fontsize=14)
    plt.xlabel('Time (h)', fontsize=14)
    plt.title(f'Gene: {g}', fontsize=14)
    
    plt.rcParams.update({'font.size':14})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
    plt.savefig(fr'{intdir}/Mean_intensity_plot_{g}_n{n}_mean_line.pdf', dpi=600, transparent=True) 

#plotting normalized intensity by applying min max normalization
df_norm=pd.DataFrame()
for id, df_id in df_int.groupby('Movie Unique ID'):
    df_id=df_id.sort_values('Timepoint (h)')
    int_min=df_id['Mean Intensity'].min()
    int_max=df_id['Mean Intensity'].max()
    df_id['Normalized intensity']=(df_id['Mean Intensity']-int_min)/(int_max-int_min)
    df_norm=pd.concat([df_norm,df_id])

for g, d_g in df_norm.groupby('Gene'):
    
    n=d_g['Movie Unique ID'].nunique()
    
    fig,ax=plt.subplots(1,1)
    sns.lineplot(d_g, x='Timepoint (h)', y='Normalized intensity', hue='Condition order for plots', palette=color_map, errorbar=('pi', 50), estimator=np.nanmean)
    plt.ylabel('Normalized mean intensity (a.u.)', fontsize=14)
    plt.xlabel('Time (h)', fontsize=14)
    plt.title(f'Gene: {g}', fontsize=14)
    
    plt.rcParams.update({'font.size':14})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
    plt.savefig(fr'{intdir}/Normalized_Mean_intensity_plot_{g}_n{n}_mean_line.pdf', dpi=600, transparent=True) 


# %%[markdown]
#### ------------ Plotting and saving the examples for Time of max EOMES expression (h), Time of inflection of E-cad expression (h) and Time of half-maximal SOX2 expression (h), for each condition respectively---------
#### Fig.6C., Fig.6 G, Fig.6 K.
print('generating plots for Fig.6C., Fig.6 G, Fig.6 K  ')

##### creating directory to save the example plots
exdir=os.path.join(figdir, r'Example_metric_plots')
if not os.path.exists(exdir):
    os.makedirs(exdir)
#filtering to 10 z-slices over which the mean intensity is calculated
df_z=df_f[(df_f['Normalized Z plane']>=0) & (df_f['Normalized Z plane']<10)]

#Grouping by condition and gene and each movie to get mean itnensity over time for each movie
df_int=df_z.groupby(['Experimental Condition','Condition order for plots','Gene','Movie Unique ID','Timepoint (h)']).agg({'Total intensity per Z':'sum','Area of all cells mask per Z (pixels)':'sum',
                                                                                                                            'Time of max EOMES expression (h)':'first','Time of inflection of E-cad expression (h)':'first',
                                                                                                                            'Time of half-maximal SOX2 expression (h)':'first'}).reset_index()
df_int['Mean Intensity']=df_int['Total intensity per Z']/df_int['Area of all cells mask per Z (pixels)']


## Time of max EOMES expression (h) examples-

Eomes_id_plf='3500005551_6'
Eomes_id_2d='3500005551_11'
Eomes_id_3d='3500005551_19'

plot_examples(df_int,id_plf=Eomes_id_plf, id_2d=Eomes_id_2d, id_3d=Eomes_id_3d, gene="EOMES", metric='Time of max EOMES expression (h)', figuredir=exdir)

## Time of inflection of E-cad expression (h) examples-
CDH_id_plf='3500006073_35'
CDH_id_2d='3500006212_19'
CDH_id_3d='3500006102_7'

plot_examples(df_int,id_plf=CDH_id_plf, id_2d=CDH_id_2d, id_3d=CDH_id_3d, gene="CDH1", metric='Time of inflection of E-cad expression (h)', figuredir=exdir)


##plotting SOX metric examples

SOX_id_plf='3500005548_49'
SOX_id_2d='3500005827_34'
SOX_id_3d='3500005548_67'

plot_examples(df_int,id_plf=SOX_id_plf, id_2d=SOX_id_2d, id_3d=SOX_id_3d, gene="SOX2", metric='Time of half-maximal SOX2 expression (h)', figuredir=exdir)

# %% [markdown]
#### -------- Connected box and line plots to show pair-wise relationship between gene metric (Time of max EOMES expression (h), Time of inflection of E-cad expression (h) and Time of half-maximal SOX2 expression (h)) and respective migration time ---------
###### Fig.6 D,H, L

print('Generating plots for connected scatter and box plots- Fig.6 D,H, L')
###### Defining the function to plot the connected pair-wise box plots
def plot_connected_box_plot(df_summary, Gene, column_1_name, column_2_name, ylim=[10,40], ylabel='Time (hr)'):
    '''
    This function  plots pairwise connected scatter and box plots to show the dynamics and relationship between gene metrics (Time of max EOMES expression (h),
      Time of inflection of E-cad expression (h) and Time of half-maximal SOX2 expression (h)) and respective migration time.

    Parameters
    ----------
    df_summary: DataFrame
        Dataframe with gene metric and migration time for each movie
    
    Gene: String
        Value of the column 'Gene' identifying the cell line for which the mtric and plots are generated
    
    column_1_name: String
         Name of the gene metric (one of these three values : 'Time of max EOMES expression (h)','Time of inflection of E-cad expression (h)','Time of half-maximal SOX2 expression (h)')
         for which the first box plot and scatter points are generated

    column_2_name: String
        'Migration time (h)' column for the second box plot and scatter to be connected to the first box plot and scatter points.

    ylim: List
        List giving the y-axis limit

    metric: String
        Name of the column of gene metric : one of these three values : 'Time of max EOMES expression (h)','Time of inflection of E-cad expression (h)','Time of half-maximal SOX2 expression (h)'

    ylabel: String
        Label for y axis

    Returns
    -------
    Figure and axes'''

    df_gene=df_summary[df_summary.Gene==Gene]
    fig, axs = plt.subplots(figsize=(10, 5))
    conditions=['a.2D PLF EMT','b.2D EMT','c.3D EMT']
    for i in range(len(conditions)):
        df_con = df_gene[df_gene["Condition order for plots"]==conditions[i]]
        conditionXindex = i*4

        colors = color_map[conditions[i]]
        plt.boxplot(df_con[column_1_name], positions=[conditionXindex],patch_artist=True,widths=0.5, boxprops=dict(facecolor=colors, alpha=0.6, linewidth=1.75, color=colors), medianprops=dict(linestyle='-', linewidth=1.75, color=colors), whiskerprops=dict(color=colors), capprops=dict(color=colors), flierprops=dict(markeredgecolor=colors))
        plt.scatter(np.zeros(len(df_con))+conditionXindex+0.5,df_con[column_1_name],c=colors, edgecolor='white',alpha = 0.6, s=80)
        plt.scatter(np.zeros(len(df_con))+conditionXindex+1.5,df_con[column_2_name],c=colors,edgecolor='white',alpha = 0.6, s=80)
   
        plt.plot( [conditionXindex+0.5,conditionXindex+1.5], [df_con[column_1_name],df_con[column_2_name]], c='k',alpha = 0.3, linewidth=0.5)


        plt.boxplot(df_con[column_2_name], positions=[conditionXindex+2],patch_artist=True, widths=0.5, boxprops=dict(facecolor=colors, alpha=0.6, linewidth=1.75, color=colors), medianprops=dict(linestyle='-', linewidth=1.75, color=colors),whiskerprops=dict(color=colors), capprops=dict(color=colors), flierprops=dict(markeredgecolor=colors))

    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.title(f'{column_1_name}vs{column_2_name}')
    return fig,axs

##### Creating directory for connected plots
connecteddir=os.path.join(figdir, r'Connected_box_plots')
if not os.path.exists(connecteddir):
    os.makedirs(connecteddir)

##### Plotting pair-wise connected plots for Time of max EOMES expression (h) and Migration time for EOMES (Fig.6D)
fig1,ax1=plot_connected_box_plot(df_summary, 'EOMES', 'Time of max EOMES expression (h)','Migration time (h)')
fig1.savefig(rf'{connecteddir}/Connected_box_plot_Eomes_time_at_max_EOMES_expression_vs_Migration_hr.pdf', dpi=600)

##### Plotting pair-wise connected plots for Time of inflection of E-cad expression (h) and Migration time for E-Cad (Fig.6H)
fig2,ax2=plot_connected_box_plot(df_summary, 'CDH1','Time of inflection of E-cad expression (h)','Migration time (h)')
fig2.savefig(rf'{connecteddir}/Connected_box_plot_Ecad_time_of_inflection_E_Cad_expression_Migration_hr.pdf', dpi=600)

##### Plotting pair-wise connected plots for Time of half-maximal SOX2 expression (h) and Migration time for SOX2 (Fig.6L)
fig3,ax3=plot_connected_box_plot(df_summary, 'SOX2','Time of half-maximal SOX2 expression (h)','Migration time (h)')
fig3.savefig(rf'{connecteddir}/Connected_box_plot_Sox_time_of_half_maximal_vs_Migration_hr.pdf', dpi=600)

# %% [markdown]
#####------- Plotting box plots for Supplementary figure S5 a. timing of expression change relative to time of induction of EMT (time 0), b. Difference between gene metrics and migration time,
##### c. timing of expression change normalized between the time of induction and migration (0-time of EMT induction and 1- time of migration)
print('Generating plots for supplementary Fig.5. Timing of expression change')
##### Creating directory for timing of expression
expressiondir=os.path.join(figdir, r'Timing_of_expression')
if not os.path.exists(expressiondir):
    os.makedirs(expressiondir)

##compiling data for the plots
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

## only plotting for 2D colony EMT and 3D colony EMT conditions
conditions_colony=['b.2D EMT','c.3D EMT']
df_comb_f=df_comb[df_comb['Condition order for plots'].isin(conditions_colony)]

##Fig.S5 a.timing of expression change relative to time of induction of EMT (time 0)
fig_metric=px.box(df_comb_f, y='gene_metric', x='Gene', color='Condition order for plots', color_discrete_map=color_map, points='all', template='simple_white',range_y=(10,40),width=1200, height=600)
fig_metric.update_layout(showlegend=False)

fig_metric.update_layout(xaxis_title='Cell lines', yaxis_title='Time of expression change (h)', font=dict(size=18))
fig_metric.update_layout(boxgroupgap=0.5, boxgap=0.25)
fig_metric.write_image(rf'{expressiondir}/Timing_of_expression_change_relative_to_EMT_induction_T_0_FigS5a.pdf', scale=2 )

##Fig.S5 b.Difference between gene metrics and migration time- time of expression change relative to migration time
fig_difference=px.box(df_comb_f, y='Difference', x='Gene', color='Condition order for plots', color_discrete_map=color_map, points='all', template='simple_white',range_y=(-30,10),width=1200, height=600)
fig_difference.update_layout(showlegend=False)

fig_difference.update_layout(xaxis_title='Cell lines', yaxis_title='Time of expression change relative to migiration time (h)', font=dict(size=18))
fig_difference.update_layout(boxgroupgap=0.5, boxgap=0.25)
fig_difference.write_image(rf'{expressiondir}/Timing_of_expression_change_relative_to_migration_time_FigS5b.pdf', scale=2 )

##Fig.S5 c.timing of expression change normalized between the time of induction and migration (0-time of EMT induction and 1- time of migration)
fig_difference=px.box(df_comb_f, y='Normalized_metric', x='Gene', color='Condition order for plots', color_discrete_map=color_map, points='all', template='simple_white',range_y=(0,1.5),width=1200, height=600)
fig_difference.update_layout(showlegend=False)

fig_difference.update_layout(xaxis_title='Cell lines', yaxis_title='Time of expression change divided by migration time (h)', font=dict(size=18))
fig_difference.update_layout(boxgroupgap=0.5, boxgap=0.25)
fig_difference.write_image(rf'{expressiondir}/Timing_of_expression_change_divided_by_migration_time_FigS5c.pdf', scale=2 )

#%%
print('.......Statistical comparison for gene metric:')
for g, df_g in df_comb.groupby('Gene'):
    print(f'gene={g}')
    x=df_g['gene_metric'][df_g['Experimental Condition']=='2D PLF colony EMT']
    y=df_g['gene_metric'][df_g['Experimental Condition']=='2D colony EMT']
    z=df_g['gene_metric'][df_g['Experimental Condition']=='3D lumenoid EMT']

    run_statistics(x,y,z)




# %% [markdown]
####-----Plotting heatmaps to show ZO1 (TJP1) intensity changes in Z plane over time (Fig.7 and Fig. S6)
print('Generating Heatmaps for ZO1 - Fig.7 and Fig. S6 ')
##### Creating directory to save ZO1 histograms
ZO1dir=os.path.join(figdir, r'ZO1_heatmaps')
if not os.path.exists(ZO1dir):
    os.makedirs(ZO1dir)

##### Defining the function to create hieatmaps 

def Intensity_over_z(df, color_map='coolwarm', directory=None):
    '''
    This function  plots a heatmap for a movie to show ZO1 intensity over time across the z plane to show ZO1 dynamics 

    Parameters
    ----------
    df: DataFrame
        Dataframe with ZO1 intensity information extracted for each z plane over time
    
    color_map: String
        Color map tp be used in the heatmap
    
    
    directory : directory path
        Directory to save the heatmap.

    Returns
    -------
    saves heatmap to the directory'''
        
    
    for id, df_id in df.groupby('Movie Unique ID'):
        df_id=df_id[df_id['Normalized Z plane']>=0]
        c=df_id['Experimental Condition'].unique()[0]
        
        fig,ax=plt.subplots(1,1,figsize=(8,5))
        df_a=df_id[df_id['Area of all cells mask per Z (pixels)']>50000] # providing pixel threshold to filter out noise

        color_min=df_id['Mean intensity per Z'].min()
        color_max=df_id['Mean intensity per Z'].max()

        df_h=df_a.pivot_table(index="Normalized Z plane", columns="Timepoint (h)", values='Mean intensity per Z')
        n_df_row=df_h.shape[0]

        # to get uniform range of normalized Z values
        df_nan = pd.DataFrame(np.nan, index=np.arange(n_df_row,27,1), columns=np.arange(0,48.5,0.5))
        df_nanmerge=pd.concat([df_h,df_nan])
        ax.set_ylim(0,26)

        ax= sns.heatmap(df_nanmerge, cmap=color_map, vmin=color_min, vmax=color_max )
        ax.invert_yaxis()
        plt.title(f'Condition={c}, Movie ID={id}')
        fig.savefig(rf'{directory}/histogram_zo1_{c}_{id}.pdf', dpi=600)


## Filtering the dataset to only ZO1 data
df_zo=df_f[df_f.Gene=='TJP1']

#example ZO1 Movie IDs
Example_zo1_ids=['3500005829_12','3500005698_13','3500005834_69','3500005834_68','3500005834_65','3500005834_50','3500005829_7','3500005829_41',
                 '3500005698_5','3500005829_44','3500005829_46','3500006071_7','3500006071_3','3500005829_48','3500005829_60','3500005834_55']

df_zo_examples=df_zo[df_zo['Movie Unique ID'].isin(Example_zo1_ids)]

#generating and saving the heatmaps
Intensity_over_z(df_zo_examples, directory=ZO1dir)



# %%[markdown]
####----Plotting inside-outside classification results- Fraction of nuclei outside the basement membrane- Fig. 5 G, H I ------
print('Generating plots for inside-outside classification and migration time (Fig.5 G, H ,I)')

inside_outside_file_path = input('Enter the file path to inside_outside_manifest: ')

# e.g. C:\Users\abc\Desktop\example.csv or /home/abc/Desktop/example.csv


if os.path.exists(inside_outside_file_path):
    print('Loading the file')
    df_io=pd.read_csv(inside_outside_file_path)
   
else:
    raise FileNotFoundError('No such file or directory.')


#%%
### filtering out the movie with additional colony or cells in the FOV and merging with feature manifest for plots

df_info=df_summary[['Condition order for plots','Movie Unique ID','Gene','Migration time (h)','Bottom Z plane']]
## merging with feature manifest
dfio_merge=pd.merge(df_io, df_info, on='Movie Unique ID')

n_movies_io=dfio_merge['Movie Unique ID'].nunique()
print(f'Total number of movies for which plots are generated={n_movies_io}')


##grouping the data and getting the fraction of 'True' values in 'Inside' column to get fraction of nuclei inside the basement membrane for each movie
dfio_grouped=dfio_merge.groupby(['Condition order for plots','Gene','Movie Unique ID','Time hr']).agg({'Inside':'mean', 'Migration time (h)':'first'}).reset_index()
dfio_grouped['Fraction_outside']=1-dfio_grouped['Inside'] #fraction of nuclei outside the basement membrane


####-----Creating directory to save the inside outside classification plots (Fig.5 G, H I)

iodir=os.path.join(figdir, r'Inside_outside_classification_plots')
if not os.path.exists(iodir):
    os.makedirs(iodir)

####-----Plotting the fraction of nuclei over time (Fig.5G)

fig,ax=plt.subplots(1,1)
sns.lineplot(dfio_grouped, x='Time hr', y='Fraction_outside', hue='Condition order for plots', palette=color_map, errorbar=('pi',50), estimator=np.median)
plt.xlabel('Time (h)', fontsize=16)
plt.ylim(-0.1,1)
plt.xlim(left=10)
plt.ylabel('Fraction of nuclei outside lumen', fontsize=16)
plt.legend(loc='upper left')
plt.savefig(fr'{iodir}/Fraction_of_nuclei_outside_lumen.pdf', dpi=600, transparent=True)

#%%
#estimating migration time from inside outside classification
movie_ids, io_migration_time,migration_time, condition, gene=[],[],[],[],[]
for id, df_id in dfio_grouped.groupby('Movie Unique ID'):
   
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

df_migration_io=pd.DataFrame(zip(movie_ids, io_migration_time,migration_time, condition, gene), columns=['Movie Unique ID','Migration time IO (h)', 'Migration time (h)','Condition order for plots','Gene'])



#####-----Plotting migration time estimated from inside and outside classification of nuclei w.r.t basement memebrane vs migration time estimated from area at the glass (Fig. 5I)
fig_scatter,ax=plt.subplots(1,1, figsize=(10,10))
fig_scatter=sns.scatterplot(df_migration_io, x='Migration time (h)', y='Migration time IO (h)', hue='Condition order for plots', palette=color_map, s=100, alpha=0.7, linewidth=2, edgecolor='coral', legend=False)
plt.xlim(20,36)
plt.ylim(20,36)

plt.xlabel('Migration time from area at glass (h)', fontsize=16)
plt.ylabel('Migration time fraction of nuclei outside basement membrane (h)', fontsize=16)
plt.rcParams.update({'font.size':16})
plt.savefig(fr'{iodir}/Scatter_plot_between_computer_migration_area_on_glass_vs_inside_outside.pdf', dpi=600, transparent=True)


#####------Plotting example to show how migration time is estimated from fraction of nuclei outside the basement membrane over time (Fig. 5H )

example_io_id='3500005834_21'
df_io_id=dfio_grouped[dfio_grouped['Movie Unique ID']==example_io_id]

fig,ax=plt.subplots(1,1,figsize=(8,6))

x_io=df_migration_io['Migration time IO (h)'][df_migration_io['Movie Unique ID']==example_io_id].values[0]
y_io=df_io_id['Fraction_outside'][df_io_id['Time hr']==x_io].values[0]
ax.plot(df_io_id['Time hr'],df_io_id['Fraction_outside'], c='orange', linewidth=3)
ax.scatter(x_io,y_io,c='black', marker='D', s=100) 
plt.ylabel(f'Fraction of nuclei outside the lumen', fontsize=16)
plt.xlabel('Time (hr)', fontsize=16)
plt.xlim(left=10)
plt.tight_layout()
plt.savefig(fr'{iodir}/Example_migration_estimation_fraction_nuclei_outside_basement_membrane.pdf', dpi=600, transparent=True)