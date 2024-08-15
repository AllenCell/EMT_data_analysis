import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt

from EMT_data_analysis.tools import const

def plot_examples(df_int, id_plf, id_2d, id_3d, gene, figs_dir, metric,variable='Mean Intensity'):
    '''
    This function  plots one example for individual trajectories of mean intensity over time for each condition to represent how the gene metrics 
    (time at max EOMES expression, Time at inflection of E-Cad loss and Time at half maximal loss of SOX2 expression) were estimated.
    It is also used to plot migration time estimation example for area at glass over time.
    Parameters
    ----------
    df_int: DataFrame
        Dataframe with mean intensity over time information for each movie in the dataset along with the respective gene metrics.
    
    id_plf: String
        Movie ID to plot the mean intensity trajectory for a movie with 2D PLF colony EMT condition
    
    id_2d: String
        Movie ID to plot the mean intensity trajectory for a movie with 2D colony EMT condition

    id_3d: String
        Movie ID to plot the mean intensity trajectory for a movie with 3D lumenoid EMT condition

    gene : String
        Value from the column 'Gene' to represent the gene for which the example plots are being generated

    metric: String
        Name of the column of gene metric : one of these three values : 'Time of max EOMES expression (h)','Time of inflection of E-cad expression (h)','Time of half-maximal SOX2 expression (h)'

    figs_dir: directory path
        Directory to save the example plots

    Returns
    -------
    saves plots in the figs_dir'''

    df_plf=df_int[df_int['Movie ID']==id_plf]
    df_2d=df_int[df_int['Movie ID']==id_2d]
    df_3d=df_int[df_int['Movie ID']==id_3d]

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
    plt.savefig(rf'{figs_dir}/Example_{gene}_{metric}.pdf', dpi=600, transparent=True)

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

        colors = const.COLOR_MAP[conditions[i]]
        plt.boxplot(df_con[column_1_name], positions=[conditionXindex],patch_artist=True,widths=0.5, boxprops=dict(facecolor=colors, alpha=0.6, linewidth=1.75, color=colors), medianprops=dict(linestyle='-', linewidth=1.75, color=colors), whiskerprops=dict(color=colors), capprops=dict(color=colors), flierprops=dict(markeredgecolor=colors))
        plt.scatter(np.zeros(len(df_con))+conditionXindex+0.5,df_con[column_1_name],c=colors, edgecolor='white',alpha = 0.6, s=80)
        plt.scatter(np.zeros(len(df_con))+conditionXindex+1.5,df_con[column_2_name],c=colors,edgecolor='white',alpha = 0.6, s=80)
   
        plt.plot( [conditionXindex+0.5,conditionXindex+1.5], [df_con[column_1_name],df_con[column_2_name]], c='k',alpha = 0.3, linewidth=0.5)


        plt.boxplot(df_con[column_2_name], positions=[conditionXindex+2],patch_artist=True, widths=0.5, boxprops=dict(facecolor=colors, alpha=0.6, linewidth=1.75, color=colors), medianprops=dict(linestyle='-', linewidth=1.75, color=colors),whiskerprops=dict(color=colors), capprops=dict(color=colors), flierprops=dict(markeredgecolor=colors))

    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.title(f'{column_1_name}vs{column_2_name}')
    return fig,axs

def Intensity_over_z(df, figs_dir, color_map='coolwarm'):
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
        
    
    for id, df_id in df.groupby('Movie ID'):
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
        fig.savefig(rf'{figs_dir}/Histogram_zo1_{c}_{id}.pdf', dpi=600)
