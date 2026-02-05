import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['svg.fonttype'] = 'none'  # Keep text editable in SVG

from EMT_data_analysis.tools import const


def _find_nearest_timepoint(df, time_column, target_time):
    """
    Find the nearest timepoint value in a dataframe column to a target time.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing the time column
    time_column : str
        Name of the column containing timepoint values
    target_time : float
        Target time value to find the nearest match for

    Returns
    -------
    float
        The nearest timepoint value from the dataframe
    """
    timepoints = df[time_column].unique()
    idx = np.abs(timepoints - target_time).argmin()
    return timepoints[idx]


def plot_examples(df_int, id_plf, id_2d, id_3d, gene, figs_dir, metric,variable='Mean Intensity', out_type='pdf'):
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

    # Publication figure dimensions
    cm_to_inch = 1 / 2.54
    fig_width_cm = 2.8846   # x-axis width
    fig_height_cm = 1.889   # y-axis height
    pad_left = 0.55
    pad_bottom = 0.45
    pad_right = 0.05
    pad_top = 0.05
    total_w = fig_width_cm * cm_to_inch + pad_left + pad_right
    total_h = fig_height_cm * cm_to_inch + pad_bottom + pad_top
    pt_to_inch = 1 / 72.0

    y_cfg = const.INTENSITY_Y_CONFIG.get(gene, None)

    # Colors
    color_orange = (255/255, 165/255, 0/255)
    color_blue   = (0/255, 191/255, 255/255)
    color_purple = (139/255, 0/255, 139/255)
    trace_lw = 0.75 * pt_to_inch * 72  # 0.75 pt

    df_plf=df_int[df_int['Data ID']==id_plf]
    df_2d=df_int[df_int['Data ID']==id_2d]
    df_3d=df_int[df_int['Data ID']==id_3d]

    fig, ax = plt.subplots(1, 1)

    # Use nearest timepoint matching to handle floating point precision differences
    x_metric_2d=df_2d[metric].values[0]
    nearest_tp_2d = _find_nearest_timepoint(df_2d, 'Timepoint (h)', x_metric_2d)
    y_metric_2d=df_2d[variable][df_2d['Timepoint (h)']==nearest_tp_2d].values[0]
    ax.plot(df_2d['Timepoint (h)'],df_2d[variable], c=color_blue, linewidth=trace_lw)
    ax.scatter(x_metric_2d,y_metric_2d,c='black', marker='D', s=8, zorder=5)

    x_metric_plf=df_plf[metric].values[0]
    nearest_tp_plf = _find_nearest_timepoint(df_plf, 'Timepoint (h)', x_metric_plf)
    y_metric_plf=df_plf[variable][df_plf['Timepoint (h)']==nearest_tp_plf].values[0]
    ax.plot(df_plf['Timepoint (h)'],df_plf[variable], c=color_purple, linewidth=trace_lw)
    ax.scatter(x_metric_plf,y_metric_plf,c='black', marker='D', s=8, zorder=5)

    x_metric_3d=df_3d[metric].values[0]
    nearest_tp_3d = _find_nearest_timepoint(df_3d, 'Timepoint (h)', x_metric_3d)
    y_metric_3d=df_3d[variable][df_3d['Timepoint (h)']==nearest_tp_3d].values[0]
    ax.plot(df_3d['Timepoint (h)'],df_3d[variable], c=color_orange, linewidth=trace_lw)
    ax.scatter(x_metric_3d,y_metric_3d,c='black', marker='D', s=8, zorder=5)

    # Y-axis
    if y_cfg is not None:
        ymin, ymax = y_cfg['ylim']
        y_pad = y_cfg['ytick_interval'] * 0.3
        ax.set_ylim(ymin - y_pad, ymax + y_pad)
        ax.set_yticks(np.arange(ymin, ymax + 1, y_cfg['ytick_interval']))
        ylabel = y_cfg.get('ylabel', 'Mean intensity (AU)')
        ax.set_ylabel(ylabel, fontsize=5, fontfamily='Arial')
    else:
        ax.set_ylabel(variable, fontsize=5, fontfamily='Arial')

    # X-axis: 0-50, interval 10, with padding
    ax.set_xlim(-2, 52)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_xlabel('Time (h)', fontsize=5, fontfamily='Arial')

    # Tick label styling
    ax.tick_params(axis='both', labelsize=5, width=0.5 * pt_to_inch * 72,
                   length=3, direction='out', pad=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(5)

    # Axis line (spine) width: 0.5 pt
    for spine in ax.spines.values():
        spine.set_linewidth(0.5 * pt_to_inch * 72)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(rf'{figs_dir}/Example_{gene}_{metric}.{out_type}', dpi=600, transparent=True)
    plt.close(fig)

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
        pd.options.display.float_format = '{:.3g}'.format
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
    plt.title(f'{column_1_name} vs {column_2_name}')
    return fig,axs

def Intensity_over_z(df, figs_dir, color_map='coolwarm', out_type='pdf'):
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
        
    
    for id, df_id in df.groupby('Data ID'):
        df_id=df_id[df_id['Normalized Z plane']>=0]
        c=df_id['Experimental Condition'].unique()[0]
        
        fig,ax=plt.subplots(1,1,figsize=(8,5))
        df_a=df_id[df_id['Area of all cells mask per Z (pixels)']>50000] # providing pixel threshold to filter out noise

        color_min=df_id[df_id['Mean intensity per Z']>0]['Mean intensity per Z'].min()
        color_max=df_id['Mean intensity per Z'].max()

        df_h=df_a.pivot_table(index="Normalized Z plane", columns="Timepoint (h)", values='Mean intensity per Z')
        n_df_row=df_h.shape[0]

        # to get uniform range of normalized Z values
        df_nan = pd.DataFrame(np.nan, index=np.arange(n_df_row,27,1), columns=np.arange(0,48.5,0.5))
        df_nanmerge=pd.concat([df_h,df_nan])
        ax.set_ylim(0,26)

        ax= sns.heatmap(df_nanmerge, cmap=color_map, vmin=color_min, vmax=color_max )
        ax.invert_yaxis()

        # Set x-axis ticks at increments of 4 hours
        x_cols = df_nanmerge.columns.tolist()
        xtick_positions = [i for i, v in enumerate(x_cols) if v % 4 == 0]
        xtick_labels = [int(x_cols[i]) for i in xtick_positions]
        ax.set_xticks([p + 0.5 for p in xtick_positions])
        ax.set_xticklabels(xtick_labels)

        # Set y-axis ticks at increments of 4 z-planes
        y_rows = df_nanmerge.index.tolist()
        ytick_positions = [i for i, v in enumerate(y_rows) if v % 4 == 0]
        ytick_labels = [int(y_rows[i]) for i in ytick_positions]
        ax.set_yticks([p + 0.5 for p in ytick_positions])
        ax.set_yticklabels(ytick_labels)

        plt.title(f'Condition={c}, Data ID={id}')
        fig.savefig(rf'{figs_dir}/Histogram_zo1_{c}_{id}.{out_type}', dpi=600)
