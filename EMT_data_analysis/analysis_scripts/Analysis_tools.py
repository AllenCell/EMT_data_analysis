import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from EMT_data_analysis.tools import io, const
from EMT_data_analysis.analysis_scripts import plot_tools
from pathlib import Path
import scikit_posthocs as sp
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

# Set font to be Arial and configure text in figures to be editable in Adobe Illustrator
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as editable text in SVG
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"

warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)


def bootstrap_corr(x, y, method="Pearson", confidence_interval=0.95,
                   n_bootstraps=2000, seed=42, verbose=True):
    """
    Compute correlation with bootstrap confidence intervals.

    Parameters
    ----------
    x, y : array-like
        Data arrays to correlate
    method : str
        "Pearson" or "Spearman"
    confidence_interval : float
        Confidence level (default 0.95)
    n_bootstraps : int
        Number of bootstrap resamples
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print results if True

    Returns
    -------
    tuple : (r_observed, p_value, ci_low, ci_high)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    corr_func = pearsonr if method == "Pearson" else spearmanr

    r_observed, p_value = corr_func(x, y)
    n_samples = len(x)

    boot_rng = np.random.default_rng(seed)
    inds = np.arange(n_samples)
    boot_corrs = []
    for _ in range(n_bootstraps):
        resampled_inds = boot_rng.choice(inds, size=n_samples, replace=True)
        rx = x[resampled_inds]
        ry = y[resampled_inds]
        boot_r, _ = corr_func(rx, ry)
        boot_corrs.append(boot_r)

    alpha = (1.0 - confidence_interval) / 2.0
    ci_low = np.nanpercentile(boot_corrs, 100 * alpha)
    ci_high = np.nanpercentile(boot_corrs, 100 * (1.0 - alpha))

    if verbose:
        print(f'{method} Correlation: {r_observed:.3g} | p-value: {p_value:.3g} | 95% CI: [{ci_low:.3g}, {ci_high:.3g}]')

    return r_observed, p_value, ci_low, ci_high


def run_all_analyses():
    """
    Run all analysis functions
    """
    OUT_TYPE = 'svg'
    FIGS_DIR = io.setup_base_directory_name("figures")

    df = io.load_image_analysis_extracted_features()
    
    plot_area_at_glass_all_data(df, FIGS_DIR, OUT_TYPE)
    plot_area_at_glass_h2b(df, FIGS_DIR, OUT_TYPE)
    plot_migration_timing_all_data(df, FIGS_DIR, OUT_TYPE)
    plot_migration_timing_h2b(df, FIGS_DIR, OUT_TYPE)
    plot_migration_timing_by_gene(df, FIGS_DIR, OUT_TYPE)
    plot_mean_intensity_by_gene(df, FIGS_DIR, OUT_TYPE)
    plot_gene_expression_experiments(df, FIGS_DIR, OUT_TYPE)
    plot_collagenase_analysis(df, FIGS_DIR, OUT_TYPE)
    analyze_crispr_knockdown_experiments(df, FIGS_DIR, OUT_TYPE)
    plot_inside_outside_migration_timing(df, FIGS_DIR, OUT_TYPE)
    plot_mmp_inhibitor_migration(df, FIGS_DIR, OUT_TYPE)
    plot_mmp_inhibitor_migration_aics0036(df, FIGS_DIR, OUT_TYPE)
    plot_mmp_inhibitor_migration_aics0000(df, FIGS_DIR, OUT_TYPE)
    plot_bmp_inhibitor_migration(df, FIGS_DIR, OUT_TYPE)
    plot_zo1_heatmaps(df, FIGS_DIR, OUT_TYPE)
    plot_immunolabeling_heatmap(df, FIGS_DIR, OUT_TYPE)
    run_bland_altman_analysis(df, FIGS_DIR)
    immunlabeling_mean_intensity_analysis(df, FIGS_DIR, OUT_TYPE)
    

def load_io_data(df):
    """
    Helper function for importing the inside-outside nucleus localization data and appending it to the
    main manifest, filtering for only movies for which the analysis was conducted.
    """
    df_f = df[(df['Gene']=='HIST1H2BJ') & (df['Experimental Condition']=='3D lumenoid EMT')]
    df_f = df_f[
        (df_f['Single Colony Or Lumenoid At Time of Migration']==True)& \
        (df_f['Absence Of Migrating Cells Coming From Colony Out Of FOV At Time Of Migration']==True)& \
        (df_f['Perturbation']=='No perturbation')& \
        (df_f['Absence Of Excessive Cell Death']==True)& \
        (df_f['Image Size Z']==30)& \
        (df_f['Fixation Status']=='Live Cells')
    ]
    df_f['Migration Onset Time (Inside/Outside Basement Membrane Based)'].replace('',np.nan, inplace=True)
    df_f = df_f.dropna(subset=['Migration Onset Time (Inside/Outside Basement Membrane Based)'])

    # Adding a Timepoint (h) column which converts frames into hours using  the Timelapse Interval column value
    time_interval=30 #int(''.join(filter(lambda i: i.isdigit(),df_f['Timelapse Interval'].unique()[0] )))
    df_f['Timepoint (h)']=df_f['Timepoint']*(time_interval/60)

    # For plotting the conditions in the order- 2D PLF EMT, 2D EMT, 3D EMT
    df_f['Condition order for plots']=df_f['Experimental Condition'].apply(lambda x: 'a.2D PLF EMT' if '2D PLF colony EMT' in x else 'b.2D EMT' if '2D colony EMT' in x else 'c.3D EMT')

    df_summary = df_f.groupby(['Data ID']).agg('first').reset_index()

    df_info = df_summary[[
        'Condition order for plots',
        'Data ID',
        'Gene',
        'Migration Onset Time (Footprint Area Based)',
        'Migration Onset Time (Inside/Outside Basement Membrane Based)', 
        'Timepoint (h)',
        'Bottom Z plane', 
        'Plate Barcode',
        'Scene Index',
        'Position Index',
        'Well Label'
    ]]

    df_io = io.load_inside_outside_classification()
    #df_io = pd.read_csv(io.get_results_directory_name() / "nuclei_localization_mesh_inv_final_final_01282026" / f"{data_id}_localized_nuclei.csv")
    #df_io = df_io[df_io['Z']<27]
    #df_io = pd.read_csv()

    dfio_merged=pd.merge(df_io, df_info, on='Data ID', suffixes=['','_remove'])
    remove = [col for col in dfio_merged.columns if 'remove' in col]
    dfio_merged.drop(columns=remove, inplace=True)

    return dfio_merged


def create_df_f(df, time_interval=30):
    """
    Helper function to filter main data manifest to only include movies which were used for the main EMT migration analysis.
    """
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
    df_f['Timepoint (h)']=df_f['Timepoint']*(time_interval/60)

    # For plotting the conditions in the order- 2D PLF EMT, 2D EMT, 3D EMT
    df_f['Condition order for plots']=df_f['Experimental Condition'].apply(
        lambda x: 'a.2D PLF EMT' if '2D PLF colony EMT' in x else \
            'b.2D EMT' if '2D colony EMT' in x else 'c.3D EMT')
    
    return df_f


def create_df_IF(df):
    """
    Helper function to prune dataset to necessary columns and reorganize so that each label is in its own row.
    And get quantitative versions of the immunolabeling data for heatmap generation.
    """

    df_f = df[(df['Immunostaining Set']=='First Set Of Immunostaining')|(df['Immunostaining Set']=='Second Set Of Immunostaining')|(df['Immunostaining Set']=='Third Set Of Immunostaining')]
    df_f = df_f[(df_f['Normalized Z plane']>=0)&(df_f['Normalized Z plane']<10)]

    df_f['Content Of Channel 2'].fillna('No Antibody Control', inplace=True)
    df_f['Content Of Channel 3'].fillna('No Antibody Control', inplace=True)

    df_summary = []
    for data_id, df_id in df_f.groupby('Data ID'):
        volume = df_id['Area of all cells mask per Z (pixels)'].sum()
        if volume == 0:
            continue
        for ch in [2,3]:
            int_total = df_id[f'Total intensity per Z (Channel {ch})'].sum()
            if int_total == 0:
                continue

            row = {
                'Data ID': data_id,
                'Label': df_id.iloc[0][f'Content Of Channel {ch}'],
                'Condition': df_id.iloc[0]['Experimental Condition'],
                'Time (h)': float(df_id.iloc[0]['Timepoint'])*0.5,
                'Round': df_id.iloc[0]['Immunostaining Set'],
                'Volume': volume,
                'Mean Intensity': int(int_total/volume)
            }
            df_summary.append(pd.DataFrame(row, index=[0]))
    df_summary = pd.concat(df_summary, ignore_index=True)
    return df_summary



def plot_area_at_glass_all_data(df, figs_dir, out_type):
    """
    Generates plots for area at the glass for all three conditions and corresponding migration time estimated from the inflection of area at glass over time
    
    Parameters:
    -------
    df : pd.DataFrame
        Dataframe containing the area at the glass for each movie, gene and timepoint
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """

    # Set up dataset
    df_f = create_df_f(df)
    df_a = df_f.groupby(['Condition order for plots','Gene','Data ID','Timepoint (h)']).agg({'Area at the glass(square micrometer)':'first', 'Migration Onset Time (Footprint Area Based)':'first'}).reset_index()
    n_a = df_a['Data ID'].nunique()
    fig,ax = plt.subplots(1,1)

    sns.lineplot(df_a, x='Timepoint (h)', y='Area at the glass(square micrometer)', hue='Condition order for plots', palette=const.COLOR_MAP, errorbar=('pi', 50), estimator=np.median)
    plt.ylabel('Colony area over bottom 2 Z ( $\ um^2$)', fontsize=14)
    plt.xlabel('Time (hr)', fontsize=14)
    plt.ylim(-6000, 176000)
    plt.yticks(np.arange(0, 170001, 20000))
    plt.xlim(-2, 52)
    plt.xticks(np.arange(0, 51, 10))
    plt.rcParams.update({'font.size':14})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(rf'{figs_dir}/Area_at_the_glass_over_time_MIP_n{n_a}.{out_type}', transparent=True, dpi=600)

    (figs_dir / 'Individual_Examples').mkdir(exist_ok=True, parents=True)
    plot_tools.plot_examples(
        df_int = df_f,
        id_plf = const.EXAMPLE_PLF,
        id_2d = const.EXAMPLE_2D,
        id_3d = const.EXAMPLE_3D,
        gene = "HIST1H2BJ",
        metric = 'Migration Onset Time (Footprint Area Based)',
        variable = 'Area at the glass(square micrometer)',
        figs_dir = figs_dir / 'Individual_Examples',
        out_type=out_type)


def plot_area_at_glass_h2b(df, figs_dir, out_type):
    """
    Generates plots for area at the glass for HIST1H2BJ gene and corresponding migration time estimated from the inflection of area at glass over time
    
    Parameters:
    -------
    df : pd.DataFrame
        Dataframe containing the area at the glass for each movie and timepoint for HIST1H2BJ data
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """

    # Set up dataset
    df_f = create_df_f(df)
    df_a_h2b = df_f[df_f['Gene']=='HIST1H2BJ'].groupby(['Condition order for plots','Gene','Data ID','Timepoint (h)']).agg({'Area at the glass(square micrometer)':'first', 'Migration Onset Time (Footprint Area Based)':'first'}).reset_index()
    df_a = df_f.groupby(['Condition order for plots','Gene','Data ID','Timepoint (h)']).agg({'Area at the glass(square micrometer)':'first', 'Migration Onset Time (Footprint Area Based)':'first'}).reset_index()
    n_a = df_a['Data ID'].nunique()
    fig, ax = plt.subplots(1,1)

    sns.lineplot(df_a_h2b, x='Timepoint (h)', y='Area at the glass(square micrometer)', hue='Condition order for plots', palette=const.COLOR_MAP, errorbar=('pi', 50), estimator=np.median)
    plt.ylabel('Colony area over bottom 2 Z ( $\ um^2$)', fontsize=14)
    plt.xlabel('Time (hr)', fontsize=14)
    plt.ylim(-6000, 176000)
    plt.yticks(np.arange(0, 170001, 20000))
    plt.xlim(-2, 52)
    plt.xticks(np.arange(0, 51, 10))
    plt.rcParams.update({'font.size':14})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # Used_for information -
    # these figures are labeled as - Fig. 3B
    plt.savefig(rf'{figs_dir}/Area_at_the_glass_over_time_MIP_HIST1H2BJ_n{n_a}.{out_type}', transparent=True, dpi=600)

    plot_tools.plot_examples(
        df_int = df_a,
        id_plf = const.EXAMPLE_PLF,
        id_2d = const.EXAMPLE_2D,
        id_3d = const.EXAMPLE_3D,
        gene = "HIST1H2BJ",
        metric = 'Migration Onset Time (Footprint Area Based)',
        variable = 'Area at the glass(square micrometer)',
        figs_dir = figs_dir,
        out_type=out_type)


def plot_migration_timing_all_data(df, figs_dir, out_type):
    """
    Generates box plots for migration timing for all three conditions and corresponding migration time estimated from the inflection of area at glass over time
    
    Parameters:
    -------
    df : pd.DataFrame
        Dataframe containing the migration time for each movie, gene and condition
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """ 

    # Set up dataset
    df_f = create_df_f(df)
    df_f = df_f.sort_values('Timepoint (h)')
    # Summarizing the dataframe/manifest to have one line/metric per movie
    df_summary = df_f.groupby('Data ID').agg('first').reset_index()

    n_m = df_summary['Data ID'].nunique()
    df_summary = df_summary.sort_values(['Gene','Condition order for plots'])
    fig_mig = px.box(df_summary, x='Condition order for plots', y='Migration Onset Time (Footprint Area Based)', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(15,35), width=800, height=600)
    fig_mig.update_layout(yaxis_title='Migration Onset Time (Footprint Area Based)',font=dict(size=18))
    fig_mig.write_image(rf'{figs_dir}/Migration_box_plot_n{n_m}.{out_type}', scale=2 )


def plot_migration_timing_h2b(df, figs_dir, out_type):
    """
    Generate box plots for migration timing for HIST1H2BJ gene and corresponding migration time estimated from the inflection of area at glass over time
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing the migration time for each movie, gene and condition
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """
    # Set up dataset
    df_f = create_df_f(df)
    df_f = df_f.sort_values('Timepoint (h)')
    # Summarizing the dataframe/manifest to have one line/metric per movie
    df_summary = df_f.groupby('Data ID').agg('first').reset_index()
    n_m = df_summary[df_summary['Gene']=='HIST1H2BJ']['Data ID'].nunique()
    df_summary = df_summary.sort_values(['Gene','Condition order for plots'])
    df_summary = df_summary.sort_values(by='Condition order for plots')
    
    fig_mig = px.box(df_summary[df_summary['Gene']=='HIST1H2BJ'], x='Condition order for plots', y='Migration Onset Time (Footprint Area Based)', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(15,35), width=800, height=600)
    fig_mig.update_layout(yaxis_title='Migration Onset Time (Footprint Area Based)',font=dict(size=18))
    # Used_for information - 
    # these figures are labeled as - Fig. 3D
    fig_mig.write_image(rf'{figs_dir}/Migration_box_plot_HIST1H2BJ_n{n_m}.{out_type}', scale=2 )

    print('\n\n\n...statitsitcal analysis of overall migriation timing between the conditions for HIST1H2BJ...')
    x_mig = df_summary['Migration Onset Time (Footprint Area Based)'][['2D PLF' in val for val in df_summary['Experimental Condition'].values]].dropna()
    y_mig = df_summary['Migration Onset Time (Footprint Area Based)'][['2D colony EMT' in val for val in df_summary['Experimental Condition'].values]].dropna()
    z_mig = df_summary['Migration Onset Time (Footprint Area Based)'][['3D lumenoid EMT' in val for val in df_summary['Experimental Condition'].values]].dropna()

    print('2D PLF: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(x_mig), np.median(x_mig), np.std(x_mig), np.min(x_mig), np.max(x_mig)))
    print('2D EMT: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(y_mig), np.median(y_mig), np.std(y_mig), np.min(y_mig), np.max(y_mig)))
    print('3D EMT: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(z_mig), np.median(z_mig), np.std(z_mig), np.min(z_mig), np.max(z_mig)))

    plot_tools.run_statistics(x_mig,y_mig,z_mig)

def plot_migration_timing_by_gene(df, figs_dir, out_type):
    """
    Generates box plots for migration timing for each gene in the dataset and for each condition within that gene

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing the migration time for each movie, gene and condition
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """

    # print('Generating Box plots for migration timing for each gene in the dataset and fo each condition within that gene (Fig.S3)')

    # Set up dataset
    df_f = create_df_f(df)
    df_f = df_f.sort_values('Timepoint (h)')
    # Summarizing the dataframe/manifest to have one line/metric per movie
    df_summary = df_f.groupby('Data ID').agg('first').reset_index()
    n_m = df_summary[df_summary['Gene']=='HIST1H2BJ']['Data ID'].nunique()
    df_summary = df_summary.sort_values(['Gene','Condition order for plots'])
    df_summary = df_summary.sort_values(by='Condition order for plots')
    
    df_summary['gene_m'] = pd.Categorical(df_summary['Gene'], df_summary['Gene'].unique())
    df_summary = df_summary.sort_values(['gene_m','Condition order for plots'])
    df_summary = df_summary[df_summary['Gene']!='AAVS1']
    
    fig_mig_g = px.box(df_summary, y='Migration Onset Time (Footprint Area Based)', x='gene_m', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(10,35),width=1800, height=600)
    fig_mig_g.update_layout(showlegend=False)
    fig_mig_g.update_layout(xaxis_title='Cell lines', yaxis_title='Migration in real time (h)', font=dict(size=18))
    fig_mig_g.update_layout(boxgroupgap=0.5, boxgap=0.5)

    # Used_for information - 
    # these figures are labeled as - Extended Data Fig. 5
    # all data ids should be labeled as - Extended Data Fig. 5
    fig_mig_g.write_image(rf'{figs_dir}/Migration_box_plot_per_gene_all_conditions.{out_type}', scale=2 )

    df_summary = df_summary[df_summary['Gene']!='CLYBL']
    for g, df_g in df_summary.groupby('Gene'):
        print(f'\n\n\n...statistical analysis for the migration timing per condition for gene={g}')
        x_mig = df_g['Migration Onset Time (Footprint Area Based)'][['2D PLF' in val for val in df_g['Experimental Condition'].values]].dropna()
        y_mig = df_g['Migration Onset Time (Footprint Area Based)'][['2D colony EMT' in val for val in df_g['Experimental Condition'].values]].dropna()
        z_mig = df_g['Migration Onset Time (Footprint Area Based)'][['3D lumenoid EMT' in val for val in df_g['Experimental Condition'].values]].dropna()

        print('2D PLF: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(x_mig), np.median(x_mig), np.std(x_mig), np.min(x_mig), np.max(x_mig)))
        print('2D EMT: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(y_mig), np.median(y_mig), np.std(y_mig), np.min(y_mig), np.max(y_mig)))
        print('3D EMT: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(z_mig), np.median(z_mig), np.std(z_mig), np.min(z_mig), np.max(z_mig)))
            
        plot_tools.run_statistics(x_mig, y_mig, z_mig)


def plot_mean_intensity_by_gene(df, figs_dir, out_type):
    """
    Generates mean intensity plots for each gene in the dataset over time

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing the mean intensity for each movie, condition, gene and timepoint
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """

    # Set up dataset
    df_f = create_df_f(df)
    df_f = df_f.sort_values('Timepoint (h)')
    # Filtering to 10 z-slices over which the mean intensity is calculated
    df_z = df_f[(df_f['Normalized Z plane']>=0) & (df_f['Normalized Z plane']<10)]

    # Grouping by condition and gene and each movie to get mean itnensity over time for each movie
    df_int = df_z.groupby(['Experimental Condition','Condition order for plots','Gene','Data ID','Timepoint (h)']).agg(
        {
            'Total intensity per Z':'sum',
            'Area of all cells mask per Z (pixels)':'sum',
            'Time of max EOMES expression (h)':'first',
            'Time of max TBXT expression (h)':'first',
            'Time of inflection of E-cad expression (h)':'first',
            'Time of half-maximal SOX2 expression (h)':'first'
        }
    ).reset_index()
    df_int['Mean Intensity']=df_int['Total intensity per Z']/df_int['Area of all cells mask per Z (pixels)']
    df_int['Mean Intensity'] = df_int['Mean Intensity'].replace(0,np.nan)
    df_int = df_int[df_int['Gene']!='HIST1H2BJ']

    # Publication figure dimensions and styling
    cm_to_inch = 1 / 2.54
    fig_width_cm = 2.8846   # x-axis width
    fig_height_cm = 1.889   # y-axis height
    # Pad for axis labels/ticks (approximate)
    pad_left = 0.55
    pad_bottom = 0.45
    pad_right = 0.05
    pad_top = 0.05
    total_w = fig_width_cm * cm_to_inch + pad_left + pad_right
    total_h = fig_height_cm * cm_to_inch + pad_bottom + pad_top

    # Colors: orange, blue, purple
    intensity_palette = {
        'a.2D PLF EMT': (139/255, 0/255, 139/255),   # purple
        'b.2D EMT':     (0/255, 191/255, 255/255),    # blue
        'c.3D EMT':     (255/255, 165/255, 0/255),    # orange
    }

    pt_to_inch = 1 / 72.0  # for linewidth conversions

    # Plotting mean intensity
    for g, d_g in df_int.groupby('Gene'):
        n = d_g['Data ID'].nunique()
        y_cfg = const.INTENSITY_Y_CONFIG.get(g, {'ylim': (100, 200), 'ytick_interval': 10})

        fig, ax = plt.subplots(1, 1, figsize=(total_w, total_h))
        fig.subplots_adjust(
            left=pad_left / total_w,
            bottom=pad_bottom / total_h,
            right=1 - pad_right / total_w,
            top=1 - pad_top / total_h,
        )

        sns.lineplot(
            d_g, x='Timepoint (h)', y='Mean Intensity',
            hue='Condition order for plots', palette=intensity_palette,
            errorbar=('pi', 50), estimator=np.nanmean,
            linewidth=0.75 * pt_to_inch * 72,  # 0.75 pt
            ax=ax,
        )

        # Style the error band: fill only, no stroke
        for child in ax.get_children():
            if isinstance(child, mpl.collections.PolyCollection):
                child.set_edgecolor('none')
                child.set_linewidth(0)

        # Y-axis
        ymin, ymax = y_cfg['ylim']
        y_pad = y_cfg['ytick_interval'] * 0.3
        ax.set_ylim(ymin - y_pad, ymax + y_pad)
        ax.set_yticks(np.arange(ymin, ymax + 1, y_cfg['ytick_interval']))
        ax.set_ylabel('Mean intensity (AU)', fontsize=5, fontfamily='Arial')

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

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend off (conditions identified by color)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        # Used_for information -
        # these figures are labeled as - Fig. 5
        # for Gene = SOX2, labels are Fig. 5B
        # for Gene = TBXT, labels are Fig. 5G
        # for Gene = EOMES, labels are Fig. 5L
        # for Gene = CDH1, labels are Fig. 5Q
        plt.savefig(fr'{figs_dir}/Mean_intensity_plot_{g}_n{n}_mean_line.{out_type}', dpi=600, transparent=True)
        plt.close(fig)

    Path(rf'{figs_dir}/Individual_Examples').mkdir(exist_ok=True, parents=True)
    # Time of max EOMES expression (h) examples
    plot_tools.plot_examples(
        df_int = df_int,
        id_plf = const.EOMES_PLF,
        id_2d = const.EOMES_2D,
        id_3d = const.EOMES_3D,
        gene = "EOMES",
        figs_dir = figs_dir / 'Individual_Examples',
        metric='Time of max EOMES expression (h)',
        out_type=out_type)

    # Time of max TBXT expression (h) examples
    plot_tools.plot_examples(
        df_int = df_int,
        id_plf = const.TBXT_PLF,
        id_2d = const.TBXT_2D,
        id_3d = const.TBXT_3D,
        gene = "TBXT",
        figs_dir = figs_dir / 'Individual_Examples',
        metric='Time of max TBXT expression (h)',
        out_type=out_type)

    # Time of inflection of E-cad expression (h) examples-
    plot_tools.plot_examples(
        df_int = df_int,
        id_plf = const.CDH_PLF,
        id_2d = const.CDH_2D,
        id_3d = const.CDH_3D,
        gene = "CDH1",
        figs_dir = figs_dir / 'Individual_Examples',
        metric='Time of inflection of E-cad expression (h)',
        out_type=out_type)

    # Time of inflection of SOX expression (h) examples-
    plot_tools.plot_examples(
        df_int = df_int,
        id_plf = const.SOX_PLF,
        id_2d = const.SOX_2D,
        id_3d = const.SOX_3D,
        gene = "SOX2",
        figs_dir = figs_dir / 'Individual_Examples',
        metric = 'Time of half-maximal SOX2 expression (h)',
        out_type=out_type)


def plot_gene_expression_experiments(df, figs_dir, out_type):
    """
    Plotting supplementary box plots
    a. timing of expression change relative to time of induction of EMT (time 0)
    b. Difference between gene metrics and migration time
    c. Timing of expression change normalized between the time of induction and
    migration (0-time of EMT induction and 1- time of migration)

    Parameters:
    df : pd.DataFrame
        Dataframe containing the summary of the data with migration time and other metrics
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """

    # Set up dataset
    df_f = create_df_f(df)
    df_f = df_f.sort_values('Timepoint (h)')
    df_summary = df_f.groupby('Data ID').agg('first').reset_index()
    df_summary = df_summary.sort_values(['Gene','Condition order for plots'])
    df_summary = df_summary.sort_values(by='Condition order for plots')
    df_summary['gene_m'] = pd.Categorical(df_summary['Gene'], df_summary['Gene'].unique())
    df_summary = df_summary.sort_values(['gene_m','Condition order for plots'])

    # Compiling data for the plots
    df_eomes = df_summary[df_summary.Gene=='EOMES']
    df_eomes['Difference']=df_eomes['Time of max EOMES expression (h)']-df_eomes['Migration Onset Time (Footprint Area Based)']
    df_eomes['Normalized_metric']=df_eomes['Time of max EOMES expression (h)']/df_eomes['Migration Onset Time (Footprint Area Based)']
    df_eomes.rename(columns={'Time of max EOMES expression (h)':'gene_metric'}, inplace=True)

    df_tbxt = df_summary[df_summary.Gene=='TBXT']
    df_tbxt['Difference']=df_tbxt['Time of max TBXT expression (h)']-df_tbxt['Migration Onset Time (Footprint Area Based)']
    df_tbxt['Normalized_metric']=df_tbxt['Time of max TBXT expression (h)']/df_tbxt['Migration Onset Time (Footprint Area Based)']
    df_tbxt.rename(columns={'Time of max TBXT expression (h)':'gene_metric'}, inplace=True)

    df_cdh = df_summary[df_summary.Gene=='CDH1']
    df_cdh['Difference']=df_cdh['Time of inflection of E-cad expression (h)']-df_cdh['Migration Onset Time (Footprint Area Based)']
    df_cdh['Normalized_metric']=df_cdh['Time of inflection of E-cad expression (h)']/df_cdh['Migration Onset Time (Footprint Area Based)']
    df_cdh.rename(columns={'Time of inflection of E-cad expression (h)':'gene_metric'}, inplace=True)

    df_sox = df_summary[df_summary.Gene=='SOX2']
    df_sox['Difference']=df_sox['Time of half-maximal SOX2 expression (h)']-df_sox['Migration Onset Time (Footprint Area Based)']
    df_sox['Normalized_metric']=df_sox['Time of half-maximal SOX2 expression (h)']/df_sox['Migration Onset Time (Footprint Area Based)']
    df_sox.rename(columns={'Time of half-maximal SOX2 expression (h)':'gene_metric'},inplace=True)

    df_comb=pd.concat([df_cdh,df_eomes,df_sox,df_tbxt])
    df_comb=df_comb.sort_values(['Gene','Condition order for plots'], ascending=[False,True])

    # Only plotting for 2D colony EMT and 3D colony EMT conditions
    conditions_colony=['a.2D PLF EMT','b.2D EMT','c.3D EMT']
    df_comb_f = df_comb[df_comb['Condition order for plots'].isin(conditions_colony)]
    df_comb_f['sort_value'] = df_summary['Gene'].apply(lambda gene: 0 if gene=='SOX2' else 1 if gene=='TBXT' else 2 if gene=='EOMES' else 3)
    df_comb_f.sort_values(by=['sort_value', 'Condition order for plots'], axis=0, inplace=True)

    # Timing of expression change relative to time of induction of EMT (time 0)
    fig_metric = px.box(df_comb_f, y='gene_metric', x='Gene', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(10,50),width=1200, height=600)
    fig_metric.update_layout(showlegend=False)

    fig_metric.update_layout(xaxis_title='Cell lines', yaxis_title='Time of expression change (h)', font=dict(size=18))
    fig_metric.update_layout(boxgroupgap=0.5, boxgap=0.25)
    fig_metric.write_image(rf'{figs_dir}/Timing_of_expression_change_relative_to_EMT_induction_T_0_FigS5a.{out_type}', scale=2 )

    # Difference between gene metrics and migration time- time of expression change relative to migration time
    fig_difference = px.box(df_comb_f, y='Difference', x='Gene', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(-30,20),width=1200, height=600)
    fig_difference.update_layout(showlegend=False)

    fig_difference.update_layout(xaxis_title='Cell lines', yaxis_title='Time of expression change relative to migiration time (h)', font=dict(size=18))
    fig_difference.update_layout(boxgroupgap=0.5, boxgap=0.25)
    fig_difference.write_image(rf'{figs_dir}/Timing_of_expression_change_relative_to_migration_time_FigS5b.{out_type}', scale=2 )

    # Timing of expression change normalized between the time of induction and migration (0-time of EMT induction and 1- time of migration)
    fig_difference = px.box(df_comb_f, y='Normalized_metric', x='Gene', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(0,2.2),width=1200, height=600)
    fig_difference.update_layout(showlegend=False)

    fig_difference.update_layout(xaxis_title='Cell lines', yaxis_title='Time of expression change divided by migration time (h)', font=dict(size=18))
    fig_difference.update_layout(boxgroupgap=0.5, boxgap=0.25)
    fig_difference.write_image(rf'{figs_dir}/Timing_of_expression_change_divided_by_migration_time_FigS5c.{out_type}', scale=2 )

    # Publication figure dimensions for per-gene box and scatter plots
    # Match y-axis height to mean intensity plots, x-axis width = 2.2516 cm
    _cm = 1 / 2.54
    _pt = 1 / 72.0
    _box_fig_w_cm = 2.2516
    _box_fig_h_cm = 1.889
    _pad_l, _pad_b, _pad_r, _pad_t = 0.55, 0.45, 0.05, 0.05
    _tot_w = _box_fig_w_cm * _cm + _pad_l + _pad_r
    _tot_h = _box_fig_h_cm * _cm + _pad_b + _pad_t

    # Colors
    _box_palette = {
        'a.2D PLF EMT': (139/255, 0/255, 139/255),
        'b.2D EMT':     (0/255, 191/255, 255/255),
        'c.3D EMT':     (255/255, 165/255, 0/255),
    }

    # Plotting of expression timing relative to EMT induction as independent graphs per-gene
    for g, df_g in df_comb.groupby('Gene'):
        fig, ax = plt.subplots(1, 1, figsize=(_tot_w, _tot_h))
        fig.subplots_adjust(
            left=_pad_l / _tot_w, bottom=_pad_b / _tot_h,
            right=1 - _pad_r / _tot_w, top=1 - _pad_t / _tot_h,
        )

        conditions = ['a.2D PLF EMT', 'b.2D EMT', 'c.3D EMT']
        box_data = [df_g[df_g['Condition order for plots'] == c]['gene_metric'].dropna() for c in conditions]

        # Convert cm specs to data units
        # x-axis spans 3 data units (-0.5 to 2.5) over _box_fig_w_cm
        _cm_per_du = _box_fig_w_cm / 3.0  # cm per data unit
        _box_w_du = 0.279 / _cm_per_du           # box width in data units
        _dot_diam_cm = 0.0441
        _dot_diam_pt = _dot_diam_cm / 2.54 * 72  # ~1.25 pt
        _dot_area_pt2 = 3.14159 * (_dot_diam_pt / 2) ** 2  # marker area in pt²
        _dot_stroke_pt = 0.125
        _jitter_half_cm = 0.07 / 2.0             # ±half of 0.07 cm spread
        _jitter_half_du = _jitter_half_cm / _cm_per_du

        bp = ax.boxplot(box_data, positions=range(len(conditions)), widths=_box_w_du,
                        patch_artist=True, showfliers=False)
        for patch, c in zip(bp['boxes'], conditions):
            patch.set_facecolor(_box_palette[c])
            patch.set_alpha(0.5)
            patch.set_linewidth(0.5 * _pt * 72)
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_linewidth(0.5 * _pt * 72)

        # Overlay individual points
        for i, c in enumerate(conditions):
            vals = df_g[df_g['Condition order for plots'] == c]['gene_metric'].dropna()
            jitter = np.random.default_rng(42).uniform(-_jitter_half_du, _jitter_half_du, size=len(vals))
            ax.scatter(i + jitter, vals, s=_dot_area_pt2, color=_box_palette[c],
                       zorder=5, alpha=0.7, linewidth=_dot_stroke_pt, edgecolor=_box_palette[c])

        ax.set_ylim(10, 50)
        ax.set_yticks(np.arange(10, 51, 10))
        ax.set_ylabel('Time of expression change (h)', fontsize=5, fontfamily='Arial')
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([])
        ax.set_xlabel('', fontsize=5)

        ax.tick_params(axis='both', labelsize=5, width=0.5 * _pt * 72, length=3, direction='out', pad=2)
        for label in ax.get_yticklabels():
            label.set_fontfamily('Arial')
            label.set_fontsize(5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5 * _pt * 72)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Used_for information -
        # these figures are labeled as - Fig. 5
        # data ids with CDH1 are labeled as - Fig. 5S
        # data ids with EOMES are labeled as - Fig. 5N
        # data ids with TBXT are labeled as - Fig. 5I
        # data ids with SOX2 are labeled as - Fig. 5D
        plt.savefig(rf'{figs_dir}/Timing_of_expression_change_relative_to_EMT_induction_T_0_for_{g}_FigS5a.{out_type}', dpi=600, transparent=True)
        plt.close(fig)


    # Plotting scatter plots of gene metrics vs migration time
    metric_dict = {
        'EOMES':'Time of max EOMES expression (h)',
        'TBXT':'Time of max TBXT expression (h)',
        'CDH1':'Time of inflection of E-cad expression (h)',
        'SOX2':'Time of half-maximal SOX2 expression (h)'
    }
    for g, df_g in df_comb.groupby('Gene'):
        fig, ax = plt.subplots(1, 1, figsize=(_tot_w, _tot_h))
        fig.subplots_adjust(
            left=_pad_l / _tot_w, bottom=_pad_b / _tot_h,
            right=1 - _pad_r / _tot_w, top=1 - _pad_t / _tot_h,
        )

        for c in ['a.2D PLF EMT', 'b.2D EMT', 'c.3D EMT']:
            mask = df_g['Condition order for plots'] == c
            ax.scatter(
                df_g.loc[mask, 'gene_metric'],
                df_g.loc[mask, 'Migration Onset Time (Footprint Area Based)'],
                s=6, color=_box_palette[c], alpha=0.7, linewidth=0, zorder=5,
            )

        ax.set_xlim(10, 50)
        ax.set_ylim(10, 50)
        ax.set_xticks(np.arange(10, 51, 10))
        ax.set_yticks(np.arange(10, 51, 10))
        ax.set_xlabel(metric_dict[g], fontsize=5, fontfamily='Arial')
        ax.set_ylabel('Migration time (h)', fontsize=5, fontfamily='Arial')

        ax.tick_params(axis='both', labelsize=5, width=0.5 * _pt * 72, length=3, direction='out', pad=2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily('Arial')
            label.set_fontsize(5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5 * _pt * 72)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Used_for information -
        # these figures are labeled as - Fig. 5
        # data ids with CDH1 are labeled as - Fig. 5T
        # data ids with EOMES are labeled as - Fig. 5O
        # data ids with TBXT are labeled as - Fig. 5J
        # data ids with SOX2 are labeled as - Fig. 5E
        plt.savefig(fr'{figs_dir}/Scatter_plot_between_{g}_metric_and_migration_time.{out_type}', dpi=600, transparent=True)
        plt.close(fig)


    print('\n\n\n.......Statistical comparison for gene metric:')
    for g, df_g in df_comb.groupby('Gene'):
        print(f'gene={g}')
        x = df_g['gene_metric'][['2D PLF' in val for val in df_g['Experimental Condition'].values]].dropna()
        y = df_g['gene_metric'][['2D colony EMT' in val for val in df_g['Experimental Condition'].values]].dropna()
        z = df_g['gene_metric'][['3D lumenoid EMT' in val for val in df_g['Experimental Condition'].values]].dropna()

        print('2D PLF: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(x), np.median(x), np.std(x), np.min(x), np.max(x)))
        print('2D EMT: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(y), np.median(y), np.std(y), np.min(y), np.max(y)))
        print('3D EMT: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(z), np.median(z), np.std(z), np.min(z), np.max(z)))
        
        plot_tools.run_statistics(x,y,z)

    print('\n\n\n.......Statistical comparison for gene expression change relative to migration time:')
    for g, df_g in df_comb.groupby('Gene'):
        print(f'gene={g}')
        x = df_g['Difference'][['2D PLF' in val for val in df_g['Experimental Condition'].values]].dropna()
        y = df_g['Difference'][['2D colony EMT' in val for val in df_g['Experimental Condition'].values]].dropna()
        z = df_g['Difference'][['3D lumenoid EMT' in val for val in df_g['Experimental Condition'].values]].dropna()

        print('2D PLF: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(x), np.median(x), np.std(x), np.min(x), np.max(x)))
        print('2D EMT: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(y), np.median(y), np.std(y), np.min(y), np.max(y)))
        print('3D EMT: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(z), np.median(z), np.std(z), np.min(z), np.max(z)))
        
        plot_tools.run_statistics(x,y,z)

    print('\n\n\n.......Statistical comparison for gene expression change divided by migration time:')
    for g, df_g in df_comb.groupby('Gene'):
        print(f'gene={g}')
        x = df_g['Normalized_metric'][['2D PLF' in val for val in df_g['Experimental Condition'].values]].dropna()
        y = df_g['Normalized_metric'][['2D colony EMT' in val for val in df_g['Experimental Condition'].values]].dropna()
        z = df_g['Normalized_metric'][['3D lumenoid EMT' in val for val in df_g['Experimental Condition'].values]].dropna()

        print('2D PLF: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(x), np.median(x), np.std(x), np.min(x), np.max(x)))
        print('2D EMT: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(y), np.median(y), np.std(y), np.min(y), np.max(y)))
        print('3D EMT: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(z), np.median(z), np.std(z), np.min(z), np.max(z)))
        
        plot_tools.run_statistics(x,y,z)

    print('\n\n\n.......Correlation for gene metric vs migration time:')
    for g, df_g in df_comb.groupby('Gene'):
        print('\n')
        print('-------------')
        print(f'gene={g}')
        for cond in ['2D PLF', '2D colony EMT', '3D lumenoid EMT']:
            print('\n-----------')
            print(f'Condition: {cond}')
            # Filter for condition first, then drop rows where either column is NaN
            df_cond = df_g[[cond in val for val in df_g['Experimental Condition'].values]].copy()
            df_cond = df_cond[['Migration Onset Time (Footprint Area Based)', 'gene_metric']].dropna()
            migration = df_cond['Migration Onset Time (Footprint Area Based)']
            metric = df_cond['gene_metric']

            pearson, p_pvalue, _, _ = bootstrap_corr(migration, metric, "Pearson")
            spearman, s_pvalue, _, _ = bootstrap_corr(migration, metric, "Spearman")

            X = migration
            Y = metric

            # It's important to add a constant (intercept) to the model
            X = sm.add_constant(X)
            
            # Fit the Ordinary Least Squares (OLS) model
            model = sm.OLS(Y, X)
            results = model.fit()

            slope_p_value = results.pvalues['Migration Onset Time (Footprint Area Based)']
            r_squared = results.rsquared
            slope_coeff = results.params['Migration Onset Time (Footprint Area Based)']

            print(f"R-squared: {r_squared:.3g}")
            print(f"Slope (Coefficient for gene expression): {slope_coeff:.3g}")
            print(f"P-value for the slope: {slope_p_value:.3g}") # Using 'g' for scientific notation if needed

        print('\n\n-------Statistics for entire metric------------')
        # Drop rows where either column is NaN to ensure aligned data
        df_gene_clean = df_g[['Migration Onset Time (Footprint Area Based)', 'gene_metric']].dropna()
        migration = df_gene_clean['Migration Onset Time (Footprint Area Based)']
        metric = df_gene_clean['gene_metric']

        pearson, p_pvalue, _, _ = bootstrap_corr(migration, metric, "Pearson")
        spearman, s_pvalue, _, _ = bootstrap_corr(migration, metric, "Spearman")

        X = migration
        Y = metric

        # It's important to add a constant (intercept) to the model
        X = sm.add_constant(X)

        # Fit the Ordinary Least Squares (OLS) model
        model = sm.OLS(Y, X)
        results = model.fit()

        slope_p_value = results.pvalues['Migration Onset Time (Footprint Area Based)']
        r_squared = results.rsquared
        slope_coeff = results.params['Migration Onset Time (Footprint Area Based)']

        print(f"R-squared: {r_squared:.3g}")
        print(f"Slope (Coefficient for gene expression): {slope_coeff:.3g}")
        print(f"P-value for the slope: {slope_p_value:.3g}") # Using 'g' for scientific notation if needed


def _collagenase_boxplot_matplotlib(df_plot, save_path):
    """
    Create a publication-quality collagenase box plot with consistent styling.
    Same format as the MMPi plots.

    Parameters
    ----------
    df_plot : pd.DataFrame
        Filtered dataframe with 'Drug Concentration' and
        'Migration Onset Time (Footprint Area Based)' columns,
        already sorted by concentration.
    save_path : str
        Full path (including extension) where the figure is saved.
    """
    cm = 1 / 2.54
    pt = 1 / 72.0

    # Physical dimensions (same as MMPi)
    x_axis_cm = 4.2655
    y_axis_cm = 3.0951
    pad_l, pad_b, pad_r, pad_t = 0.70, 0.55, 0.10, 0.10
    total_w = x_axis_cm * cm + pad_l + pad_r
    total_h = y_axis_cm * cm + pad_b + pad_t

    # Orange: R=255, G=153, B=0
    orange = (255 / 255, 153 / 255, 0 / 255)
    line_w = 0.75  # pt

    concentrations = df_plot['Drug Concentration'].unique()  # already sorted
    box_data = [
        df_plot[df_plot['Drug Concentration'] == c]['Migration Onset Time (Footprint Area Based)'].dropna().values
        for c in concentrations
    ]

    fig, ax = plt.subplots(1, 1, figsize=(total_w, total_h))
    fig.subplots_adjust(
        left=pad_l / total_w,
        bottom=pad_b / total_h,
        right=1 - pad_r / total_w,
        top=1 - pad_t / total_h,
    )

    bp = ax.boxplot(
        box_data,
        positions=range(len(concentrations)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )

    # Style boxes
    for patch in bp['boxes']:
        patch.set_facecolor(orange)
        patch.set_alpha(0.5)
        patch.set_linewidth(line_w * pt * 72)
        patch.set_edgecolor('black')
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_linewidth(line_w * pt * 72)
            line.set_color('black')

    # Overlay individual points with jitter
    jitter_rng = np.random.default_rng(42)
    for i, vals in enumerate(box_data):
        jitter = jitter_rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            i + jitter, vals,
            s=12, color=orange, zorder=5, alpha=0.8,
            linewidth=0.3, edgecolor='black',
        )

    # Y-axis: 14-36, ticks every 2 h (shared range across MMPi and collagenase)
    ax.set_ylim(14, 36)
    ax.set_yticks(np.arange(14, 37, 2))
    ax.set_ylabel('Migration Onset Time (Footprint Area Based)',
                   fontsize=7, fontfamily='Arial', fontweight='regular')

    # X-axis
    ax.set_xticks(range(len(concentrations)))
    ax.set_xticklabels(concentrations, fontsize=7, fontfamily='Arial', rotation=0)
    ax.set_xlabel('Concentration of collagenase',
                   fontsize=7, fontfamily='Arial', fontweight='regular')

    # Tick styling: Arial 7pt regular
    ax.tick_params(axis='both', labelsize=7, width=line_w * pt * 72,
                   length=3, direction='out', pad=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(7)
        label.set_fontweight('regular')

    # Spine width 0.75 pt, remove top and right
    for spine in ax.spines.values():
        spine.set_linewidth(line_w * pt * 72)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(save_path, dpi=600, transparent=True)
    plt.close(fig)


def plot_collagenase_analysis(df, figs_dir, out_type):
    """
    Analyzes collagenase experiments and generates box plots for migration timing and statistical analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing the experimental data with collagenase treatment information
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """

    # Collagenase Experiment Graphs
    Path(rf'{figs_dir}/Collagenase').mkdir(exist_ok=True, parents=True)
    df_coll = df[df['Perturbation']=='Collagenase']

    df_summary = df_coll.drop_duplicates(subset=['Data ID'])
    df_summary['sort_value'] = df_summary['Drug Concentration'].apply(lambda c: float(c.split()[0]) if 'HBSS' not in c else -1)
    df_summary['Collagenease concentration (ug/mL)'] = [float(c.split(' ')[0])  if 'HBSS' not in c else 0 for c in df_summary['Drug Concentration'].values]
    df_summary = df_summary.sort_values(by=['sort_value'], axis=0)


    print('\n\n\n.......Statistical comparison for migration time with collagenase treatment:')
    for gene, df_gene in df_summary.groupby('Gene'):
        _collagenase_boxplot_matplotlib(
            df_gene,
            save_path=rf'{figs_dir}/Collagenase/Migration_box_plot_{gene}_per_conditions.{out_type}',
        )

        print('Gene: ', gene)
        key = {g:i for i, g in enumerate(df_gene['Drug Concentration'].unique())}
        concentrations = list(key.keys())
        migration = [[]]*len(concentrations)

        for g, d_g in df_gene.groupby('Drug Concentration'):
            migration[key[g]] = d_g['Migration Onset Time (Footprint Area Based)'].values

        for c, v in zip(concentrations, migration):
            print('{5}: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(v), np.median(v), np.std(v), np.min(v), np.max(v), c))

        print('Concentrations:')
        print(' '.join(['{0} n:{1} |'.format(t, len(v)) for t, v in zip(concentrations, migration)]))
        print('Significant difference between the distributions: applying post hoc with Holm-Bonferroni adjustment of p-value')

        print(sp.posthoc_mannwhitney(migration, p_adjust = 'holm'))


        # Calculating statistics for downward trend in collagenase concentrations vs migration time

        X = df_gene['Collagenease concentration (ug/mL)']
        Y = df_gene['Migration Onset Time (Footprint Area Based)']

        _ = bootstrap_corr(X, Y, "Spearman")

        # It's important to add a constant (intercept) to the model
        X = sm.add_constant(X)

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
        const_coef = results.params['const']

        print(f"R-squared: {r_squared:.3g}")
        print(f"Slope (Coefficient for concentration): {slope_coeff:.3g}")
        print(f"P-value for the slope: {slope_p_value:.3g}")

        alpha = 0.05
        if slope_p_value < alpha:
            print(f"\nConclusion: The p-value for the slope is less than {alpha}, indicating a statistically significant linear relationship between drug concentration and migration time.")
            print(f"On average, for each 1 ug/mL increase in drug concentration, the migration time changes by {slope_coeff:.2f} hours.")
        else:
            print("\nConclusion: The p-value for the slope is not less than 0.05, so we cannot conclude there is a significant linear relationship.")

        # Bootstrap correlation: Spearman's rho and Pearson
        print("\n--- Bootstrap Correlation (concentration vs migration time) ---")
        X_corr = df_gene['Collagenease concentration (ug/mL)'].values
        Y_corr = df_gene['Migration Onset Time (Footprint Area Based)'].values
        bootstrap_corr(X_corr, Y_corr, method="Spearman")
        bootstrap_corr(X_corr, Y_corr, method="Pearson")


def _mmpi_boxplot_matplotlib(df_plot, save_path):
    """
    Create a publication-quality MMPi box plot with consistent styling.

    Parameters
    ----------
    df_plot : pd.DataFrame
        Filtered dataframe with 'Drug Concentration' and
        'Migration Onset Time (Manual First Cell Detection)' columns,
        already sorted by concentration.
    save_path : str
        Full path (including extension) where the figure is saved.
    """
    cm = 1 / 2.54
    pt = 1 / 72.0

    # Physical dimensions
    x_axis_cm = 4.2655
    y_axis_cm = 3.0951
    pad_l, pad_b, pad_r, pad_t = 0.70, 0.55, 0.10, 0.10
    total_w = x_axis_cm * cm + pad_l + pad_r
    total_h = y_axis_cm * cm + pad_b + pad_t

    # Orange: R=255, G=153, B=0
    orange = (255 / 255, 153 / 255, 0 / 255)
    line_w = 0.75  # pt

    concentrations = df_plot['Drug Concentration'].unique()  # already sorted
    box_data = [
        df_plot[df_plot['Drug Concentration'] == c]['Migration Onset Time (Manual First Cell Detection)'].dropna().values
        for c in concentrations
    ]

    fig, ax = plt.subplots(1, 1, figsize=(total_w, total_h))
    fig.subplots_adjust(
        left=pad_l / total_w,
        bottom=pad_b / total_h,
        right=1 - pad_r / total_w,
        top=1 - pad_t / total_h,
    )

    bp = ax.boxplot(
        box_data,
        positions=range(len(concentrations)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )

    # Style boxes
    for patch in bp['boxes']:
        patch.set_facecolor(orange)
        patch.set_alpha(0.5)
        patch.set_linewidth(line_w * pt * 72)
        patch.set_edgecolor('black')
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_linewidth(line_w * pt * 72)
            line.set_color('black')

    # Overlay individual points with jitter
    jitter_rng = np.random.default_rng(42)
    for i, vals in enumerate(box_data):
        jitter = jitter_rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            i + jitter, vals,
            s=12, color=orange, zorder=5, alpha=0.8,
            linewidth=0.3, edgecolor='black',
        )

    # Y-axis: 14-36, ticks every 2 h (shared range across MMPi and collagenase)
    ax.set_ylim(14, 36)
    ax.set_yticks(np.arange(14, 37, 2))
    ax.set_ylabel('Migration Onset Time (Manual First Cell Detection)',
                   fontsize=7, fontfamily='Arial', fontweight='regular')

    # X-axis
    ax.set_xticks(range(len(concentrations)))
    ax.set_xticklabels(concentrations, fontsize=7, fontfamily='Arial', rotation=0)
    ax.set_xlabel('Concentration of MMP inhibitor',
                   fontsize=7, fontfamily='Arial', fontweight='regular')

    # Tick styling: Arial 7pt regular
    ax.tick_params(axis='both', labelsize=7, width=line_w * pt * 72,
                   length=3, direction='out', pad=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(7)
        label.set_fontweight('regular')

    # Spine width 0.75 pt, remove top and right
    for spine in ax.spines.values():
        spine.set_linewidth(line_w * pt * 72)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(save_path, dpi=600, transparent=True)
    plt.close(fig)


def plot_mmp_inhibitor_migration(df, figs_dir, out_type):
    """
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing the experimental data with MMPi treatment information
    figs_dir: str
        Directory where the figures will be saved
    out_type: str
        File type for the output figures (e.g. 'svg', 'png')
    """

    Path(rf'{figs_dir}/MMPi').mkdir(exist_ok=True, parents=True)
    df_coll = df[df['Perturbation']=='MMPi']

    df_summary = df_coll.drop_duplicates(subset=['Data ID'])
    df_summary.dropna(subset=['Migration Onset Time (Manual First Cell Detection)'],inplace=True)

    n_m = df_summary['Data ID'].nunique()
    df_summary['sort_value'] = df_summary['Drug Concentration'].apply(lambda c: float(c.split()[0]) if 'MMPi' not in c else -1)
    df_summary['MMPi concentration (uM)'] = [float(c.split(' ')[0])  if 'MMPi' not in c else 0 for c in df_summary['Drug Concentration'].values]
    df_summary = df_summary.sort_values(by=['sort_value'], axis=0)

    print('\n\n\n.......Statistical comparison for migration time with MMPi treatment:')
    for gene, df_gene in df_summary.groupby('Gene'):
        _mmpi_boxplot_matplotlib(
            df_gene,
            save_path=rf'{figs_dir}/MMPi/Migration_box_plot_{gene}_per_conditions.{out_type}',
        )

        print('Gene: ', gene)
        key = {g:i for i, g in enumerate(df_gene['Drug Concentration'].unique())}
        concentrations = list(key.keys())
        migration = [[]]*len(concentrations)

        for g, d_g in df_gene.groupby('Drug Concentration'):
            migration[key[g]] = d_g['Migration Onset Time (Manual First Cell Detection)'].values

        for c, v in zip(concentrations, migration):
            print('{5}: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(v), np.median(v), np.std(v), np.min(v), np.max(v), c))

        print('Concentrations:')
        print(' '.join(['{0} n:{1} |'.format(t, len(v)) for t, v in zip(concentrations, migration)]))
        print('Significant difference between the distributions: applying post hoc with Holm-Bonferroni adjustment of p-value')    
        
        print(sp.posthoc_mannwhitney(migration, p_adjust = 'holm'))


        # Calculating statistics for downward trend in collagenase concentrations vs migration time

        X = df_gene['MMPi concentration (uM)']
        Y = df_gene['Migration Onset Time (Manual First Cell Detection)']

        # It's important to add a constant (intercept) to the model
        X = sm.add_constant(X)

        # Fit the Ordinary Least Squares (OLS) model
        model = sm.OLS(Y, X)
        results = model.fit()

        # Print the full summary of the regression results
        print(results.summary())

        # Interpretation of key results from the linear regression
        print("\n--- Interpretation of Linear Regression Results ---")
        slope_p_value = results.pvalues['MMPi concentration (uM)']
        r_squared = results.rsquared
        slope_coeff = results.params['MMPi concentration (uM)']
        const_coeff = results.params['const']
        
        print(f"R-squared: {r_squared:.4f}")
        print(f"Slope (Coefficient for concentration): {slope_coeff:.4f}")
        print(f"P-value for the slope: {slope_p_value:.4g}") # Using 'g' for scientific notation if needed

        alpha = 0.05
        if slope_p_value < alpha:
            print(f"\nConclusion: The p-value for the slope is less than {alpha}, indicating a statistically significant linear relationship between drug concentration and migration time.")
            print(f"On average, for each 1 uM increase in drug concentration, the migration time changes by {slope_coeff:.2f} hours.")
        else:
            print("\nConclusion: The p-value for the slope is not less than 0.05, so we cannot conclude there is a significant linear relationship.")

        # Bootstrap correlation: Spearman's rho and Pearson
        print("\n--- Bootstrap Correlation (concentration vs migration time) ---")
        X_corr = df_gene['MMPi concentration (uM)'].values
        Y_corr = df_gene['Migration Onset Time (Manual First Cell Detection)'].values
        bootstrap_corr(X_corr, Y_corr, method="Spearman")
        bootstrap_corr(X_corr, Y_corr, method="Pearson")


def _plot_mmpi_for_cell_line(df, figs_dir, out_type, cell_line, subfolder):
    """
    Helper function to plot MMPi migration analysis for a specific cell line.
    Follows the same format as plot_mmp_inhibitor_migration (AICS-0061).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe containing the experimental data
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    cell_line : str
        Cell line to filter for (e.g. 'AICS-0036', 'AICS-0000')
    subfolder : str
        Subdirectory name under figs_dir for saving plots
    """

    Path(rf'{figs_dir}/{subfolder}').mkdir(exist_ok=True, parents=True)
    df_coll = df[(df['Perturbation']=='MMPi') & (df['Cell Line']==cell_line)]

    df_summary = df_coll.drop_duplicates(subset=['Data ID'])
    df_summary.dropna(subset=['Migration Onset Time (Manual First Cell Detection)'], inplace=True)

    if df_summary.empty:
        print(f'No MMPi data with migration time for cell line {cell_line}')
        return

    n_m = df_summary['Data ID'].nunique()
    df_summary['sort_value'] = df_summary['Drug Concentration'].apply(lambda c: float(c.split()[0]) if 'MMPi' not in c else -1)
    df_summary['MMPi concentration (uM)'] = [float(c.split(' ')[0]) if 'MMPi' not in c else 0 for c in df_summary['Drug Concentration'].values]
    df_summary = df_summary.sort_values(by=['sort_value'], axis=0)

    print(f'\n\n\n.......Statistical comparison for migration time with MMPi treatment ({cell_line}):')

    _mmpi_boxplot_matplotlib(
        df_summary,
        save_path=rf'{figs_dir}/{subfolder}/Migration_box_plot_{cell_line}_per_conditions.{out_type}',
    )

    print(f'Cell Line: {cell_line}')
    key = {g: i for i, g in enumerate(df_summary['Drug Concentration'].unique())}
    concentrations = list(key.keys())
    migration = [[]] * len(concentrations)

    for g, d_g in df_summary.groupby('Drug Concentration'):
        migration[key[g]] = d_g['Migration Onset Time (Manual First Cell Detection)'].values

    for c, v in zip(concentrations, migration):
        print('{5}: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(v), np.median(v), np.std(v), np.min(v), np.max(v), c))

    print('Concentrations:')
    print(' '.join(['{0} n:{1} |'.format(t, len(v)) for t, v in zip(concentrations, migration)]))
    print('Significant difference between the distributions: applying post hoc with Holm-Bonferroni adjustment of p-value')

    print(sp.posthoc_mannwhitney(migration, p_adjust='holm'))

    # Linear regression for concentration vs migration time
    X = df_summary['MMPi concentration (uM)']
    Y = df_summary['Migration Onset Time (Manual First Cell Detection)']

    X = sm.add_constant(X)

    model = sm.OLS(Y, X)
    results = model.fit()

    print(results.summary())

    print("\n--- Interpretation of Linear Regression Results ---")
    slope_p_value = results.pvalues['MMPi concentration (uM)']
    r_squared = results.rsquared
    slope_coeff = results.params['MMPi concentration (uM)']
    const_coeff = results.params['const']

    print(f"R-squared: {r_squared:.4f}")
    print(f"Slope (Coefficient for concentration): {slope_coeff:.4f}")
    print(f"P-value for the slope: {slope_p_value:.4g}")

    alpha = 0.05
    if slope_p_value < alpha:
        print(f"\nConclusion: The p-value for the slope is less than {alpha}, indicating a statistically significant linear relationship between drug concentration and migration time.")
        print(f"On average, for each 1 uM increase in drug concentration, the migration time changes by {slope_coeff:.2f} hours.")
    else:
        print("\nConclusion: The p-value for the slope is not less than 0.05, so we cannot conclude there is a significant linear relationship.")

    # Bootstrap correlation: Spearman's rho and Pearson
    print(f"\n--- Bootstrap Correlation for {cell_line} (concentration vs migration time) ---")
    X_corr = df_summary['MMPi concentration (uM)'].values
    Y_corr = df_summary['Migration Onset Time (Manual First Cell Detection)'].values
    bootstrap_corr(X_corr, Y_corr, method="Spearman")
    bootstrap_corr(X_corr, Y_corr, method="Pearson")


def plot_mmp_inhibitor_migration_aics0036(df, figs_dir, out_type):
    """
    Plot MMPi migration analysis for AICS-0036 cell line (Gene: AAVS1).
    Same format as AICS-0061 plot with drug concentration conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the experimental data with MMPi treatment information
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """
    _plot_mmpi_for_cell_line(df, figs_dir, out_type, cell_line='AICS-0036', subfolder='MMPi_AICS-0036')


def plot_mmp_inhibitor_migration_aics0000(df, figs_dir, out_type):
    """
    Plot MMPi migration analysis for AICS-0000 cell line (parental, no gene tag).
    Same format as AICS-0061 plot with drug concentration conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the experimental data with MMPi treatment information
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """
    _plot_mmpi_for_cell_line(df, figs_dir, out_type, cell_line='AICS-0000', subfolder='MMPi_AICS-0000')


def analyze_crispr_knockdown_experiments(df, figs_dir, out_type):
    """
    Analyzes CRISPRi knockdown experiments and generates box plots for migration timing.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing the experimental data with gene knockdown information
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """

    (Path(figs_dir) / 'CRISPRi').mkdir(parents=True, exist_ok=True)
    df_f = df[df['Gene']=='CLYBL'].drop_duplicates(subset=['Data ID'])

    df_f = df_f[
        (df_f['Single Colony Or Lumenoid At Time of Migration']==True)& \
        (df_f['Absence Of Migrating Cells Coming From Colony Out Of FOV At Time Of Migration']==True)& \
        (df_f['Absence Of Excessive Cell Death']==True)& \
        (df_f['Image Size Z']==30)& \
        (df_f['Fixation Status']=='Live Cells')
    ]

    df_summary = df_f.groupby('Data ID').agg('first').reset_index()
    df_summary['Migration Onset Time (Footprint Area Based)'] = (df_summary['Migration Onset Time (Footprint Area Based)'] * 2).round() / 2
    df_summary['Colony'] = df_summary['Experimental Condition'].apply(lambda s: s.split(' CRISPRi ')[0] if 'CRISPRi' in s else s)
    df_summary = df_summary[df_summary['Colony']=='3D lumenoid EMT']
    df_summary['Knockdown'] = df_summary['Experimental Condition'].apply(lambda s: s.split(' CRISPRi ')[-1] if 'CRISPRi' in s else 'Control')
    df_summary['Target'] = df_summary['Knockdown'].apply(lambda s: s.replace(' g1','').replace(' g2','').replace(' g3',''))

    print('Generating Box plots for migration timing for CLYBL gene knockdown experiment')
    df_summary['Condition order for plots']=df_summary['Knockdown'].apply(lambda x: 'a.Control' if 'Control' in x else 'b.Seq Scr' if 'seq' in x else f'c.{x}' if 'g1' in x else f'd.{x}' if 'g2' in x else f'e.{x}')
    df_summary = df_summary.sort_values(['Condition order for plots'])

    print('\n\n\n.......Statistical comparison for migration time with gene knockdowns:')

    for gene in ['TBXT', 'Snail']:
        df_gene = df_summary[(df_summary['Target']=='Control')|(df_summary['Target']=='seq scr')|(df_summary['Target']==gene)]

        color_map={tgt:'orange' for tgt in df_gene['Condition order for plots'].unique()}

        fig_mig_g = px.box(df_gene, y='Migration Onset Time (Footprint Area Based)', x='Knockdown', points='all', color='Condition order for plots', color_discrete_map=color_map, template='simple_white',range_y=(10,35),width=800, height=600)
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
            values[key[g]] = d_g['Migration Onset Time (Footprint Area Based)'].values

        for c, v in zip(targets, values):
            print('{5}: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(v), np.median(v), np.std(v), np.min(v), np.max(v), c))

        print('Targets:')
        print(' '.join(['{0} n:{1} |'.format(t, len(v)) for t, v in zip(targets, values)]))
        print('Significant difference between the distributions: applying post hoc with Holm-Bonferroni adjustment of p-value')
        print(sp.posthoc_mannwhitney(values, p_adjust = 'holm'))


def plot_zo1_heatmaps(df, figs_dir, out_type):
    """
    Plot Z01 heatmaps

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing the experimental data with ZO1 information
    figs_dir: str
        Directory where the figures will be saved
    out_type: str
        File type for the output figures (e.g. 'svg', 'png')
    """

    # Filtering the dataset to only ZO1 data
    (Path(figs_dir) / 'ZO1').mkdir(parents=True, exist_ok=True)

    df_f = df[df['Gene']=='TJP1']

    # Adding a Timepoint (h) column which converts frames into hours using  the Timelapse Interval column value
    time_interval=30 #int(''.join(filter(lambda i: i.isdigit(),df_f['Timelapse Interval'].unique()[0] )))
    df_f['Timepoint (h)']=df_f['Timepoint']*(time_interval/60)

    # For plotting the conditions in the order- 2D PLF EMT, 2D EMT, 3D EMT
    df_f['Condition order for plots']=df_f['Experimental Condition'].apply(lambda x: 'a.2D PLF EMT' if '2D PLF colony EMT' in x else 'b.2D EMT' if '2D colony EMT' in x else 'c.3D EMT')

    df_summary = df_f.groupby(['Data ID']).agg('first').reset_index()
    df_zo = df_f[df_f.Gene=='TJP1']

    df_zo_examples = df_zo[df_zo['Data ID'].isin(const.EXAMPLE_ZO1_IDS)]

    # Generating and saving the heatmaps
    plot_tools.Intensity_over_z(df_zo_examples, figs_dir=figs_dir/'ZO1', out_type=out_type)


def plot_inside_outside_migration_timing(df, figs_dir, out_type):
    """
    Analyzes the inside-outside classification of nuclei in the basement membrane and plots the fraction of nuclei outside the lumen over time.
    Also plots the migration time estimated from inside and outside classification of nuclei w.r.t basement membrane vs migration time estimated from area at the glass.

    Parameters:
    df : pd.DataFrame
        Dataframe containing the experimental data with inside-outside classification of nuclei
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """

    dfio_merge = load_io_data(df)

    n_movies_io=dfio_merge['Data ID'].nunique()

    # Grouping the data and getting the fraction of 'True' values in 'Inside' column to get fraction of nuclei inside the basement membrane for each movie
    dfio_grouped=dfio_merge.groupby([
        'Condition order for plots',
        'Gene',
        'Data ID',
        'Time hr'
    ]).agg({
        'Inside':'mean',
        'Migration Onset Time (Footprint Area Based)':'first', 
        'Migration Onset Time (Inside/Outside Basement Membrane Based)':'first'
    }).reset_index()
    dfio_grouped['Fraction_outside']=1-dfio_grouped['Inside'] #fraction of nuclei outside the basement membrane

    (Path(figs_dir) / 'Inside-Outside').mkdir(parents=True, exist_ok=True)

    # Publication dimensions for inside/outside plots
    _io_cm = 1 / 2.54
    _io_pt = 1 / 72.0
    _io_x_cm = 3.8883
    _io_y_cm = 3.1424
    _io_pad_l, _io_pad_b, _io_pad_r, _io_pad_t = 0.65, 0.50, 0.05, 0.05
    _io_tot_w = _io_x_cm * _io_cm + _io_pad_l + _io_pad_r
    _io_tot_h = _io_y_cm * _io_cm + _io_pad_b + _io_pad_t

    # Colors: purple, blue, orange (R=255,G=153,B=0)
    _io_palette = {
        'a.2D PLF EMT': (139/255, 0/255, 139/255),
        'b.2D EMT':     (0/255, 191/255, 255/255),
        'c.3D EMT':     (255/255, 153/255, 0/255),
    }

    # Plotting the fraction of nuclei over time (Fig.5G)
    fig, ax = plt.subplots(1, 1, figsize=(_io_tot_w, _io_tot_h))
    fig.subplots_adjust(
        left=_io_pad_l / _io_tot_w, bottom=_io_pad_b / _io_tot_h,
        right=1 - _io_pad_r / _io_tot_w, top=1 - _io_pad_t / _io_tot_h,
    )
    sns.lineplot(
        dfio_grouped, x='Time hr', y='Fraction_outside',
        hue='Condition order for plots', palette=_io_palette,
        errorbar=('pi', 50), estimator=np.median,
        linewidth=0.75 * _io_pt * 72,  # 0.75 pt
        ax=ax,
    )

    # Error band: fill only, no stroke
    for child in ax.get_children():
        if isinstance(child, mpl.collections.PolyCollection):
            child.set_edgecolor('none')
            child.set_linewidth(0)

    # Y-axis
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel('Fraction of nuclei outside of basement membrane', fontsize=7, fontfamily='Arial')

    # X-axis: 0-50, interval 10, with padding
    ax.set_xlim(-2, 52)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_xlabel('Time (h)', fontsize=7, fontfamily='Arial')

    # Tick styling
    ax.tick_params(axis='both', labelsize=7, width=0.5 * _io_pt * 72,
                   length=3, direction='out', pad=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(7)

    # Spine width 0.5 pt
    for spine in ax.spines.values():
        spine.set_linewidth(0.5 * _io_pt * 72)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Remove legend
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    plt.savefig(fr'{figs_dir}/Inside-Outside/Fraction_of_nuclei_outside_lumen.{out_type}', dpi=600, transparent=True)
    plt.close(fig)


    dfio_scatter=dfio_merge.groupby([
        'Condition order for plots',
        'Data ID',
    ]).agg({
        'Migration Onset Time (Footprint Area Based)':'first', 
        'Migration Onset Time (Inside/Outside Basement Membrane Based)':'first'
    })

    # Plotting migration time estimated from inside and outside classification of nuclei w.r.t basement memebrane vs migration time estimated from area at the glass (Fig. 5I)
    fig_scatter, ax = plt.subplots(1,1, figsize=(10,10))
    fig_scatter = sns.scatterplot(dfio_scatter, x='Migration Onset Time (Footprint Area Based)', y='Migration Onset Time (Inside/Outside Basement Membrane Based)', hue='Condition order for plots', palette=const.COLOR_MAP, s=100, alpha=0.7, linewidth=2, edgecolor='coral', legend=False)
    plt.xlim(20,36)
    plt.ylim(20,36)

    plt.xlabel('Migration Time from area at glass (h)', fontsize=16)
    plt.ylabel('Migration Time fraction of nuclei outside basement membrane (h)', fontsize=16)
    plt.rcParams.update({'font.size':16})
    plt.savefig(fr'{figs_dir}/Inside-Outside/Scatter_plot_between_computer_migration_area_on_glass_vs_inside_outside.{out_type}', dpi=600, transparent=True)
    plt.close()

    # Bias-corrected scatter plot (saved alongside the uncorrected scatter)
    A_io = dfio_scatter['Migration Onset Time (Footprint Area Based)'].to_numpy(float)
    B_io = dfio_scatter['Migration Onset Time (Inside/Outside Basement Membrane Based)'].to_numpy(float)

    # Bland-Altman stats for bias
    diff_io = B_io - A_io
    bias_io = float(diff_io.mean())
    B_corr = B_io - bias_io

    # Linear regression on bias-corrected data
    X_ols = sm.add_constant(A_io)
    ols_corr = sm.OLS(B_corr, X_ols).fit()
    slope_corr = float(ols_corr.params[1])
    intercept_corr = float(ols_corr.params[0])
    r2_corr = float(ols_corr.rsquared)
    n_io = len(A_io)

    fig_bc, ax_bc = plt.subplots(1, 1, figsize=(_io_tot_w, _io_tot_h))
    fig_bc.subplots_adjust(
        left=_io_pad_l / _io_tot_w, bottom=_io_pad_b / _io_tot_h,
        right=1 - _io_pad_r / _io_tot_w, top=1 - _io_pad_t / _io_tot_h,
    )

    # Per-condition scatter with publication colors
    dfio_scatter_reset = dfio_scatter.reset_index()
    for c in ['a.2D PLF EMT', 'b.2D EMT', 'c.3D EMT']:
        mask = dfio_scatter_reset['Condition order for plots'] == c
        ax_bc.scatter(
            dfio_scatter_reset.loc[mask, 'Migration Onset Time (Footprint Area Based)'],
            dfio_scatter_reset.loc[mask, 'Migration Onset Time (Inside/Outside Basement Membrane Based)'] - bias_io,
            s=6, color=_io_palette[c], alpha=0.7, linewidth=0, zorder=5,
        )

    # Unity line
    lo_bc, hi_bc = 18, 38
    ax_bc.plot([lo_bc, hi_bc], [lo_bc, hi_bc], linestyle='--', color='grey',
               linewidth=0.5 * _io_pt * 72, zorder=1)

    # Regression line
    xline = np.array([lo_bc, hi_bc])
    yline = intercept_corr + slope_corr * xline
    ax_bc.plot(xline, yline, linestyle='-', color='black',
               linewidth=0.75 * _io_pt * 72, zorder=2)

    ax_bc.set_xlim(lo_bc, hi_bc)
    ax_bc.set_ylim(lo_bc, hi_bc)
    ax_bc.set_xticks(np.arange(20, 37, 4))
    ax_bc.set_yticks(np.arange(20, 37, 4))
    ax_bc.set_xlabel('Migration time from area at glass (h)', fontsize=7, fontfamily='Arial')
    ax_bc.set_ylabel('Migration time in/out (bias-corrected) (h)', fontsize=7, fontfamily='Arial')

    # Text annotation: bias shift, R², slope, n
    text_str = (
        f'Bias shift: {bias_io:+.2f} h\n'
        f'R\u00B2 = {r2_corr:.3f}\n'
        f'Slope = {slope_corr:.3f}\n'
        f'n = {n_io}'
    )
    ax_bc.text(0.05, 0.95, text_str, transform=ax_bc.transAxes,
               fontsize=5, fontfamily='Arial', verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

    # Tick styling
    ax_bc.tick_params(axis='both', labelsize=7, width=0.5 * _io_pt * 72,
                      length=3, direction='out', pad=2)
    for label in ax_bc.get_xticklabels() + ax_bc.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(7)

    for spine in ax_bc.spines.values():
        spine.set_linewidth(0.5 * _io_pt * 72)
    ax_bc.spines['top'].set_visible(False)
    ax_bc.spines['right'].set_visible(False)

    plt.savefig(fr'{figs_dir}/Inside-Outside/Bias_corrected_scatter_migration_area_on_glass_vs_inside_outside.{out_type}', dpi=600, transparent=True)
    plt.close(fig_bc)

    # Plotting example to show how migration time is estimated from fraction of nuclei outside the basement membrane over time (Fig. 5H )
    df_io_id = dfio_grouped[dfio_grouped['Data ID']==const.EXAMPLE_IO_ID]
    fig, ax = plt.subplots(1, 1, figsize=(_io_tot_w, _io_tot_h))
    fig.subplots_adjust(
        left=_io_pad_l / _io_tot_w, bottom=_io_pad_b / _io_tot_h,
        right=1 - _io_pad_r / _io_tot_w, top=1 - _io_pad_t / _io_tot_h,
    )

    x_io = df_io_id['Migration Onset Time (Inside/Outside Basement Membrane Based)'].values[0]
    # Find nearest Time hr to the migration onset time
    nearest_idx = (df_io_id['Time hr'] - x_io).abs().idxmin()
    x_io = df_io_id.loc[nearest_idx, 'Time hr']
    y_io = df_io_id.loc[nearest_idx, 'Fraction_outside']
    ax.plot(df_io_id['Time hr'], df_io_id['Fraction_outside'], c=(255/255, 153/255, 0/255), linewidth=0.75 * _io_pt * 72)
    ax.scatter(x_io, y_io, c='black', marker='D', s=8, zorder=5)

    # Y-axis
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel('Fraction of nuclei outside of basement membrane', fontsize=7, fontfamily='Arial')

    # X-axis
    ax.set_xlim(-2, 52)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_xlabel('Time (h)', fontsize=7, fontfamily='Arial')

    # Tick styling
    ax.tick_params(axis='both', labelsize=7, width=0.5 * _io_pt * 72,
                   length=3, direction='out', pad=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(7)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5 * _io_pt * 72)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(fr'{figs_dir}/Individual_Examples/Example_migration_estimation_fraction_nuclei_outside_basement_membrane.{out_type}', dpi=600, transparent=True)
    plt.close(fig)

    print('\n\n\n.......Statistical comparison for migration time using Area-at-Mask vs Inside-Outside:')
    X = dfio_scatter['Migration Onset Time (Footprint Area Based)']
    Y = dfio_scatter['Migration Onset Time (Inside/Outside Basement Membrane Based)']

    p_results = bootstrap_corr(X.values, Y.values, "Pearson")
    r_results = bootstrap_corr(X.values, Y.values, "Spearman")
    print('n: {0:d}'.format(n_movies_io))

    X = dfio_scatter['Migration Onset Time (Footprint Area Based)']
    Y = dfio_scatter['Migration Onset Time (Inside/Outside Basement Membrane Based)']
    X = sm.add_constant(X)

    # Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(Y, X)
    results = model.fit()
    slope_p_value = results.pvalues['Migration Onset Time (Footprint Area Based)']
    r_squared = results.rsquared
    slope_coeff = results.params['Migration Onset Time (Footprint Area Based)']

    print(f"R-squared: {r_squared:.3g}")
    print(f"Slope (Coefficient for migration timing): {slope_coeff:.3g}")
    print(f"P-value for the slope: {slope_p_value:.3g}")

def _bland_altman_stats(A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
    """
    Compute Bland–Altman stats + diagnostics (using normal approx for CIs).
    
    Parameters
    ----------
    A : np.ndarray
        Values from one measurement method 
    B: np.ndarray
        Values from a differeing measurement method
    
    Returns
    -------
    Dict: Dictionary containing Bland-Altman statistics

    """
    
    mean_vals = (A + B) / 2.0
    diff_vals = B - A
    n = diff_vals.size

    bias = float(diff_vals.mean())
    sd = float(diff_vals.std(ddof=1))
    z = 1.96  # 95% normal quantile

    loa_lower = bias - z * sd
    loa_upper = bias + z * sd

    # 95% CI for bias (normal approx)
    se_bias = sd / np.sqrt(n)
    bias_ci_low = bias - z * se_bias
    bias_ci_high = bias + z * se_bias

    # 95% CIs for LoA (Bland & Altman 1999)
    se_loa = sd * np.sqrt(1.0 / n + (z**2) / (2 * (n - 1)))
    loa_lower_ci_low = loa_lower - z * se_loa
    loa_lower_ci_high = loa_lower + z * se_loa
    loa_upper_ci_low = loa_upper - z * se_loa
    loa_upper_ci_high = loa_upper + z * se_loa

    # Normality of differences (Jarque–Bera via statsmodels)
    jb_stat, jb_p, _, _ = jarque_bera(diff_vals)

    # Proportional bias: diff ~ mean (OLS with intercept)
    X = sm.add_constant(mean_vals)
    ols = sm.OLS(diff_vals, X).fit()
    slope = float(ols.params[1])
    slope_p = float(ols.pvalues[1])
    r2_prop = float(ols.rsquared)

    # Heteroscedasticity (Breusch–Pagan) on residuals vs mean
    bp_stat, bp_p, _, _ = het_breuschpagan(ols.resid, X)

    return {
        "n": int(n),
        "bias": bias,
        "bias_ci_low": float(bias_ci_low),
        "bias_ci_high": float(bias_ci_high),
        "sd_diff": sd,
        "loa_lower": float(loa_lower),
        "loa_upper": float(loa_upper),
        "loa_lower_ci_low": float(loa_lower_ci_low),
        "loa_lower_ci_high": float(loa_lower_ci_high),
        "loa_upper_ci_low": float(loa_upper_ci_low),
        "loa_upper_ci_high": float(loa_upper_ci_high),
        "normality_test": "Jarque–Bera",
        "normality_stat": float(jb_stat),
        "normality_p": float(jb_p),
        "prop_bias_slope": slope,
        "prop_bias_pval": slope_p,
        "prop_bias_r2": r2_prop,
        "breusch_pagan_stat": float(bp_stat),
        "breusch_pagan_pval": float(bp_p),
        "mean_vals": mean_vals,
        "diff_vals": diff_vals,
    }

def _regress_fit(A: np.ndarray, B: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit linear regression of B on A and return slope, intercept, and R².
    
    Parameters
    ----------
    A : np.ndarray
        Values from one measurement method 
    B: np.ndarray
        Values from a differeing measurement method

    Returns
    -------
    Tuple: slope, intercept, and r2 measurements for a linear regression of the two measurement methods
    """
    X = sm.add_constant(A)
    fit = sm.OLS(B, X).fit()
    slope = float(fit.params[1])
    intercept = float(fit.params[0])
    r2 = float(fit.rsquared)
    return slope, intercept, r2

def _plot_scatter(A: np.ndarray, B: np.ndarray, out_svg: Path, title: str, y_label: str) -> Tuple[float, float, float]:
    """
    Scatter plot data with regression line and unity line.
    
    Parameters:
    -----------
    A : np.ndarray
        Values from one measurement method 
    B: np.ndarray
        Values from a differeing measurement method
    out_svg: Path
        Path to where to save output figure svg
    title: str
        Title for scatter plot of two measurement methods
    y_label: str
        Label for the y axis

    Returns:
    --------
    Tuple: slope, intercept and r2 from linear regression of two measurement methods
           performed as part of the plotting
    """
    
    slope, intercept, r2 = _regress_fit(A, B)
    lo = float(min(A.min(), B.min()) - 0.5)
    hi = float(max(A.max(), B.max()) + 0.5)
    xline = np.array([lo, hi])
    yline = intercept + slope * xline

    plt.figure(figsize=(6, 6))
    plt.scatter(A, B)
    plt.plot([lo, hi], [lo, hi], linestyle="--")  # unity
    plt.plot(xline, yline, linestyle="-")         # regression
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("Method A: area-at-glass (h)")
    plt.ylabel(y_label)
    plt.title(f"{title} (slope={slope:.3f}, R²={r2:.3f})")
    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()
    return slope, intercept, r2

def _plot_bland_altman(mean_vals: np.ndarray, diff_vals: np.ndarray, out_svg: Path, bias: float, loa_lower: float, loa_upper: float) -> None:
    """
    Generate Bland-Altman plot of bias vs means

    mean_vals: np.ndarray
        Mean of two methods for measuring the migration onset time
    diff_vals: np.ndarray
        Difference between two methods for measuring the migration onset time
    out_svg: Path
        Path to where to save output figure svg
    bias: float
        Mean of bias, or difference, between two measurement methods
    loa_lower: float
        Lower limit of agreement
    loa_upper: flaot
        Upper limit of agreement
    """
    
    plt.figure(figsize=(6, 6))
    plt.scatter(mean_vals, diff_vals)
    plt.axhline(bias, linestyle="--")
    plt.axhline(loa_lower, linestyle="--")
    plt.axhline(loa_upper, linestyle="--")
    plt.xlabel("Mean of methods (h)")
    plt.ylabel("Difference (B − A) (h)")
    plt.title(f"Bland–Altman (bias={bias:+.2f} h, LoA [{loa_lower:+.2f}, {loa_upper:+.2f}] h)")
    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()

def _write_report(s: Dict[str, Any], out_txt: Path) -> None:
    """
    Write a text report summarizing Bland–Altman stats.
    
    Parameters
    ----------
    s : Dict[str, Any]
        Dictionary containing Bland-Altman statistics
    out_txt : Path
        Path to where to save output text report
    """
    
    # UTF-8 avoids Windows cp1252 errors on characters like "−", "±", "²"
    lines = [
        f"n = {s['n']}",
        f"Mean bias (B − A): {s['bias']:+.2f} h",
        f"95% CI for bias: [{s['bias_ci_low']:+.2f}, {s['bias_ci_high']:+.2f}] h",
        f"SD of differences: {s['sd_diff']:.2f} h",
        f"95% LoA: [{s['loa_lower']:+.2f}, {s['loa_upper']:+.2f}] h",
        f"95% CI for lower LoA: [{s['loa_lower_ci_low']:+.2f}, {s['loa_lower_ci_high']:+.2f}] h",
        f"95% CI for upper LoA: [{s['loa_upper_ci_low']:+.2f}, {s['loa_upper_ci_high']:+.2f}] h",
        f"Proportional bias (diff ~ mean): slope = {s['prop_bias_slope']:+.2f} h per hour, "
        f"p = {s['prop_bias_pval']:.2g}, R² = {s['prop_bias_r2']:.2f}",
        f"Normality ({s['normality_test']}): stat = {s['normality_stat']:.2f}, p = {s['normality_p']:.2g}",
        f"Heteroscedasticity (Breusch–Pagan): stat = {s['breusch_pagan_stat']:.2f}, p = {s['breusch_pagan_pval']:.2g}",
    ]
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def run_bland_altman_analysis(
        df, 
        FIGS_DIR, 
        a_col="Migration Onset Time (Footprint Area Based)", 
        b_col="Migration Onset Time (Inside/Outside Basement Membrane Based)", 
        id_col="Data ID"
):
    """
    Main function to run Bland-Altmane analysis comaring
    two different measurement methods for the migration onset timing
    
    df: pd.DataFrame
        Dataframe containing the experimental data with migration timing information
    FIGS_DIR: str
        Directory where the figures will be saved
    a_col: str
        Column name for the first measurement method (default: "Migration Time (h)")
    b_col: str
        Column name for the second measurement method (default: "Migration Time InOut (h)")
    id_col: str
        Column name for the unique identifier of each data point (default: "Data ID")
    """

    # Set up dataset
    dfio_merge = load_io_data(df)
    dfio_scatter=dfio_merge.groupby([
        'Condition order for plots',
        'Data ID',
    ]).agg({
        'Migration Onset Time (Footprint Area Based)':'first', 
        'Migration Onset Time (Inside/Outside Basement Membrane Based)':'first'
    })

    A = dfio_scatter[a_col].to_numpy(float)
    B =dfio_scatter[b_col].to_numpy(float)

    # Stats
    s = _bland_altman_stats(A, B)

    # Bias-corrected points (shift B by −bias so corrected B aligns with A)
    B_corr = B - s["bias"]
   
    # Figures (SVG)
    FIGS_DIR = Path(FIGS_DIR)
    _plot_scatter(A, B, FIGS_DIR / "uncorrected_scatter.svg", "Uncorrected scatter", "Method B: in/out (h)")
    _plot_scatter(A, B_corr, FIGS_DIR / "bias_corrected_scatter.svg", f"Bias-corrected scatter (shift {(-s['bias']):+.2f} h)", "Method B (bias-corrected) (h)")
    _plot_bland_altman(s["mean_vals"], s["diff_vals"], FIGS_DIR / "bland_altman.svg", s["bias"], s["loa_lower"], s["loa_upper"])

    # Summary CSV + text report
    pd.DataFrame({k: [v] for k, v in s.items() if not isinstance(v, np.ndarray)}).to_csv(FIGS_DIR / "bland_altman_summary.csv", index=False, encoding="utf-8")
    _write_report(s, FIGS_DIR / "report.txt")

def plot_bmp_inhibitor_migration(df, figs_dir: str, out_type):
    (Path(figs_dir) / 'BMP').mkdir(parents=True, exist_ok=True)

    df_BMP = df[df['Perturbation']=='LDN']
    def _parse_treatment(s):
        s = s.replace('BMP4 EMT','BMP4')
        s = s.split('BMP4')
        out = ['BMP4']
        if len(s)>1:
            out.append(s[-1].lstrip(' '))
        return ' '.join(out)

    df_BMP['Treatment'] = df_BMP['Experimental Condition'].apply(_parse_treatment)
    df_BMP['Colony Type'] = df_BMP['Experimental Condition'].apply(lambda s: s.split(' BMP4')[0])

    df_BMP['Condition order for plots']=df_BMP['Experimental Condition'].apply(lambda x: 'a.2D PLF EMT' if '2D PLF' in x else 'b.2D EMT' if '2D colony' in x else 'c.3D EMT')
    df_BMP['Treatment order for plots']=df_BMP['Treatment'].apply(lambda x: 'b.LDN 0.1uM' if '0.1' in x else 'c.cLDN 0.5uM' if '0.5' in x else 'a.Control')

    df_BMP = df_BMP.sort_values(by=['Condition order for plots', 'Treatment order for plots'])

    for col, df_col in df_BMP.groupby('Colony Type'):
        fig_mig = px.box(df_col, x='Treatment', y='Migration Onset Time (Manual First Cell Detection)', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white', range_y=(25,65), range_x=(-0.5,2.5), width=800, height=600)
        fig_mig.update_layout(yaxis_title='Migration Onset Time (Manual First Cell Detection)',font=dict(size=18))
        fig_mig.update_layout(showlegend=False)

        col_type = col.replace(' ','-')
        # Used_for information - 
        # these figures are labeled as - Extended Data Fig. 2
        # data ids for 2D PLF EMT cases should be labeled as - Extended Data Fig. 2C
        # data ids for 2D EMT cases should be labeled as - Extended Data Fig. 2D
        # data ids for 3D EMT cases should be labeled as - Extended Data Fig. 2E
        fig_mig.write_image(fr'{figs_dir}/BMP/BMP_inhibitor_migration_timing_for_{col_type}.{out_type}', scale=2 )


def _normalize_to_T0_mean_by_round_and_condiiton(group: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function for making immunolabeling heatmap
    Normalize each to time 0 mean intensity (for that condition and round)
    loop through each round and condition
    """
    group["Mean Intensity"] = group["Mean Intensity"].astype(float)
    for round in group["Round"].unique():
        sub_group = group[(group["Round"] == round)]
        group.loc[sub_group.index, "Mean Intensity"] = sub_group["Mean Intensity"] / sub_group[sub_group["Time (h)"] == 0]["Mean Intensity"].mean()
    return group


def _average_across_time(group: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function for making immunolabeling heatmap
    Average all the normalized intensities across the time-point
    """
    return group.groupby(["Label", "Condition", "Time (h)"], as_index=False, sort=False).agg({"Mean Intensity": "mean"})


def _normalize_to_100(group: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function for making immunolabeling heatmap
    Normalize the intensity values over time for a given Label from 0-100%
    """
    group["Mean Intensity"] = (group["Mean Intensity"] - group["Mean Intensity"].min()) / (group["Mean Intensity"].max() - group["Mean Intensity"].min()) * 100
    return group


def _sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function for making immunolabeling heatmap
    Sort dataframe to custom order of labels and conditions
    """

    rename_map = {
        'E-cadherin (Rabbit host)': 'E-cadherin',
        'N-cadherin (Mouse host)': 'N-cadherin',
        'Brachyury (Rabbit host)': 'TBXT',
        'Eomes (Mouse host)': 'Eomes',
        'Snail (Mouse host)': 'Snail',
        'Twist1 (Rabbit host)': 'Twist1',
        'Vimentin (Chicken host)': 'Vimentin',
        'H3K36me2 (Rabbit host)': 'H3K36me2',
    }

    custom_label_order = [
        "E-cadherin",
        "N-cadherin",
        "TBXT",
        "Eomes",
        "Snail",
        "Twist1",
        "Vimentin",
        "H3K36me2",
    ]

    custom_condition_order = [
        "2D PLF colony EMT",
        "2D colony EMT",
        "3D lumenoid EMT"
    ]

    # Use map to rename labels
    df["Label"] = df["Label"].map(rename_map).fillna(df["Label"])

    # Sort label order
    df["Label"] = pd.Categorical(df["Label"], categories=custom_label_order, ordered=True)

    # Rename conditions and sort condition order
    df["Condition"] = pd.Categorical(df["Condition"], categories=custom_condition_order, ordered=True)
    
    return df.sort_values("Label")


def _create_heatmap(data: pd.DataFrame, title: str, figs_dir: str, output_type: str) -> None:
    """
    Helper function for making immunolabeling heatmap
    Create heatmap of normalized intensities for each time for each Label.

    Each colored box is 1.0084 cm (W) x 0.2282 cm (H).
    Groups of 3 conditions are separated by a 2 pt white gap.
    Only tick marks on the outside edges (0.5 pt), no text labels.
    """
    import matplotlib as mpl

    cm_to_inch = 1 / 2.54
    pt_to_inch = 1 / 72.0

    # Box dimensions
    box_w_cm = 1.0084
    box_h_cm = 0.2282
    gap_pt = 2.0  # white space between groups in pt
    gap_inch = gap_pt * pt_to_inch

    data['Time (h)'] = data['Time (h)'].astype(int)
    pivot_table = data.pivot_table(
        index=["Label", "Condition"], columns="Time (h)", values="Mean Intensity"
    )

    n_rows = len(pivot_table)
    n_cols = len(pivot_table.columns)

    # Number of label groups and gaps
    group_sizes = pivot_table.groupby(level="Label").size()
    n_groups = len(group_sizes)
    n_gaps = n_groups - 1

    # Total plot area dimensions
    plot_w_inch = n_cols * box_w_cm * cm_to_inch
    plot_h_inch = n_rows * box_h_cm * cm_to_inch + n_gaps * gap_inch

    # Small margin for tick marks only (no labels)
    margin = 0.15  # inches
    fig_w = plot_w_inch + 2 * margin
    fig_h = plot_h_inch + 2 * margin

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=margin / fig_w,
        right=1 - margin / fig_w,
        bottom=margin / fig_h,
        top=1 - margin / fig_h,
    )

    # Build the heatmap data array with gap rows inserted between groups
    # We'll use imshow for precise control over row heights
    arr = pivot_table.values.astype(float)

    # Insert NaN rows for gaps between groups
    gap_row_count = max(1, int(round(gap_inch / (box_h_cm * cm_to_inch))))
    rows_with_gaps = []
    cumulative = 0
    gap_positions = []  # track where gaps are for tick placement
    for i, size in enumerate(group_sizes):
        block = arr[cumulative:cumulative + size]
        rows_with_gaps.append(block)
        cumulative += size
        if i < n_groups - 1:
            gap_block = np.full((gap_row_count, n_cols), np.nan)
            rows_with_gaps.append(gap_block)
            gap_positions.append(len(np.vstack(rows_with_gaps)))

    arr_gapped = np.vstack(rows_with_gaps)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad('white')  # gaps show as white

    ax.imshow(
        arr_gapped,
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        vmin=np.nanmin(arr_gapped),
        vmax=np.nanmax(arr_gapped),
    )

    # Remove all labels and titles
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Tick marks on the outside, 0.5 pt width
    tick_lw = 0.5 * pt_to_inch * 72
    # X ticks: at each column boundary
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=False)
    ax.set_xticks([], minor=True)
    # Y ticks: at each row boundary (excluding gap rows)
    ax.set_yticks(np.arange(-0.5, arr_gapped.shape[0], 1), minor=False)
    ax.set_yticks([], minor=True)

    ax.tick_params(
        axis='both', which='major',
        length=3, width=tick_lw,
        direction='out',
        bottom=True, top=False, left=True, right=False,
        labelbottom=False, labelleft=False,
    )

    # Spine width 0.5 pt
    for spine in ax.spines.values():
        spine.set_linewidth(tick_lw)

    # Save
    plt.savefig(f"{figs_dir}/{title}.{output_type}", dpi=600, transparent=True)
    plt.close(fig)


def plot_immunolabeling_heatmap(df: pd.DataFrame, figs_dir: str, output_type: str) -> None:
    """
    Function to plot immunolabeling heatmap from a dataset

    For each label and condition this workflow n
    1. Normalizes the data for each round by the mean T=0 intensity value,
    2. Averages the normalized intensities at each time points
    3. Applies min-max normalization to report intensities over time in 0-100 range
    It then sorts this data by custom label and condition orders, and creates a heatmap
    of the final normalized intensities for each time point for each condition and label.

    Parameters:
    ----------
    df: pd.DataFrame
        DataFrame containing the immunolabeling data
    figs_dir : str
        Directory where the figures will be saved
    output_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """

    # Set up dataset 
    df = create_df_IF(df)

    # Normalize each to time 0 mean intensity (for that condition and round)
    df_normalized = df.groupby(["Label", "Condition"], sort=False).apply(_normalize_to_T0_mean_by_round_and_condiiton).reset_index(drop=True)

    # Average all the normalized intensities across the time-point
    df_averaged = df_normalized.groupby(["Label", "Condition"], sort=False).apply(_average_across_time).reset_index(drop=True)

    # Normalize each to 0-100% for easier comparison across Labels
    df_final = df_averaged.groupby(["Label", "Condition"], sort=False).apply(_normalize_to_100).reset_index(drop=True)

    # Sorting label and condiiton order
    df_sort = _sort_dataframe(df_final)
    
    # Used_for information - 
    # these figures are labeled as - Fig. 4C
    # Create heatmap of the final intensities for each time for each Label
    _create_heatmap(df_sort, title="immuno_heatmap", figs_dir=figs_dir, output_type=output_type)


def immunlabeling_mean_intensity_analysis(df, FIGS_DIR, OUT_TYPE):
    """
    Generates plots of mean intensity of immunolabeling for different genes across conditions and rounds.

    Parameters:
    -----------
    FIGS_DIR : str
        Directory where the figures will be saved
    OUT_TYPE : str
        File type for the output figures (e.g. 'svg', 'png')
    """

    # Load and set up data
    df_summary = create_df_IF(df)
    Path(f"{FIGS_DIR}/Immunostaining").mkdir(exist_ok=True, parents=True)

    # Round colors: gold/dark orange, royal blue, violet
    colors = {
        'First Set Of Immunostaining': '#DAA520',   # Goldenrod (dark orange/gold)
        'Second Set Of Immunostaining': '#4169E1',  # Royal Blue
        'Third Set Of Immunostaining': '#8A2BE2'    # Blue Violet
    }

    # Target physical dimensions (cm) and scaling strategy:
    # Creating the figure at this tiny size makes markers/fonts blow out.
    # Instead, work at scale*target size; all proportions are preserved.
    # In Illustrator: select all -> scale to target cm (constrain proportions).
    # Text is kept as editable text (not outlines) so fonts can be changed
    # to Arial in Illustrator if needed.
    import matplotlib as mpl
    mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as editable text in SVG
    fig_width_cm = 2.1566
    fig_height_cm = 2.0382
    cm_to_inch = 1 / 2.54
    scale = 4
    fig_width = fig_width_cm * cm_to_inch * scale
    fig_height = fig_height_cm * cm_to_inch * scale
    save_dpi = int(600 / scale)  # 150 DPI -> same pixels as target @ 600 DPI

    # Create individual plots of mean immunolabel intensity for different genes for each round and condition
    for gene, df_gene in df_summary.groupby('Label'):
        min_start = {rnd:df_rnd[df_rnd['Time (h)']==0]['Mean Intensity'].mean() for rnd, df_rnd in df_gene[df_gene['Condition']=='2D PLF colony EMT'].groupby('Round')}
        fold = max([i/min_start[rnd] for rnd, df_rnd in df_gene.groupby('Round') for i in df_rnd['Mean Intensity'].values])

        n_rnds = len(df_gene['Round'].unique())
        w = 2.5/(n_rnds-1) if n_rnds>1 else 3
        offsets = {rnd:(i-1)/n_rnds for i, rnd in enumerate(df_gene['Round'].unique())}

        legend_handles = []
        for cond, df_cond in df_gene.groupby('Condition'):

            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
            max_fold = 0
            for rnd, df_rnd in df_cond.groupby('Round'):
                ax.set_title(gene + ' - ' + cond, fontsize=5, fontfamily='Nimbus Sans', fontweight='bold')

                max_fold = max([max_fold, fold])
                ax.set_ylim([0,max_fold+0.5])

                ints = [i/df_rnd[df_rnd['Time (h)']==0]['Mean Intensity'].mean() for i in df_rnd['Mean Intensity'].values]
                ts = [t + offsets[rnd]*w for t in df_rnd['Time (h)'].values]
                round_labels = {
                    'First Set Of Immunostaining': 'Round 1',
                    'Second Set Of Immunostaining': 'Round 2',
                    'Third Set Of Immunostaining': 'Round 3'
                }
                plt.scatter(ts, ints, s=7, c=colors[rnd], marker='D', label=round_labels.get(rnd, rnd))

                data = {}
                for t, i in zip(ts,ints):
                    if t not in data.keys():
                        data[t] = []
                    data[t].append(i)

                vplot = ax.violinplot(
                    dataset = list(data.values()),
                    positions = list(data.keys()),
                    widths = w,
                    showextrema=False
                )

                for patch in vplot['bodies']:
                    patch.set_color(colors[rnd])

                if len(legend_handles) < n_rnds:
                    legend_handles.append(vplot)

            # Y-axis: Arial bold 5pt (scaled), title "Mean Intensity (Fold)"
            ax.set_ylabel('Mean Intensity (Fold)', fontsize=5, fontfamily='Nimbus Sans', fontweight='bold')
            ax.tick_params(axis='y', labelsize=5)
            for label in ax.get_yticklabels():
                label.set_fontfamily('Nimbus Sans')
                label.set_fontweight('bold')

            # X-axis: no numbers, no title
            # Ticks at every 4h, bold (longer) ticks at every 8h
            max_time = int(df_gene['Time (h)'].max())
            minor_ticks = [t for t in range(0, max_time + 1, 4) if t % 8 != 0]
            major_ticks = list(range(0, max_time + 1, 8))
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_xticklabels([])  # No numbers on x-axis
            ax.set_xlabel('')       # No x-axis title
            ax.tick_params(axis='x', which='major', length=4*scale, width=1.0*scale)
            ax.tick_params(axis='x', which='minor', length=2*scale, width=0.5*scale)

            ax.legend(fontsize=4)
            plt.tight_layout()
            # Used_for information -
            # these figures are labeled as - Extended Data Fig. 3
            # data ids with label "TBXT" cases should be labeled as - Extended Data Fig. 3B
            # data ids with label "E-cadherin", "N-cadherin", "Eomes", "Snail", "Twist1", "Vimentin", "H3K36me2" cases should be labeled as - Extended Data Fig. 3C
            plt.savefig(f"{FIGS_DIR}/Immunostaining/{gene} - {cond}.{OUT_TYPE}", dpi=save_dpi)
            plt.close(fig)


# Run all analyses if this script is run
if __name__ == '__main__':
    run_all_analyses()

