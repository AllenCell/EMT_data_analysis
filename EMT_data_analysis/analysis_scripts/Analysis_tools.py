import warnings
import numpy as np
import math
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from EMT_data_analysis.tools import io, const
from EMT_data_analysis.analysis_scripts import plot_tools
from pathlib import Path
import scikit_posthocs as sp
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

# Set font to be Arial and configure text in figures to be editable in Adobe Illustrator
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"

warnings.filterwarnings("ignore")

# TODO  I assume this will live somewhere else eventually but placed this here for now
# to test that the functions all run
def run_all_analyses():
    """
    Run all analysis functions
    """

    DATA_PATH = '/allen/aics/users/filip.sluzewski/Public_Repos/emt-data-analysis/resubmission_scripts/Complete EMT Data - Segmentation Data.csv'
    BMP_DATA_PATH ='/allen/aics/emt/qc_and_scoring/Dataset making/July/July 24/Leica files with path to bad omezarr July 24 2025.csv'
    BMP_MIG_DATA_PATH = '/allen/aics/users/filip.sluzewski/Public_Repos/emt-data-analysis/resubmission_scripts/GE00006359_FINAL_BMP_Inhibitor_Scores_update_1.csv'
    FIGS_DIR = '/allen/aics/emt/data_analysis_plots/Colony_Metrics/full_dataset_figures/'
    OUT_TYPE = 'svg'

    df, df_bmp = load_and_prep_datasets(
        data_path=DATA_PATH,
        bmp_data_path = BMP_DATA_PATH, 
        bmp_mig_data_path = BMP_MIG_DATA_PATH,
        figs_dir=FIGS_DIR)
    
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
    plot_bmp_inhibitor_migration(df_bmp, FIGS_DIR, OUT_TYPE)
    plot_zo1_heatmaps(df, FIGS_DIR, OUT_TYPE)
    # plot_immunolabeling_heatmap(FIGS_DIR, OUT_TYPE)  # need data added for this


def load_and_prep_datasets(
        data_path, bmp_data_path, bmp_mig_data_path, figs_dir):

    # figs_dir = io.setup_base_directory_name("figures")
    # df = io.load_image_analysis_extracted_features(load_from_aws=True)

    df = pd.read_csv(data_path, index_col=None)

    # Create the directory for figures if it does not exist
    Path(figs_dir).mkdir(parents=True, exist_ok=True)

    # check if/why we have nan values here
    df['Experimental Condition'].replace('',np.nan, inplace=True)
    df.dropna(subset=['Experimental Condition'], inplace=True)

    df['Experimental Condition'] = df['Experimental Condition'].apply(lambda x: x.replace('2D MG EMT 1:60 MG','2D colony EMT').replace('2D PLF EMT 1:60 MG', '2D PLF colony EMT').replace('3D MG EMT 1:60 MG', '3D lumenoid EMT'))

    # drop EOMES|TBR2 data
    df = df[(df['Gene']!='EOMES|TBR2')&(df['Gene']!='TBR2|EOMES')]
    df['Gene']=df['Gene'].apply(lambda x: 'H2B' if 'H2B' in x else x)

    df.rename(columns={'Inflection Point':'Migration Time (h)', 'Inflection Point InOut':'Migration Time InOut (h)'}, inplace=True)

    # n_filtered_movies=df_f['Movie ID'].nunique()
    # print(f'No. of movies for analysis post filtering ={n_filtered_movies} ')

    df_bmp = pd.read_csv(bmp_data_path, index_col=None)
    df_bmp_mig = pd.read_csv(bmp_mig_data_path, index_col=None)
    
    df_bmp_mig.rename(columns={'Plate_barcode': 'Plate Barcode', 'Well_label': 'Well Label', 'Average_onset_of_migration':'Average Migration Onset (h)'}, inplace=True)

    df_bmp = pd.merge(df_bmp, df_bmp_mig, how='inner', on=['Plate Barcode', 'Well Label'])
    df_bmp = df_bmp[['Plate Barcode', 'Well Label', 'Experimental Condition', 'Average Migration Onset (h)']]
    df_bmp.replace('NM',np.nan, inplace=True)
    df_bmp['Average Migration Onset (h)'] = df_bmp['Average Migration Onset (h)'].apply(lambda x: float(x))

    return df, df_bmp


def create_df_f(df, time_interval=30):
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
    # print('Generating plots for Area at the glass for all three conditions and corresponding migration time estimated from the inflection of area at glass over time (Fig.5 C, D , E)')

    # Set up dataset
    df_f = create_df_f(df)
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

    Path(rf'{figs_dir}/Individual_Examples').mkdir(exist_ok=True, parents=True)
    plot_tools.plot_examples(
        df_int = df_f,
        id_plf = const.EXAMPLE_PLF,
        id_2d = const.EXAMPLE_2D,
        id_3d = const.EXAMPLE_3D,
        gene = "H2B",
        metric = 'Migration Time (h)',
        variable = 'Area at the glass(square micrometer)',
        figs_dir = figs_dir+'/Individual_Examples',
        out_type=out_type)


def plot_area_at_glass_h2b(df, figs_dir, out_type):
    """
    Generates plots for area at the glass for H2B gene and corresponding migration time estimated from the inflection of area at glass over time
    
    Parameters:
    -------
    df : pd.DataFrame
        Dataframe containing the area at the glass for each movie and timepoint for H2B data
    figs_dir : str
        Directory where the figures will be saved
    out_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """

    # Set up dataset
    df_f = create_df_f(df)
    df_a_h2b = df_f[df_f['Gene']=='H2B'].groupby(['Condition order for plots','Gene','Movie ID','Timepoint (h)']).agg({'Area at the glass(square micrometer)':'first', 'Migration Time (h)':'first'}).reset_index()
    df_a = df_f.groupby(['Condition order for plots','Gene','Movie ID','Timepoint (h)']).agg({'Area at the glass(square micrometer)':'first', 'Migration Time (h)':'first'}).reset_index()
    n_a = df_a['Movie ID'].nunique()
    fig, ax = plt.subplots(1,1)

    # for scn, df_scn in df_a[df_a['Gene']=='TBXT'].groupby('Movie ID'):
    sns.lineplot(df_a_h2b, x='Timepoint (h)', y='Area at the glass(square micrometer)', hue='Condition order for plots', palette=const.COLOR_MAP, errorbar=('pi', 50), estimator=np.median)
    plt.ylabel('Colony area over bottom 2 Z ( $\ um^2$)', fontsize=14)
    plt.xlabel('Time (hr)', fontsize=14)
    plt.ylim(0,170000)
    plt.rcParams.update({'font.size':14})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left') 
    plt.savefig(rf'{figs_dir}/Area_at_the_glass_over_time_MIP_H2B_n{n_a}.{out_type}', transparent=True, dpi=600)

    plot_tools.plot_examples(
        df_int = df_a,
        id_plf = const.EXAMPLE_PLF,
        id_2d = const.EXAMPLE_2D,
        id_3d = const.EXAMPLE_3D,
        gene = "Example",
        metric = 'Migration Time (h)',
        variable = 'Area at the glass(square micrometer)',
        figs_dir = figs_dir)


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
    df_summary = df_f.groupby('Movie ID').agg('first').reset_index()

    n_m = df_summary['Movie ID'].nunique()
    df_summary = df_summary.sort_values(['Gene','Condition order for plots'])
    fig_mig = px.box(df_summary, x='Condition order for plots', y='Migration Time (h)', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(15,35), width=800, height=600)
    fig_mig.update_layout(yaxis_title='Migration Time (h)',font=dict(size=18))
    fig_mig.write_image(rf'{figs_dir}/Migration_box_plot_n{n_m}.{out_type}', scale=2 )


def plot_migration_timing_h2b(df, figs_dir, out_type):
    """
    Generate box plots for migration timing for H2B gene and corresponding migration time estimated from the inflection of area at glass over time
    
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
    df_summary = df_f.groupby('Movie ID').agg('first').reset_index()
    n_m = df_summary[df_summary['Gene']=='H2B']['Movie ID'].nunique()
    df_summary = df_summary.sort_values(['Gene','Condition order for plots'])
    df_summary = df_summary.sort_values(by='Condition order for plots')
    
    fig_mig = px.box(df_summary[df_summary['Gene']=='H2B'], x='Condition order for plots', y='Migration Time (h)', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(15,35), width=800, height=600)
    fig_mig.update_layout(yaxis_title='Migration Time (h)',font=dict(size=18))
    fig_mig.write_image(rf'{figs_dir}/Migration_box_plot_H2B_n{n_m}.{out_type}', scale=2 )

    print('\n\n\n...statitsitcal analysis of overall migriation timing between the conditions for H2B...')
    x_mig = df_summary['Migration Time (h)'][['2D PLF' in val for val in df_summary['Experimental Condition'].values]].dropna()
    y_mig = df_summary['Migration Time (h)'][['2D colony EMT' in val for val in df_summary['Experimental Condition'].values]].dropna()
    z_mig = df_summary['Migration Time (h)'][['3D lumenoid EMT' in val for val in df_summary['Experimental Condition'].values]].dropna()

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
    df_summary = df_f.groupby('Movie ID').agg('first').reset_index()
    n_m = df_summary[df_summary['Gene']=='H2B']['Movie ID'].nunique()
    df_summary = df_summary.sort_values(['Gene','Condition order for plots'])
    df_summary = df_summary.sort_values(by='Condition order for plots')
    
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

    # Time of max EOMES expression (h) examples
    plot_tools.plot_examples(
        df_int = df_int,
        id_plf = const.EOMES_PLF,
        id_2d = const.EOMES_2D,
        id_3d = const.EOMES_3D,
        gene = "EOMES",
        figs_dir = figs_dir+'/Individual_Examples',
        metric='Time of max EOMES expression (h)',
        out_type=out_type)

    # Time of max TBXT expression (h) examples
    plot_tools.plot_examples(
        df_int = df_int,
        id_plf = const.TBXT_PLF,
        id_2d = const.TBXT_2D,
        id_3d = const.TBXT_3D,
        gene = "TBXT",
        figs_dir = figs_dir+'/Individual_Examples',
        metric='Time of max TBXT expression (h)',
        out_type=out_type)

    # Time of inflection of E-cad expression (h) examples-
    # import pdb; pdb.set_trace()
    plot_tools.plot_examples(
        df_int = df_int,
        id_plf = const.CDH_PLF,
        id_2d = const.CDH_2D,
        id_3d = const.CDH_3D,
        gene = "CDH1",
        figs_dir = figs_dir+'/Individual_Examples',
        metric='Time of inflection of E-cad expression (h)',
        out_type=out_type)

    # Time of inflection of SOX expression (h) examples-
    plot_tools.plot_examples(
        df_int = df_int,
        id_plf = const.SOX_PLF,
        id_2d = const.SOX_2D,
        id_3d = const.SOX_3D,
        gene = "SOX2",
        figs_dir = figs_dir+'/Individual_Examples',
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
    df_summary = df_f.groupby('Movie ID').agg('first').reset_index()
    df_summary = df_summary.sort_values(['Gene','Condition order for plots'])
    df_summary = df_summary.sort_values(by='Condition order for plots')
    df_summary['gene_m'] = pd.Categorical(df_summary['Gene'], df_summary['Gene'].unique())
    df_summary = df_summary.sort_values(['gene_m','Condition order for plots'])

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

    # Plotting of expression timing relative to EMT induction as independent graphs per-gene
    for g, df_g in df_comb.groupby('Gene'):
        fig_difference = px.box(df_g, y='gene_metric', x='Condition order for plots', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white',range_y=(10,50),width=600, height=600)
        fig_difference.update_layout(showlegend=False)

        fig_difference.update_layout(xaxis_title='Cell lines', yaxis_title='Time of expression change (h)', font=dict(size=18))
        fig_difference.update_layout(boxgroupgap=0.5, boxgap=0.25)
        fig_difference.write_image(rf'{figs_dir}/Timing_of_expression_change_relative_to_EMT_induction_T_0_for_{g}_FigS5a.{out_type}', scale=2 )


    # Plotting scatter plots of gene metrics vs migration time
    metric_dict = {
        'EOMES':'Time of max EOMES expression (h)',
        'TBXT':'Time of max TBXT expression (h)',
        'CDH1':'Time of inflection of E-cad expression (h)',
        'SOX2':'Time of half-maximal SOX2 expression (h)'
    }
    for g, df_g in df_comb.groupby('Gene'):
        fig_scatter, ax = plt.subplots(1,1, figsize=(10,10))
        fig_scatter = sns.scatterplot(df_g, x='gene_metric', y='Migration Time (h)', hue='Condition order for plots', palette=const.COLOR_MAP, s=100, alpha=0.7, linewidth=2, edgecolor='coral', legend=False)
        plt.xlim(10,50)
        plt.ylim(10,50)

        plt.title(f'{g}\n{metric_dict[g]} vs Migration Time (h)')
        plt.xlabel(metric_dict[g], fontsize=16)
        plt.ylabel('Migration Time (h)', fontsize=16)
        plt.rcParams.update({'font.size':16})
        plt.savefig(fr'{figs_dir}/Scatter_plot_between_{g}_metric_and_migration_time.{out_type}', dpi=600)


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
            migration = df_g['Migration Time (h)'][[cond in val for val in df_g['Experimental Condition'].values]].dropna()
            metric = df_g['gene_metric'][[cond in val for val in df_g['Experimental Condition'].values]].dropna()
        
            pearson, p_pvalue = pearsonr(migration, metric)
            spearman, s_pvalue = spearmanr(migration, metric)
            print(f'Pearson Correlation: {pearson:.3g} | p-value: {p_pvalue:.3g}')
            print(f'Spearman Correlation: {spearman:.3g} | p-value: {s_pvalue:.3g}')

            X = df_g['Migration Time (h)'][[cond in val for val in df_g['Experimental Condition'].values]].dropna()
            Y = df_g['gene_metric'][[cond in val for val in df_g['Experimental Condition'].values]].dropna()

            # It's important to add a constant (intercept) to the model
            X = sm.add_constant(X)
            
            # Fit the Ordinary Least Squares (OLS) model
            model = sm.OLS(Y, X)
            results = model.fit()

            slope_p_value = results.pvalues['Migration Time (h)']
            r_squared = results.rsquared
            slope_coeff = results.params['Migration Time (h)']

            print(f"R-squared: {r_squared:.3g}")
            print(f"Slope (Coefficient for gene expression): {slope_coeff:.3g}")
            print(f"P-value for the slope: {slope_p_value:.3g}") # Using 'g' for scientific notation if needed

        print('\n\n-------Statistics for entire metric------------')
        migration = df_g['Migration Time (h)']
        metric = df_g['gene_metric']

        pearson, p_pvalue = pearsonr(migration, metric)
        spearman, s_pvalue = spearmanr(migration, metric)
        print(f'Pearson Correlation: {pearson:.3g} | p-value: {p_pvalue:.3g}')
        print(f'Spearman Correlation: {spearman:.3g} | p-value: {s_pvalue:.3g}')
        
        X = df_g['Migration Time (h)']
        Y = df_g['gene_metric']

        # It's important to add a constant (intercept) to the model
        X = sm.add_constant(X)

        # Fit the Ordinary Least Squares (OLS) model
        model = sm.OLS(Y, X)
        results = model.fit()

        slope_p_value = results.pvalues['Migration Time (h)']
        r_squared = results.rsquared
        slope_coeff = results.params['Migration Time (h)']

        print(f"R-squared: {r_squared:.3g}")
        print(f"Slope (Coefficient for gene expression): {slope_coeff:.3g}")
        print(f"P-value for the slope: {slope_p_value:.3g}") # Using 'g' for scientific notation if needed


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

    df_summary = df_coll.drop_duplicates(subset=['Movie ID'])
    df_summary['sort_value'] = df_summary['Drug Concentration'].apply(lambda c: float(c.split()[0]) if 'HBSS' not in c else -1)
    df_summary['Collagenease concentration (ug/mL)'] = [float(c.split(' ')[0])  if 'HBSS' not in c else 0 for c in df_summary['Drug Concentration'].values]
    df_summary = df_summary.sort_values(by=['sort_value'], axis=0)
    # df_summary['gene_m'] = pd.Categorical(df_summary['Gene'], df_summary['Gene'].unique())
    # df_summary = df_summary.sort_values(['gene_m','Drug Concentration'])


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
            print('{5}: Mean {0:.2f} | Median {1:.2f} | St.Dev {2:.2f} | Min: {3:.2f} | Max: {4:.2f}'.format(np.mean(v), np.median(v), np.std(v), np.min(v), np.max(v), c))

        print('Concentrations:')
        print(' '.join(['{0} n:{1} |'.format(t, len(v)) for t, v in zip(concentrations, migration)]))
        print('Significant difference between the distributions: applying post hoc with Holm-Bonferroni adjustment of p-value')    
        
        print(sp.posthoc_mannwhitney(migration, p_adjust = 'holm'))


        # Calculating statistics for downward trend in collagenase concentrations vs migration time

        X = df_gene['Collagenease concentration (ug/mL)']
        Y = df_gene['Migration Time (h)']

        fig_scatter, ax = plt.subplots(1,1, figsize=(10,10))
        fig_scatter = sns.scatterplot(df_gene, x='Collagenease concentration (ug/mL)', y='Migration Time (h)', hue='Drug Concentration', palette=color_map, s=100, alpha=0.7, linewidth=2, legend=False)
        plt.title(f'{g}\nCollagenase vs Migration Time (h)')
        plt.xlabel('Collagenease Concentration (ug/mL)', fontsize=16)
        plt.ylabel('Migration Time (h)', fontsize=16)
        plt.rcParams.update({'font.size':16})

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
        print(f"P-value for the slope: {slope_p_value:.3g}") # Using 'g' for scientific notation if needed

        xs = [df_gene['Collagenease concentration (ug/mL)'].min(), df_gene['Collagenease concentration (ug/mL)'].max()]
        ys = [x*slope_coeff+const_coef for x in xs]
        plt.plot(xs,ys,'--k')
        plt.savefig(fr'{figs_dir}/Collagenase/Scatter_plot_for_{g}_between_collagenase_conctertion_and_migration_time.{out_type}', dpi=600)


        alpha = 0.05
        if slope_p_value < alpha:
            print(f"\nConclusion: The p-value for the slope is less than {alpha}, indicating a statistically significant linear relationship between drug concentration and migration time.")
            print(f"On average, for each 1 ug/mL increase in drug concentration, the migration time changes by {slope_coeff:.2f} hours.")
        else:
            print("\nConclusion: The p-value for the slope is not less than 0.05, so we cannot conclude there is a significant linear relationship.")


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

    df_summary = df_coll.drop_duplicates(subset=['Movie ID'])
    df_summary.dropna(subset=['Time of migration first cell'],inplace=True)

    n_m = df_summary['Movie ID'].nunique()
    df_summary['sort_value'] = df_summary['Drug Concentration'].apply(lambda c: float(c.split()[0]) if 'MMPi' not in c else -1)
    df_summary['MMPi concentration (uM)'] = [float(c.split(' ')[0])  if 'MMPi' not in c else 0 for c in df_summary['Drug Concentration'].values]
    df_summary = df_summary.sort_values(by=['sort_value'], axis=0)

    # print('Generating Box plots for migration timing for each gene in the dataset collagenase')
    print('\n\n\n.......Statistical comparison for migration time with MMPi treatment:')
    for gene, df_gene in df_summary.groupby('Gene'):
        color_map={tgt:'orange' for tgt in df_gene['Drug Concentration'].unique()}
        fig_mig_g = px.box(df_gene, y='Time of migration first cell', x='Drug Concentration', color='Drug Concentration', color_discrete_map=color_map, points='all', template='simple_white',range_y=(10,40),width=800, height=600)
        fig_mig_g.update_layout(showlegend=False)
        fig_mig_g.update_layout(xaxis_title='Cell lines', yaxis_title='Migration in real time (h)', font=dict(size=18))
        fig_mig_g.update_traces(width=0.6)
        fig_mig_g.update_layout(boxgroupgap=0.4, boxgap=0.4)
        fig_mig_g.write_image(rf'{figs_dir}/MMPi/Migration_box_plot_{gene}_per_conditions.{out_type}', scale=2 )

        print('Gene: ', gene)
        key = {g:i for i, g in enumerate(df_gene['Drug Concentration'].unique())}
        concentrations = list(key.keys())
        migration = [[]]*len(concentrations)

        for g, d_g in df_gene.groupby('Drug Concentration'):
            migration[key[g]] = d_g['Time of migration first cell'].values

        for c, v in zip(concentrations, migration):
            print('{5}: Mean {0:.4f} | Median {1:.4f} | St.Dev {2:.4f} | Min: {3:.4f} | Max: {4:.4f}'.format(np.mean(v), np.median(v), np.std(v), np.min(v), np.max(v), c))

        print('Concentrations:')
        print(' '.join(['{0} n:{1} |'.format(t, len(v)) for t, v in zip(concentrations, migration)]))
        print('Significant difference between the distributions: applying post hoc with Holm-Bonferroni adjustment of p-value')    
        
        print(sp.posthoc_mannwhitney(migration, p_adjust = 'holm'))


        # Calculating statistics for downward trend in collagenase concentrations vs migration time

        X = df_gene['MMPi concentration (uM)']
        Y = df_gene['Time of migration first cell']

        fig_scatter, ax = plt.subplots(1,1, figsize=(10,10))
        fig_scatter = sns.scatterplot(df_gene, x='MMPi concentration (uM)', y='Time of migration first cell', hue='Drug Concentration', palette=color_map, s=100, alpha=0.7, linewidth=2, legend=False)
        plt.title(f'{gene}\nMMPi vs Migration Time (h)')
        plt.xlabel('MMPi Concentration (ug/mL)', fontsize=16)
        plt.ylabel('Time of Migration First Cell (h)', fontsize=16)
        plt.rcParams.update({'font.size':16})

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

        xs = [df_gene['MMPi concentration (uM)'].min(), df_gene['MMPi concentration (uM)'].max()]
        ys = [x*slope_coeff+const_coeff for x in xs]
        plt.plot(xs,ys,'--k')
        plt.savefig(fr'{figs_dir}/MMPi/Scatter_plot_for_{gene}_between_MMPi_conctertion_and_average_migration_time.{out_type}', dpi=600)


        print(f"R-squared: {r_squared:.4f}")
        print(f"Slope (Coefficient for concentration): {slope_coeff:.4f}")
        print(f"P-value for the slope: {slope_p_value:.4g}") # Using 'g' for scientific notation if needed

        alpha = 0.05
        if slope_p_value < alpha:
            print(f"\nConclusion: The p-value for the slope is less than {alpha}, indicating a statistically significant linear relationship between drug concentration and migration time.")
            print(f"On average, for each 1 uM increase in drug concentration, the migration time changes by {slope_coeff:.2f} hours.")
        else:
            print("\nConclusion: The p-value for the slope is not less than 0.05, so we cannot conclude there is a significant linear relationship.")


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

    print('Generating Box plots for migration timing for CLYBL gene knockdown experiment')
    df_summary['Condition order for plots']=df_summary['Knockdown'].apply(lambda x: 'a.Control' if 'Control' in x else 'b.Seq Scr' if 'seq' in x else f'c.{x}' if 'g1' in x else f'd.{x}' if 'g2' in x else f'e.{x}')
    df_summary = df_summary.sort_values(['Condition order for plots'])

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

    # print('Generating Heatmaps for ZO1 - Fig.7 and Fig. S6 ')
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
    plot_tools.Intensity_over_z(df_zo_examples, figs_dir=figs_dir+'/ZO1', out_type=out_type)


def _get_linear_relationship_from_covariance(x, y):
    """
    Use eigenvectors of covariance matrix to determine linear mapping between migration times from two methods.
    This method constructs the covariance matrix for the two sets of migration timing measurements and determines
    its eigenvectors and eigenvalues. The eigenvector with highest eigenvalue lies along the direction of
    greatest variation in the data. A line colinear with this vector is used to define a linear
    mapping between the two migration times. The eigenvalue associated with the second eigenvector,
    perpendicular to the linear mapping, defines the standard deviatin away from this line.

    Parameters
    ----------
    x : pd.Series
        Migration times measured by the first method
    y : pd.Series
        Migration times measured by the second method

    Returns
    -------
    slope, intercept : float
        Slope and intercept of the linear mapping between the two sets of migration times
    std: float
        Standard deviation of the migration times from the linear mapping
    """

    # Format live and fixed features as needed for analysis and calculated the mean of each
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    center = (float(mean_x), float(mean_y))

    # Get covaraince matrix of live and fixed data then calculate its eigenvectors and eigenvalues
    data = [[x_i, y_i] for x_i, y_i in zip(x, y)]
    covariance_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Get indices that would sort the eigenvalues in ascending order
    sorted_indices = eigenvalues.argsort()[::-1]

    # Sort eigenvalues and eigenvectors
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Get slope and intercept of line defined by first eigenvector
    slope = eigenvectors[1, 0] / eigenvectors[0, 0]
    intercept = mean_y - slope * mean_x
    # Get standard deviation perpendicular to the line defined by the first eigenvector
    std = np.sqrt(eigenvalues[1])

    return slope, intercept, std

def _get_perpendicular_distance(x_i, y_i, A, B, C):
  """
  Calculate the perpendicular distance from a point (x_i, y_i) to a line Ax+By+C=0.

  Args:
    x_i: x-coordinate of the point
    y_i: y-coordinate of the point
    A: coefficient of x in the line equation Ax+By+C=0
    B: coefficient of y in the line equation Ax+By+C=0
    C: constant term in the line equation Ax+By+C=0

  Returns:
    The perpendicular distance from the point to the line.
  """
  numerator = abs(A * x_i + B * y_i + C)
  denominator = math.sqrt(A**2 + B**2)
  distance = numerator / denominator
  return distance

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
        'Data ID',
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

    # Grouping the data and getting the fraction of 'True' values in 'Inside' column to get fraction of nuclei inside the basement membrane for each movie
    dfio_grouped=dfio_merge.groupby([
        'Condition order for plots',
        'Gene',
        'Movie ID',
        'Data ID',
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


    dfio_scatter=dfio_merge.groupby([
        'Condition order for plots',
        'Data ID',
    ]).agg({
        'Migration Time (h)':'first', 
        'Migration Time InOut (h)':'first'
    })

    # Plotting migration time estimated from inside and outside classification of nuclei w.r.t basement memebrane vs migration time estimated from area at the glass (Fig. 5I)
    fig_scatter, ax = plt.subplots(1,1, figsize=(10,10))
    fig_scatter = sns.scatterplot(dfio_scatter, x='Migration Time (h)', y='Migration Time InOut (h)', hue='Condition order for plots', palette=const.COLOR_MAP, s=100, alpha=0.7, linewidth=2, edgecolor='coral', legend=False)
    plt.xlim(20,36)
    plt.ylim(20,36)

    plt.xlabel('Migration Time from area at glass (h)', fontsize=16)
    plt.ylabel('Migration Time fraction of nuclei outside basement membrane (h)', fontsize=16)
    plt.rcParams.update({'font.size':16})
    plt.savefig(fr'{figs_dir}/Inside-Outside/Scatter_plot_between_computer_migration_area_on_glass_vs_inside_outside.{out_type}', dpi=600, transparent=True)

    # Plotting example to show how migration time is estimated from fraction of nuclei outside the basement membrane over time (Fig. 5H )
    df_io_id = dfio_grouped[dfio_grouped['Data ID']==const.EXAMPLE_IO_ID]
    fig,ax = plt.subplots(1,1,figsize=(8,6))

    x_io = df_io_id['Migration Time InOut (h)'].values[0]
    y_io = df_io_id['Fraction_outside'][df_io_id['Time hr']==x_io].values[0]
    ax.plot(df_io_id['Time hr'],df_io_id['Fraction_outside'], c='orange', linewidth=3)
    ax.scatter(x_io,y_io,c='black', marker='D', s=100) 
    plt.ylabel(f'Fraction of nuclei outside the lumen', fontsize=16)
    plt.xlabel('Time (hr)', fontsize=16)
    plt.xlim(left=10)
    plt.tight_layout()
    plt.savefig(fr'{figs_dir}/Individual_Examples/Example_migration_estimation_fraction_nuclei_outside_basement_membrane.{out_type}', dpi=600)

    print('\n\n\n.......Statistical comparison for migration time using Area-at-Mask vs Inside-Outside:')
    X = dfio_scatter['Migration Time (h)']
    Y = dfio_scatter['Migration Time InOut (h)']

    p_results = pearsonr(X.values, Y.values)
    r_results = spearmanr(X.values, Y.values)
    print('n: {0:d}'.format(n_movies_io))
    print('Pearson Correlation: {0:.3g} | p-Value: {1:.3g}'.format(p_results.statistic, p_results.pvalue))
    print('Spearman Correlation: {0:.3g} | p-Value: {1:.3g}'.format(r_results.statistic, r_results.pvalue))

    X = dfio_scatter['Migration Time (h)']
    Y = dfio_scatter['Migration Time InOut (h)']
    X = sm.add_constant(X)

    # Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(Y, X)
    results = model.fit()
    slope_p_value = results.pvalues['Migration Time (h)']
    r_squared = results.rsquared
    slope_coeff = results.params['Migration Time (h)']

    print(f"R-squared: {r_squared:.3g}")
    print(f"Slope (Coefficient for migration timing): {slope_coeff:.3g}")
    print(f"P-value for the slope: {slope_p_value:.3g}")

    # Using the covariance matrix to define a linear mapping between migration times from two methods
    X = dfio_scatter['Migration Time (h)']
    Y = dfio_scatter['Migration Time InOut (h)']
    slope, intercept, std = _get_linear_relationship_from_covariance(X, Y)
    print("\n--- Defining a linear relatioship from eigenvectors of covariance matrix ---")
    print(f"Slope: {slope:.3g}")
    print(f"Intercept: {intercept:.3g}")
    print(f"Standard Deviation: {std:.3g}")

    # Plotting migration time estimated from inside and outside classification of nuclei w.r.t basement memebrane vs migration time estimated from area at the glass (Fig. 5I)
    fig_scatter, ax = plt.subplots(1,1, figsize=(10,10))
    fig_scatter = sns.scatterplot(dfio_scatter, x='Migration Time (h)', y='Migration Time InOut (h)', hue='Condition order for plots', palette=const.COLOR_MAP, s=100, alpha=0.7, linewidth=2, edgecolor='coral', legend=False)
    fig_scatter = sns.lineplot(x=[16,36], y=[slope*16+intercept,slope*36+intercept], color='black', linestyle='-', label='Linear mapping', ax=ax)
    # draw unity line
    fig_scatter = sns.lineplot(x=[16, 36], y=[16, 36], color='gray', linestyle='--', label='Unity line', ax=ax)
    plt.xlim(16,36)
    plt.ylim(16,36)

    plt.xlabel('Migration Time from area at glass (h)', fontsize=16)
    plt.ylabel('Migration Time fraction of nuclei outside basement membrane (h)', fontsize=16)
    plt.rcParams.update({'font.size':16})
    plt.savefig(fr'{figs_dir}/Inside-Outside/Scatter_plot_between_computer_migration_area_on_glass_vs_inside_outside_with_linearmap.{out_type}', dpi=600, transparent=True)

    # get minimum x or y value
    sample_val = min(X.min(), Y.min())
    dist = _get_perpendicular_distance(sample_val, sample_val, slope, -1, intercept)
    print(f"Distance from linear mapping at ({sample_val},{sample_val}): {dist:.3g}")
    print(f"Fraction of a standard deviation away from the linear mapping at {sample_val},{sample_val}): {dist/std:.3g}")


def plot_bmp_inhibitor_migration(df_BMP, figs_dir: str, out_type):
    (Path(figs_dir) / 'BMP').mkdir(parents=True, exist_ok=True)

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
    df_BMP['Treatment order for plots']=df_BMP['Treatment'].apply(lambda x: 'b.BMP4 LDN 0.1uM' if '0.1' in x else 'c.BMP4 LDN 0.5uM' if '0.5' in x else 'a.BMP4')

    df_BMP = df_BMP.sort_values(by=['Condition order for plots', 'Treatment order for plots'])

    for col, df_col in df_BMP.groupby('Colony Type'):
        fig_mig = px.box(df_col, x='Treatment', y='Average Migration Onset (h)', color='Condition order for plots', color_discrete_map=const.COLOR_MAP, points='all', template='simple_white', range_y=(25,65), range_x=(-0.5,2.5), width=800, height=600)
        fig_mig.update_layout(yaxis_title='Average Migration Onset (h)',font=dict(size=18))
        fig_mig.update_layout(showlegend=False)

        col_type = col.replace(' ','-')
        fig_mig.write_image(fr'{figs_dir}/BMP/BMP_inhibitor_migration_timing_for_{col_type}.{out_type}', scale=2 )


def _normalize_to_T0_mean_by_round_and_condiiton(group: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function for making immunolabeling heatmap
    Normalize each to time 0 mean intensity (for that condition and round)
    loop through each round and condition
    """
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

    custom_label_order = [
        "E-cadherin",
        "N-cadherin",
        "TBXT",
        "Eomes",
        "Snail",
        "Twist1",
        "Vimentin",
        "H3Kme2",
    ]

    custom_condition_order = [
        "2D PLF Colony EMT",
        "2D Colony EMT",
        "3D Luminoid EMT"
    ]

    # Sort label order
    df["Label"] = pd.Categorical(df["Label"], categories=custom_label_order, ordered=True)
    
    # Rename conditions and sort condition order
    df.loc[df["Condition"] == "2D PLF", "Condition"] = "2D PLF Colony EMT"
    df.loc[df["Condition"] == "2D MG", "Condition"] = "2D Colony EMT"
    df.loc[df["Condition"] == "3D Lum", "Condition"] = "3D Luminoid EMT"
    df["Condition"] = pd.Categorical(df["Condition"], categories=custom_condition_order, ordered=True)
    
    return df.sort_values("Label")


def _create_heatmap(data: pd.DataFrame, title: str, figs_dir: str, output_type: str) -> None:
    """
    Helper function for making immunolabeling heatmap
    Create heatmap of normalized intensities for each time for each Label
    """
    FIGSIZE = (8, 6)
    MAJOR_FONT_SIZE = 12
    MINOR_FONT_SIZE = 10

    plt.clf()
    plt.figure(figsize=FIGSIZE)
    pivot_table = data.pivot_table(index=["Label", "Condition"], columns="Time (h)", values="Mean Intensity")
    cmap = plt.cm.viridis
    cmap.set_bad('lightgrey')
    clabel = "Normalized mean intensity (%)"
    
    # Create heatmap
    ax = sns.heatmap(
        pivot_table,
        xticklabels=True, yticklabels=True,
        cmap=cmap, cbar_kws={"label": clabel})
    
    # Set all font sizes
    plt.xticks(fontsize=MINOR_FONT_SIZE)
    plt.yticks(fontsize=MINOR_FONT_SIZE)
    plt.xlabel("Time (h)", fontsize=MAJOR_FONT_SIZE)
    plt.ylabel("Label and Condition", fontsize=MAJOR_FONT_SIZE)
    ax.figure.axes[-1].tick_params(labelsize=MINOR_FONT_SIZE)
    
    # Add horizontal lines between groups
    group_sizes = pivot_table.groupby(level="Label").size()
    line_positions = []
    cumulative_sum = 0
    for size in group_sizes:
        cumulative_sum += size
        line_positions.append(cumulative_sum)
    for pos in line_positions[:-1]:
        plt.axhline(y=pos, color='white', linewidth=2) 

    # Save figure in vector formats
    plt.tight_layout()
    plt.savefig(f"{figs_dir}/{title}", format=output_type)


def plot_immunolabeling_heatmap(figs_dir: str, output_type: str) -> None:
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

    figs_dir : str
        Directory where the figures will be saved
    output_type : str
        File type for the output figures (e.g. 'svg', 'png')
    """

    # Load dataset 
    # TODO: replace with loading and filtering broader dataset
    df = pd.read_csv("immuno_panel.csv")

    # Normalize each to time 0 mean intensity (for that condition and round)
    df_normalized = df.groupby(["Label", "Condition"], sort=False).apply(_normalize_to_T0_mean_by_round_and_condiiton).reset_index(drop=True)
    
    # Average all the normalized intensities across the time-point
    df_averaged = df_normalized.groupby(["Label", "Condition"], sort=False).apply(_average_across_time).reset_index(drop=True)
    
    # Normalize each to 0-100% for easier comparison across Labels
    df_final = df_averaged.groupby(["Label", "Condition"], sort=False).apply(_normalize_to_100).reset_index(drop=True)

    # Sorting label and condiiton order
    df_sort = _sort_dataframe(df_final)
    
    # Create heatmap of the final intensities for each time for each Label
    _create_heatmap(df_sort, title="immuno_heatmap", figs_dir=figs_dir, output_type=output_type)

# TODO  I assume this will live somewhere else eventually but placed this here for now
# to test that the functions all run
run_all_analyses()

