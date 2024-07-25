## Import packages:
## For numerical work
import numpy as np
## For working with filepaths and directories
from pathlib import Path
## For working with DataFrames
import pandas as pd
pd.options.mode.copy_on_write = True
from sklearn import preprocessing
## For plotting
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
## For statistical testing
from scipy.stats import f_oneway, tukey_hsd

PLOT_DATA = True
MAKE_CONTACT_SHEETS = False



def stringified_floatlist_to_floatlist(ls):
    """Converts a list that is saved as a string back to a list object."""
    # test = '[1,2,3,4,5,6]'
    # test.strip('[]')
    strfloats = ls.strip('[]')
    strfloats = ls.strip('()')
    return [float(x) for x in strfloats.split(',') if strfloats]


def plots_to_multipdf(fname, tight=False, dpi=80):
    """ This function saves a multi-page PDF of all open figures. """
    if tight == True:
        bbox = 'tight'
    else:
        bbox = None
    pdf = PdfPages(fname)
    figs = [plt.figure(f) for f in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pdf, format='pdf', bbox_inches=bbox, dpi=dpi)
    pdf.close()


def mkcontsh(out_fpath, df, x_label, y_label, group_label, hue_label=None, hue_palette=None, ncols=8, panel_width=2.5, panel_height=2, vline_lab=None, hline_lab=None, figtitle=None, legend=True, twinx=False, dpi=300):
    """Makes a contact sheet given a DataFrame. \
    Effectively the same as plotting lineplots for every group in a \
    DataFrame.groupby object. \
    out_fpath is passed to fig.savefig and is the location where the plot \
    will be saved (including its filename). \
    df is the DataFrame containing column names x_label, y_label, and group_label. \
    The x_label is a column name entered as a string. \
    y-labels should either be a string like x_label or a list of column names \
    in the dataframe that you want to plot against x_label (can be a list of \
    length 1 or more). \
    The group_label is a column name entered as a string. \
    The twinx argument, if set to True, will plot the first y_label on the left
    y-axis, and then every subsequent y-label on the right y-axis. It requires \
    more than one y_label to be passed. \
    Required packages: \
    pandas as pd, \
    matplotlib.pyplot as plt, \
    matplotlib.lines as mlines, \
    seaborn."""

    ## DESIRABLE FEATURES
    ## - add options for **kwargs to give user more control over plot parameters like the size of each ax or linestyles
    ## - can you make it so you can pass other things to plot into the function while still keeping it small and general?
    ##      - for example, what if I want to produce a contact sheet where each plot has ax.axvline(val) for a val that is specific to each plot?

    ## convert a string input of y_label to a list so that multiple y-labels
    ## become an option
    y_label = [y_label] if type(y_label)==str else y_label
    y_label_indices = range(len(y_label))

    ## 'Figure' out the number of panels you have to plot
    nax = len([nm for nm,grp in df.groupby(group_label)])

    ## Get the number of rows needed based on how many columns you want
    nrows = nax // ncols + (1 if nax % ncols else 0)

    ## Make a figure with your desired number of columns and rows
    fig, axs = plt.subplots(figsize=(panel_width*ncols, panel_height*nrows), nrows=nrows, ncols=ncols)

    ## Iterate through your grouped dataframes...
    ## (get the row, column index for each ax in axs as a flat list)
    axi = [(r,c) for r in range(nrows) for c in range(ncols)]
    ## (start axes indices at 0 and put a plot in each ax)
    if hue_label:
        if hue_palette:
            if isinstance(hue_palette, dict):
                # hue_palette[df[hue_label]]
                hue = {x:hue_palette[x] for x in hue_palette if x in df[hue_label].unique()}
            else:
                hue = dict(zip(df[hue_label].unique(), hue_palette))
        else:
            hue = dict(zip(df[hue_label].unique(), sns.color_palette(as_cmap=True)))
        fig_cpal = [[hue, 'black', 'grey'] for i in range(nax)]
        fig_clab = [y_label for i in range(nax)]
        hue_inv = {hue[k]:k for k in hue}
    else:
        fig_cpal = [sns.color_palette(as_cmap=True) for i in range(nax)]
        fig_clab = [y_label for i in range(nax)]

    i = 0
    for nm, grp in df.groupby(group_label):
        # break
        ax = axs[axi[i]]
        ax.set_title(nm, fontsize=5)

        ## Assign the colorpalette to a variable in legends or twinx are desired
        if hue_label:
            try:
                c = grp[hue_label].unique()
                assert(len(c)==1)
                fig_cpal[i][0] = hue[str(*c)]
            except AssertionError:
                print('If "plot_color_label" is not None, then each group in "group_label" must have only 1 value for "plot_color_label".')
        else:
            pass

        ## ...and plot the desired information
        ## this chunk below is incase you want to use seaborn
        ## NOTE it currently returns a future warning about use_inf_as_na option
        ## being deprecated, however all vlaues in my dataframe are finite, but
        ## to avoid importing another package (ie. warnings) we will just live 
        ## with the many FutureWarnings.

        if len(y_label) > 1 and twinx:
            ## Change the color of the xticks and xticklabels on the left side
            ax.tick_params(axis='y', colors=fig_cpal[i][0])

            ## Create a second axes object with a y-axis on the right side
            axb = ax.twinx()

            ## Plot data for both left and right axes
            sns.lineplot(data=grp, x=x_label, y=y_label[0], color=fig_cpal[i][0], ax=ax)
            [sns.lineplot(data=grp, x=x_label, y=y_label[j], color=fig_cpal[i][j], lw=0.5, ax=axb) for j in y_label_indices[1:]]

            axb.set_ylabel('')
            axb.set_xlabel('')

        ## This is the default plotting method
        else:
            [sns.lineplot(data=grp, x=x_label, y=y_lab, color=fig_cpal[i], ax=ax) for y_lab in y_label]

        ## plot vertical lines if they are provided
        vlines = grp[vline_lab].unique() if vline_lab else []
        [ax.axvline(ln, c='lightgrey', ls='--') for ln in vlines]

        ## plot horizontal lines in they are provided
        hlines = grp[hline_lab].unique() if hline_lab else []
        [ax.axhline(ln, c='lightgrey', ls='--') for ln in hlines]

        ax.set_ylabel('')
        ax.set_xlabel('')

        ## add 1 to your ax index so you don't reuse any axes
        i += 1

    ## add a figure-wide legend if desired
    if legend == True:
        fhands_flabs_dict = [{fig_clab[j][i]:fig_cpal[j][i] for i in y_label_indices} for j in range(len(fig_cpal))]
        fhands_flabs_dict = [{k:set(x[k] for x in fhands_flabs_dict)} for k in y_label]
        fhands_flabs_dict = dict((' - '.join((k, hue_inv[v])),v) if len(d[k]) > 1 else (k,v) for d in fhands_flabs_dict for k in d for v in d[k])
        fhands_flabs = [mlines.Line2D([], [], color=fhands_flabs_dict[k], label=k) for k in fhands_flabs_dict]
        fig.legend(handles=fhands_flabs, ncols=len(y_label), loc='lower center', bbox_to_anchor=(0.5, 1))
    else:
        pass
    ## change some figure-wide parameters
    fig.suptitle(figtitle) if figtitle else fig.suptitle(None)
    fig.supxlabel(x_label)
    ## if only one y_label is passed then don't include the brackets in fig.supylabel
    fig.supylabel(y_label[0], x=0.008) if len(y_label)==1 else fig.supylabel(y_label, x=0.008)
    plt.tight_layout()

    ## Delete any leftover empty axes
    while i < (nrows * ncols):
        fig.delaxes(axs[axi[i]])
        i += 1

    ## Save your figure to the specified location
    fig.savefig(out_fpath, dpi=dpi, bbox_inches='tight')

    return


def plot_and_save(ntwrk_df, out_dir):
    ## For plotting purposes split up the dataset into 2D conditions and 3D conditions:
    ntwrk_df_2D = ntwrk_df.query('expt_condition == "2D-MG-EMT-1-60-MG" or expt_condition == "2D-PLF-EMT-1-60-MG"')
    ntwrk_df_3D = ntwrk_df.query('expt_condition == "3D-MG-EMT-1-60-MG" or expt_condition == "3D-MG-EMT-no-MG"')
    SAC_subset = ntwrk_df.query('expt_condition == "2D-MG-EMT-1-60-MG" or expt_condition == "2D-PLF-EMT-1-60-MG" or expt_condition == "3D-MG-EMT-1-60-MG"')

    # ntwrk_df_subsets = {'2D':ntwrk_df_2D, '3D':ntwrk_df_3D, 'SAC':SAC_subset}
    ntwrk_df_subsets = {'SAC':SAC_subset}

    for nm_sub in ntwrk_df_subsets:
        ntwrk_df_sub = ntwrk_df_subsets[nm_sub]

        fig, ax = plt.subplots()
        sns.lineplot(x='Time (hours)', 
                    y='ntwrk_edge_len_max_norm_maxinit',
                    hue='expt_condition',
                    palette=color_map,
                    errorbar='sd',
                    data=ntwrk_df_sub)
        ax.set_ylabel('Major Compoonent Size \n(Normalized to Initial Size)')
        ax.legend(title='Experimental Condition')
        plt.show()
        fig.savefig(out_dir/f'maj_comp_sz_norm_{nm_sub}.pdf', dpi=DPI, bbox_inches='tight')

        fig, ax = plt.subplots()
        sns.lineplot(x='Time (hours)', 
                    y='ntwrk_edge_len_max',
                    hue='expt_condition',
                    palette=color_map,
                    errorbar='sd',
                    data=ntwrk_df_sub)
        ax.set_ylabel('Major Compoonent Size (px)')
        ax.legend(title='Experimental Condition')
        plt.show()
        fig.savefig(out_dir/f'maj_comp_sz_{nm_sub}.pdf', dpi=DPI, bbox_inches='tight')


        for (nm, grp) in ntwrk_df_sub.groupby('expt_condition'):
            fig, ax = plt.subplots()
            sns.lineplot(data=grp, 
                        x='Time (hours)', 
                        y='ntwrk_edge_len_max', 
                        hue='timelapse_id', 
                        palette=[color_map[nm]],
                        legend=False,
                        errorbar=None,
                        marker='.', ax=ax)
            ax.set_ylabel('Major Compoonent Size (px)')
            plt.show()
            ec = '-'.join(nm.split('-')[:2])
            fig.savefig(out_dir/f'maj_comp_sz_indiv_{ec}.pdf', dpi=DPI, bbox_inches='tight')



        fig, ax = plt.subplots()
        sns.lineplot(x='Time (hours)', 
                    y='total_edge_len_sum_norm',
                    hue='expt_condition',
                    palette=color_map,
                    errorbar='sd',
                    data=ntwrk_df_sub)
        ax.set_ylabel('Total Network Size \n(Normalized to Initial Size)')
        ax.legend(title='Experimental Condition')
        plt.show()
        fig.savefig(out_dir/f'total_ntwrk_sz_norm_{nm_sub}.pdf', dpi=DPI, bbox_inches='tight')

        fig, ax = plt.subplots()
        sns.lineplot(x='Time (hours)', 
                    y='total_edge_len_sum',
                    hue='expt_condition',
                    palette=color_map,
                    errorbar='sd',
                    data=ntwrk_df_sub)
        ax.set_ylabel('Total Network Size (px)')
        ax.legend(title='Experimental Condition')
        plt.show()
        fig.savefig(out_dir/f'total_ntwrk_sz_{nm_sub}.pdf', dpi=DPI, bbox_inches='tight')

        for (nm, grp) in ntwrk_df_sub.groupby('expt_condition'):
            fig, ax = plt.subplots()
            sns.lineplot(data=grp, 
                        x='Time (hours)', 
                        y='total_edge_len_sum', 
                        hue='timelapse_id', 
                        palette=[color_map[nm]],
                        legend=False,
                        errorbar=None,
                        marker='.', ax=ax)
            ax.set_ylabel('Total Network Size (px)')
            plt.show()
            ec = '-'.join(nm.split('-')[:2])
            fig.savefig(out_dir/f'total_ntwrk_sz_indiv_{ec}.pdf', dpi=DPI, bbox_inches='tight')



        fig, ax = plt.subplots()
        sns.lineplot(x='Time (hours)', 
                    y='total_ntwrk_fluor_mean',
                    hue='expt_condition',
                    palette=color_map,
                    errorbar='sd',
                    data=ntwrk_df_sub)
        ax.set_ylabel('ZO1 Fluorescence of Network')
        ax.legend(title='Experimental Condition')
        plt.show()
        fig.savefig(out_dir/f'fluor_mean_{nm_sub}.pdf', dpi=DPI, bbox_inches='tight')

        fig, ax = plt.subplots()
        sns.lineplot(x='Time since migration onset (hours)',
                    y='total_ntwrk_fluor_mean',
                    hue='expt_condition',
                    palette=color_map,
                    errorbar='sd',
                    data=ntwrk_df_sub)
        ax.set_ylabel('ZO1 Fluorescence of Network')
        ax.axvline(0, c='k')
        ax.legend(title='Experimental Condition')
        plt.show()
        fig.savefig(out_dir/f'fluor_mean_{nm_sub}_migr_zerod.pdf', dpi=DPI, bbox_inches='tight')

        for (nm, grp) in ntwrk_df_sub.groupby('expt_condition'):
            fig, ax = plt.subplots()
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='total_ntwrk_fluor_mean',
                        hue='timelapse_id',
                        legend=False,
                        errorbar=None,
                        marker='.', ax=ax)
            plt.show()
            fig.savefig(out_dir/f'fluor_mean_indiv_{nm}.pdf', dpi=DPI, bbox_inches='tight')

        plt.close('all')
        for (nm, grp) in ntwrk_df_sub.groupby('timelapse_id'):
            ec = grp.expt_condition.unique()[-1]
            fig, ax = plt.subplots()
            axb = ax.twinx()
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='total_ntwrk_fluor_mean',
                        color=color_map[ec],
                        legend=False,
                        errorbar=None,
                        marker='.', 
                        ax=axb)
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='ntwrk_edge_len_max',
                        color='k',
                        legend=False,
                        errorbar=None,
                        marker='.',
                        ax=ax)
            ax.axvline(grp['compaction/cell death onset (hours)'].unique(), c='grey', ls='--', zorder=0)
            ax.axvline(grp['migration onset (hours)'].unique(), c='lightgrey', ls='--', zorder=0)
            ax.set_ylim(0)
            axb.set_ylim(100)
            axb.set_ylabel('ZO1 Fluorescence of Network', c=color_map[ec])
            ax.set_ylabel('Major Component Size (px)')
            ax.set_title(nm + tuple([ec]))
        plots_to_multipdf(out_dir/f'fluor_mean_w_majcomp_indiv_{nm_sub}.pdf', tight=True, dpi=300)

        if MAKE_CONTACT_SHEETS:
            mkcontsh(
                out_fpath = out_dir/f'fluor_mean_w_majcomp_{nm_sub}_contsh.tif',
                df=ntwrk_df_sub,
                x_label='Time (hours)',
                y_label=['total_ntwrk_fluor_mean', 'ntwrk_edge_len_max'],
                group_label='timelapse_id',
                hue_label='expt_condition',
                hue_palette=color_map,
                ncols=4,
                vline_lab='migration onset (hours)',
                hline_lab=None,
                figtitle=f'{nm_sub} Colonies',
                legend=True,
                twinx=True,
                dpi=300
            )


        plt.close('all')
        for (nm, grp) in ntwrk_df_sub.groupby('timelapse_id'):
            ec = grp.expt_condition.unique()[-1]
            fig, ax = plt.subplots()
            axb = ax.twinx()
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='total_ntwrk_fluor_mean',
                        color=color_map[ec],
                        legend=False,
                        errorbar=None,
                        marker='.', 
                        ax=axb)
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='total_edge_len_sum',
                        color='k',
                        legend=False,
                        errorbar=None,
                        marker='.',
                        ax=ax)
            ax.axvline(grp['compaction/cell death onset (hours)'].unique(), c='grey', ls='--', zorder=0)
            ax.axvline(grp['migration onset (hours)'].unique(), c='lightgrey', ls='--', zorder=0)
            ax.set_ylim(0)
            axb.set_ylim(100)
            axb.set_ylabel('ZO1 Fluorescence of Network', c=color_map[ec])
            ax.set_ylabel('Total Network Size (px)')
            ax.set_title(nm + tuple([ec]))
        plots_to_multipdf(out_dir/f'fluor_mean_w_totalsize_indiv_{nm_sub}.pdf', tight=True, dpi=300)

        if MAKE_CONTACT_SHEETS:
            mkcontsh(
                out_fpath = out_dir/f'fluor_mean_w_totalsize_{nm_sub}_contsh.tif',
                df=ntwrk_df_sub,
                x_label='Time (hours)',
                y_label=['total_ntwrk_fluor_mean', 'total_edge_len_sum'],
                group_label='timelapse_id',
                hue_label='expt_condition',
                hue_palette=color_map,
                ncols=4,
                vline_lab='migration onset (hours)',
                hline_lab=None,
                figtitle=f'{nm_sub} Colonies',
                legend=True,
                twinx=True,
                dpi=300
            )


        fig, ax = plt.subplots()
        sns.lineplot(x='Time (hours)', 
                    y='total_ntwrk_fluor_mean',
                    hue='expt_condition',
                    palette=color_map,
                    errorbar='sd',
                    data=ntwrk_df_sub)
        for ec in ntwrk_df_sub['expt_condition'].unique():
            [ax.axvline(ln, c=color_map[f'{ec}'], alpha=0.4, ls='--', zorder=0) for ln in ntwrk_df_sub.query('expt_condition == @ec')['migration onset (hours)'].unique()]
        ax.set_ylabel('ZO1 Fluorescence of Network')
        ax.legend(title='Experimental Condition')
        plt.show()
        fig.savefig(out_dir/f'fluor_mean_{nm_sub}.pdf', dpi=DPI, bbox_inches='tight')

        fig, ax = plt.subplots()
        sns.lineplot(x='Time since migration onset (hours)', 
                    y='total_ntwrk_fluor_mean',
                    hue='expt_condition',
                    palette=color_map,
                    errorbar='sd',
                    data=ntwrk_df_sub)
        ax.axvline(0, c='k')
        ax.set_ylabel('ZO1 Fluorescence of Network')
        ax.legend(title='Experimental Condition')
        plt.show()
        fig.savefig(out_dir/f'fluor_mean_{nm_sub}_migr_zerod.pdf', dpi=DPI, bbox_inches='tight')

        for (nm, grp) in ntwrk_df_sub.groupby('expt_condition'):
            fig, ax = plt.subplots()
            sns.lineplot(data=grp,
                        x='Time since migration onset (hours)',
                        y='total_ntwrk_fluor_mean',
                        hue='timelapse_id',
                        legend=False,
                        errorbar=None,
                        marker='.', ax=ax)
            plt.show()
            fig.savefig(out_dir/f'fluor_mean_indiv_{nm}.pdf', dpi=DPI, bbox_inches='tight')


        plt.close('all')
        for (nm, grp) in ntwrk_df_sub.groupby('timelapse_id'):
            ec = grp.expt_condition.unique()[-1]
            fig, ax = plt.subplots()
            axb = ax.twinx()
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='total_ntwrk_fluor_mean',
                        color=color_map[ec],
                        legend=False,
                        errorbar=None,
                        marker='.',
                        ax=axb)
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='ntwrk_edge_len_max',
                        color='k',
                        legend=False,
                        errorbar=None,
                        marker='.',
                        ax=ax)
            ax.axvline(grp['compaction/cell death onset (hours)'].unique(), c='grey', ls='--', zorder=0)
            ax.axvline(grp['migration onset (hours)'].unique(), c='lightgrey', ls='--', zorder=0)
            ax.set_ylim(0)
            axb.set_ylim(100)
            axb.set_ylabel('ZO1 Fluorescence of Network', c=color_map[ec])
            ax.set_ylabel('Major Component Size (px)')
            ax.set_title(nm + tuple([ec]))
        plots_to_multipdf(out_dir/f'fluor_mean_w_majcomp_indiv_{nm_sub}.pdf', tight=True, dpi=300)

        if MAKE_CONTACT_SHEETS:
            mkcontsh(
                out_fpath = out_dir/f'fluor_mean_w_majcomp_{nm_sub}_contsh.tif',
                df=ntwrk_df_sub,
                x_label='Time (hours)',
                y_label=['total_ntwrk_fluor_mean', 'ntwrk_edge_len_max'],
                group_label='timelapse_id',
                hue_label='expt_condition',
                hue_palette=color_map,
                ncols=4,
                vline_lab='migration onset (hours)',
                hline_lab=None,
                figtitle=f'{nm_sub} Colonies',
                legend=True,
                twinx=True,
                dpi=300
            )


        plt.close('all')
        for (nm, grp) in ntwrk_df_sub.groupby('timelapse_id'):
            ec = grp.expt_condition.unique()[-1]
            fig, ax = plt.subplots()
            axb = ax.twinx()
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='total_ntwrk_fluor_mean',
                        color=color_map[ec],
                        legend=False,
                        errorbar=None,
                        marker='.',
                        ax=axb)
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='total_edge_len_sum',
                        color='k',
                        legend=False,
                        errorbar=None,
                        marker='.',
                        ax=ax)
            ax.axvline(grp['compaction/cell death onset (hours)'].unique(), c='grey', ls='--', zorder=0)
            ax.axvline(grp['migration onset (hours)'].unique(), c='lightgrey', ls='--', zorder=0)
            ax.set_ylim(0)
            axb.set_ylim(100)
            axb.set_ylabel('ZO1 Fluorescence of Network', c=color_map[ec])
            ax.set_ylabel('Total Network Size (px)')
            ax.set_title(nm + tuple([ec]))
        plots_to_multipdf(out_dir/f'fluor_mean_w_totalsize_indiv_{nm_sub}.pdf', tight=True, dpi=300)

        if MAKE_CONTACT_SHEETS:
            mkcontsh(
                out_fpath = out_dir/f'fluor_mean_w_totalsize_{nm_sub}_contsh.tif',
                df=ntwrk_df_sub,
                x_label='Time (hours)',
                y_label=['total_ntwrk_fluor_mean', 'total_edge_len_sum'],
                group_label='timelapse_id',
                hue_label='expt_condition',
                hue_palette=color_map,
                ncols=4,
                vline_lab='migration onset (hours)',
                hline_lab=None,
                figtitle=f'{nm_sub} Colonies',
                legend=True,
                twinx=True,
                dpi=300
            )


        plt.close('all')
        for (nm, grp) in ntwrk_df_sub.groupby('timelapse_id'):
            ec = grp.expt_condition.unique()[-1]
            fig, ax = plt.subplots()
            axb = ax.twinx()
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='ntwrk_edge_len_max',
                        color=color_map[ec],
                        legend=False,
                        errorbar=None,
                        marker='.', 
                        ax=axb)
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='total_edge_len_sum',
                        color='k',
                        legend=False,
                        errorbar=None,
                        marker='.',
                        ax=ax)
            ax.axvline(grp['compaction/cell death onset (hours)'].unique(), c='grey', ls='--', zorder=0)
            ax.axvline(grp['migration onset (hours)'].unique(), c='lightgrey', ls='--', zorder=0)
            ax.set_ylim(0)
            axb.set_ylim(100)
            axb.set_ylabel('ZO1 Fluorescence of Network', c=color_map[ec])
            ax.set_ylabel('Total Network Size (px)')
            ax.set_title(nm + tuple([ec]))
        plots_to_multipdf(out_dir/f'majcomp_w_totalsize_indiv_{nm_sub}.pdf', tight=True, dpi=300)

        if MAKE_CONTACT_SHEETS:
            mkcontsh(
                out_fpath = out_dir/f'majcomp_w_totalsize_{nm_sub}_contsh.tif',
                df=ntwrk_df_sub,
                x_label='Time (hours)',
                y_label=['ntwrk_edge_len_max', 'total_edge_len_sum'],
                group_label='timelapse_id',
                hue_label='expt_condition',
                hue_palette=color_map,
                ncols=4,
                vline_lab='migration onset (hours)',
                hline_lab=None,
                figtitle=f'{nm_sub} Colonies',
                legend=True,
                twinx=True,
                dpi=300
            )


        plt.close('all')
        for (nm, grp) in ntwrk_df_sub.groupby('timelapse_id'):
            ec = grp.expt_condition.unique()[-1]
            fig, ax = plt.subplots()
            axb = ax.twinx()
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='ntwrk_edge_len_max',
                        color=color_map[ec],
                        legend=False,
                        errorbar=None,
                        marker='.',
                        ax=axb)
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='total_edge_len_sum',
                        color='k',
                        legend=False,
                        errorbar=None,
                        marker='.',
                        ax=ax)
            ax.axvline(grp['compaction/cell death onset (hours)'].unique(), c='grey', ls='--', zorder=0)
            ax.axvline(grp['migration onset (hours)'].unique(), c='lightgrey', ls='--', zorder=0)
            ax.set_ylim(0)
            axb.set_ylim(100)
            axb.set_ylabel('ZO1 Fluorescence of Network', c=color_map[ec])
            ax.set_ylabel('Total Network Size (px)')
            ax.set_title(nm + tuple([ec]))
        plots_to_multipdf(out_dir/f'majcomp_w_totalsize_indiv_{nm_sub}.pdf', tight=True, dpi=300)

        if MAKE_CONTACT_SHEETS:
            mkcontsh(
                out_fpath = out_dir/f'majcomp_w_totalsize_{nm_sub}_contsh.tif',
                df=ntwrk_df_sub,
                x_label='Time (hours)',
                y_label=['ntwrk_edge_len_max', 'total_edge_len_sum'],
                group_label='timelapse_id',
                hue_label='expt_condition',
                hue_palette=color_map,
                ncols=4,
                vline_lab='migration onset (hours)',
                hline_lab=None,
                figtitle=f'{nm_sub} Colonies',
                legend=True,
                twinx=True,
                dpi=300
            )


        plt.close('all')
        for (nm, grp) in ntwrk_df_sub.groupby('timelapse_id'):
            ec = grp.expt_condition.unique()[-1]
            fig, ax = plt.subplots()
            axb = ax.twinx()
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='total_ntwrk_fluor_mean',
                        color=color_map[ec],
                        legend=False,
                        errorbar=None,
                        marker='.',
                        ax=axb)
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='node_centroids_subpx_Z_mean',
                        color='k',
                        legend=False,
                        errorbar=None,
                        marker='.',
                        ax=ax)
            ax.axvline(grp['compaction/cell death onset (hours)'].unique(), c='grey', ls='--', zorder=0)
            ax.axvline(grp['migration onset (hours)'].unique(), c='lightgrey', ls='--', zorder=0)
            ax.set_ylim(0)
            axb.set_ylim(100)
            axb.set_ylabel('ZO1 Fluorescence of Network', c=color_map[ec])
            ax.set_ylabel('Median Node Z-position')
            ax.set_title(nm + tuple([ec]))
        plots_to_multipdf(out_dir/f'totalfluor_w_nodeZ_indiv_{nm_sub}.pdf', tight=True, dpi=300)

        if MAKE_CONTACT_SHEETS:
            mkcontsh(
                out_fpath = out_dir/f'totalfluor_w_nodeZ_{nm_sub}_contsh.tif',
                df=ntwrk_df_sub,
                x_label='Time (hours)',
                y_label=['total_ntwrk_fluor_mean', 'node_centroids_subpx_Z_mean'],
                group_label='timelapse_id',
                hue_label='expt_condition',
                hue_palette=color_map,
                ncols=4,
                vline_lab='migration onset (hours)',
                hline_lab=None,
                figtitle=f'{nm_sub} Colonies',
                legend=True,
                twinx=True,
                dpi=300
            )



        plt.close('all')
        for (nm, grp) in ntwrk_df_sub.groupby('timelapse_id'):
            ec = grp.expt_condition.unique()[-1]
            fig, ax = plt.subplots()
            axb = ax.twinx()
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='total_ntwrk_fluor_mean',
                        color=color_map[ec],
                        legend=False,
                        errorbar=None,
                        marker='.',
                        ax=axb)
            sns.lineplot(data=grp,
                        x='Time (hours)',
                        y='node_centroids_subpx_Z_mean',
                        color='k',
                        legend=False,
                        errorbar=None,
                        marker='.',
                        ax=ax)
            ax.axvline(grp['compaction/cell death onset (hours)'].unique(), c='grey', ls='--', zorder=0)
            ax.axvline(grp['migration onset (hours)'].unique(), c='lightgrey', ls='--', zorder=0)
            ax.set_ylim(0)
            axb.set_ylim(100)
            axb.set_ylabel('ZO1 Fluorescence of Network', c=color_map[ec])
            ax.set_ylabel('Median Node Z-position')
            ax.set_title(nm + tuple([ec]))
        plots_to_multipdf(out_dir/f'totalfluor_w_nodeZ_indiv_{nm_sub}.pdf', tight=True, dpi=300)

        if MAKE_CONTACT_SHEETS:
            mkcontsh(
                out_fpath = out_dir/f'totalfluor_w_nodeZ_{nm_sub}_contsh.tif',
                df=ntwrk_df_sub,
                x_label='Time (hours)',
                y_label=['total_ntwrk_fluor_mean', 'node_centroids_subpx_Z_mean'],
                group_label='timelapse_id',
                hue_label='expt_condition',
                hue_palette=color_map,
                ncols=4,
                vline_lab='migration onset (hours)',
                hline_lab=None,
                figtitle=f'{nm_sub} Colonies',
                legend=True,
                twinx=True,
                dpi=300
            )


        fig, ax = plt.subplots()
        sns.lineplot(x='Time (hours)', 
                    y='mean_edge_len',
                    hue='expt_condition',
                    palette=color_map,
                    data=ntwrk_df_sub)
        ax.set_ylabel('Mean Apical Edge Length (vx)')
        ax.legend(title='Experimental Condition')
        plt.show()
        fig.savefig(out_dir/'edge_mean.pdf', dpi=DPI, bbox_inches='tight')


        fig, ax = plt.subplots()
        sns.lineplot(x='Time (hours)', 
                    y='median_edge_len',
                    hue='expt_condition',
                    palette=color_map,
                    data=ntwrk_df_sub)
        ax.set_ylabel('Median Apical Edge Length (vx)')
        [t for t in ax.legend().texts]
        ax.legend(title='Experimental Condition')
        plt.show()
        fig.savefig(out_dir/'edge_median.pdf', dpi=DPI, bbox_inches='tight')


        ## Try looking at the distribution of network component sizes and fluorescence over time
        plt.close('all')
        for nm, df in ntwrk_df_sub.groupby(['expt_condition', 'barcode', 'position_well']):
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='timeframe', y='node_centroids_subpx_Z_mean_per_network', marker='.',
                            hue='ntwrk_fluor_mean', palette='crest', linewidth=0, alpha=0.5, ax=ax, legend=True)
            ax.axvline(df['migration onset (hours)'].unique()*2, c='r', ls='--')
            ax.axhline(29, c='k', ls='--')
            ax.get_legend().remove()
            ax.legend(*ax.get_legend_handles_labels(), title='ntwrk_fluor_mean', ncols=8, loc='best')
            ax.set_title(nm)
        plots_to_multipdf(out_dir/f't_vs_Z_hue-ntwrkfluormean_{nm_sub}.pdf', tight=False, dpi=300)


        plt.close('all')
        for nm, df in ntwrk_df_sub.groupby(['expt_condition', 'barcode', 'position_well']):
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x='timeframe', y='total_ntwrk_fluor_mean', c='k', linewidth=0.5, alpha=0.3, ax=ax)
            ax2 = ax.twinx()
            sns.scatterplot(data=df, x='timeframe', y='ntwrk_fluor_mean', marker='.',
                            hue='node_centroids_subpx_Z_mean_per_network', palette='viridis',
                            linewidth=0, alpha=0.2, ax=ax2, legend=True)
            ax2.axvline(df['migration onset (hours)'].unique()*2, c='r', ls='--')
            ax2.get_legend().remove()
            ax2.legend(*ax2.get_legend_handles_labels(), title='node_centroids_subpx_Z_mean_per_network', ncols=8, loc='best')
            ax.set_title(nm)
            ax2.semilogy()
        plots_to_multipdf(out_dir/f't_vs_fluor_hue-Zmean_{nm_sub}.pdf', tight=True, dpi=300)


        plt.close('all')
        for nm, df in ntwrk_df_sub.groupby(['expt_condition', 'barcode', 'position_well']):
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='total_edge_len', y='ntwrk_fluor_mean', marker='.',
                            hue='timeframe', palette='Spectral',
                            linewidth=0, alpha=0.5, ax=ax, legend=True)
            ax.get_legend().remove()
            ax.legend(*ax.get_legend_handles_labels(), title='timeframe', ncols=8, loc='best')
            ax.set_title(nm)
            ax.semilogy()
            ax.semilogx()
        plots_to_multipdf(out_dir/f'len_vs_fluor_hue-T_{nm_sub}.pdf', tight=False, dpi=300)

        plt.close('all')
        for nm, df in ntwrk_df_sub.groupby(['expt_condition', 'barcode', 'position_well']):
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='timeframe', y='total_edge_len', marker='.',
                            hue='timeframe', palette='Spectral',
                            linewidth=0, alpha=0.5, ax=ax, legend=True)
            ax.get_legend().remove()
            ax.legend(*ax.get_legend_handles_labels(), title='timeframe', ncols=8, loc='best')
            ax.set_title(nm)
            ax.semilogy()
            ax.semilogx()
        plots_to_multipdf(out_dir/f'len_vs_fluor_hue-T_{nm_sub}.pdf', tight=False, dpi=300)

        plt.close('all')
        for nm, df in ntwrk_df_sub.groupby(['expt_condition', 'barcode', 'position_well']):
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='timeframe', y='total_ntwrk_fluor_mean', marker='.',
                            hue='node_centroids_subpx_Z_mean', palette='viridis', linewidth=0, alpha=0.5, ax=ax, legend=True)
            ax.axvline(df['migration onset (hours)'].unique()*2, c='r', ls='--')
            ax.get_legend().remove()
            ax.legend(*ax.get_legend_handles_labels(), title='node_centroids_subpx_Z_mean', ncols=8, loc='best')
            ax.set_title(nm)
        plots_to_multipdf(out_dir/f't_vs_totalfluor_hue-Zmean_{nm_sub}.pdf', tight=False, dpi=300)


        plt.close('all')
        for nm, df in ntwrk_df_sub.groupby(['expt_condition', 'barcode', 'position_well']):
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='timeframe', y='total_edge_len', marker='.',
                            hue='node_centroids_subpx_Z_mean', palette='viridis', linewidth=0, alpha=0.5, ax=ax, legend=True)
            ax.axvline(df['migration onset (hours)'].unique()*2, c='r', ls='--')
            ax.get_legend().remove()
            ax.legend(*ax.get_legend_handles_labels(), title='node_centroids_subpx_Z_mean', ncols=8, loc='best')
            ax.set_title(nm)
        plots_to_multipdf(out_dir/f't_vs_len_hue-Zmean_{nm_sub}.pdf', tight=False, dpi=300)


        plt.close('all')
        for nm, df in ntwrk_df_sub.groupby(['expt_condition', 'barcode', 'position_well']):
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x='timeframe', y='num_ntwrks', c='k', linewidth=0.5, alpha=0.3, ax=ax)
            ax2 = ax.twinx()
            sns.scatterplot(data=df, x='timeframe', y='total_edge_len', marker='.',
                            c='tab:blue', linewidth=0, alpha=0.2, ax=ax2, legend=False)
            sns.scatterplot(data=df, x='timeframe', y='total_edge_len_max', marker='.',
                            c='k', linewidth=0, alpha=1, ax=ax2, legend=False)
            ax2.axvline(df['migration onset (hours)'].unique()*2, c='r', ls='--')
            ax2.set_title(nm)
        plots_to_multipdf(out_dir/f't_vs_len_vs_numntwrks_{nm_sub}.pdf', tight=True, dpi=300)

    return






## Some global parameters for plots:
DPI = 300

## Time resolution for timelapse is 0.5 hours per timeframe:
t_res = 0.5

## The colormap used elsewhere in the manuscript:
## https://github.com/aics-int/Analysis_deliverable/blob/main/Analysis_scripts/orientation_analysis.py
color_map = {'2D-MG-EMT-1-60-MG':"deepskyblue",
             '2D-PLF-EMT-1-60-MG':"darkmagenta", 
             '3D-MG-EMT-1-60-MG':"orange",
             '3D-MG-EMT-no-MG':"darkgreen"}
SEP_to_Nivi_expt_cond_map = {'2D MG EMT 1:60MG':'2D-MG-EMT-1-60-MG',
                             '2D PLF EMT 1:60MG':'2D-PLF-EMT-1-60-MG',
                             '3D MG EMT 1:60MG':'3D-MG-EMT-1-60-MG',
                             '3D MG EMT no MG':'3D-MG-EMT-no-MG'}

## Get some local filename and directory locations:
sct_fpath = Path(__file__)
cwd = sct_fpath.parent
sct_fstem = sct_fpath.stem
barcode_annotations = pd.read_csv(Path.joinpath(cwd, 'annotations/zo1_to_seg_done.csv'))
brightfield_annotations = pd.read_csv(cwd/'annotations/movie_All_manifest_v0.csv')

## this is the directory where our datatables describing the networks are kept:
data_dir = Path.joinpath(cwd, 'ntwrks_postprocessing2_mp_out/network_tables')

## THIS LOADS THE UNFILTERED DATASET:
ntwrk_df = pd.read_csv(Path.joinpath(data_dir, 'ntwrk_unfilt.tsv'), sep='\t')

## THIS LOADS THE MAJOR COMPONENT DATASET:
# ntwrk_df_maj_comp = pd.read_csv(Path.joinpath(data_dir, 'ntwrk_majcomp.tsv'), sep='\t')

## this loads the UNFILTERED nodes dataset:
nodes_df = pd.read_csv(cwd/'concat_tables_out/nodes_dataset.tsv', sep='\t')

## Create a folder for image output if it doesn't already exist:
out_dir = Path.joinpath(cwd, sct_fstem + '_out')
Path.mkdir(out_dir, exist_ok=True)

## create a position_well column in barcode annotations that is consistent with the
## one in ntwrk_df:
barcode_annotations['position_well'] = barcode_annotations.apply(lambda x: 'P'+'-'.join(x[['Position Index', 'Well Label']].astype(str)), axis=1)

## Use this column in barcode annotations to map the compaction onset times and
## migration onset times into a column in ntwrk_df
comp_d = dict(zip(barcode_annotations.position_well, barcode_annotations['compaction/cell death onset (timeframe)']))
migr_d = dict(zip(barcode_annotations.position_well, barcode_annotations['migration onset (timeframe)']))

ntwrk_df['compaction/cell death onset (timeframe)'] = ntwrk_df.position_well.transform(lambda x: comp_d[x])
ntwrk_df['migration onset (timeframe)'] = ntwrk_df.position_well.transform(lambda x: migr_d[x])

ntwrk_df['compaction/cell death onset (hours)'] = ntwrk_df['compaction/cell death onset (timeframe)'].transform(lambda x: x*t_res)
ntwrk_df['migration onset (hours)'] = ntwrk_df['migration onset (timeframe)'].transform(lambda x: x*t_res)

## convert the barcodes columns to strings
ntwrk_df['barcode'] = ntwrk_df.barcode.astype(str)
nodes_df['barcode'] = nodes_df.barcode.astype(str)

## remove the barcode that was an initial test (if present)
nodes_df.query("barcode != 'initial_test'", inplace=True)

## remove barcodes that were excluded from the dataset because
## of the way that the movies were acquired
bad_barcodes = ('3500006060', '3500005818', 'AD00004917')
ntwrk_df = ntwrk_df.query('barcode not in @bad_barcodes')
nodes_df = nodes_df.query('barcode not in @bad_barcodes')

## Keep just the ZO-1 (aka TJP1) filepaths
brightfield_annotations = brightfield_annotations.query('gene == "TJP1"')

## Assign experimental condition labels from barcode_annotations to nodes_df and edges_df
## according to the plate barcode, position index, and well label
poswell_exptcond_map = {x[0]:x[1] for x in barcode_annotations[['position_well', 'expt_condition']].values}

## create the experimental condition column for nodes_df
nodes_df['expt_condition'] = nodes_df['position_well'].apply(lambda x: poswell_exptcond_map[x])

## convert my experimental condition labels to be consistent with Nivi's for nodes_df
nodes_df['expt_condition'] = nodes_df['expt_condition'].apply(lambda x: SEP_to_Nivi_expt_cond_map[x] if x in SEP_to_Nivi_expt_cond_map else x)

## add a column with the color for each expt_condition
nodes_df['expt_condition_color'] = nodes_df['expt_condition'].transform(lambda x: color_map[x] if x in color_map else x)
ntwrk_df['expt_condition_color'] = ntwrk_df['expt_condition'].transform(lambda x: color_map[x] if x in color_map else x)


### Code for filtering:
cols = ['expt_condition', 'barcode', 'position_well']

init_6hrs_total_ntwrk_fluor_mean = ntwrk_df.query('timeframe < 12').groupby(cols).apply(lambda df: df.total_ntwrk_fluor_mean.mean())
d = init_6hrs_total_ntwrk_fluor_mean.to_dict()
ntwrk_df['init_6hrs_total_ntwrk_fluor_mean_mean'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

init_6hrs_total_ntwrk_fluor_mean_std = ntwrk_df.query('timeframe < 12').groupby(cols).apply(lambda df: df.total_ntwrk_fluor_mean.std())
d = init_6hrs_total_ntwrk_fluor_mean_std.to_dict()
ntwrk_df['init_6hrs_total_ntwrk_fluor_mean_std'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

init_6hrs_total_ntwrk_fluor_mean_max = ntwrk_df.query('timeframe < 12').groupby(cols).apply(lambda df: df.total_ntwrk_fluor_mean.max())
d = init_6hrs_total_ntwrk_fluor_mean_max.to_dict()
ntwrk_df['init_6hrs_total_ntwrk_fluor_mean_max'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)



init_6hrs_total_ntwrk_fluor_median = ntwrk_df.query('timeframe < 12').groupby(cols).apply(lambda df: df.total_ntwrk_fluor_median.median())
d = init_6hrs_total_ntwrk_fluor_median.to_dict()
ntwrk_df['init_6hrs_total_ntwrk_fluor_median_median'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

init_6hrs_total_ntwrk_fluor_median_std = ntwrk_df.query('timeframe < 12').groupby(cols).apply(lambda df: df.total_ntwrk_fluor_median.std())
d = init_6hrs_total_ntwrk_fluor_median_std.to_dict()
ntwrk_df['init_6hrs_total_ntwrk_fluor_median_std'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

init_6hrs_total_ntwrk_fluor_median_max = ntwrk_df.query('timeframe < 12').groupby(cols).apply(lambda df: df.total_ntwrk_fluor_median.max())
d = init_6hrs_total_ntwrk_fluor_median_max.to_dict()
ntwrk_df['init_6hrs_total_ntwrk_fluor_median_max'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

ntwrk_df['num_nodes'] = ntwrk_df.node_label.transform(lambda x: len(x))
ntwrk_df['num_edges'] = ntwrk_df.edge_label.transform(lambda x: len(x))


ntwrk_df['timeframe_max'] = ntwrk_df.groupby(cols)['timeframe'].transform(max)
last_6hrs_ntwrk_len_mean = ntwrk_df.query('timeframe >= (timeframe_max - 12)').groupby(cols).apply(lambda df: df['total_edge_len'].mean())
d = last_6hrs_ntwrk_len_mean.to_dict()
ntwrk_df['last_6hrs_ntwrk_len_mean'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

last_6hrs_ntwrk_len_std = ntwrk_df.query('timeframe >= (timeframe_max - 12)').groupby(cols).apply(lambda df: df['total_edge_len'].std())
d = last_6hrs_ntwrk_len_std.to_dict()
ntwrk_df['last_6hrs_ntwrk_len_std'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)


## Get the size of the biggest network for each timepoint (ie. the major component):
cols = ['expt_condition', 'barcode', 'position_well', 'timeframe']

ntwrk_sizes_max = ntwrk_df.groupby(cols).apply(lambda df: df.total_edge_len.max())
d = ntwrk_sizes_max.to_dict()
ntwrk_df['ntwrk_edge_len_max'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

## Get standard deviation in network sizes for each timepoint:
ntwrk_sizes_std = ntwrk_df.groupby(cols).apply(lambda df: df.total_edge_len.std())
d = ntwrk_sizes_std.to_dict()
ntwrk_df['ntwrk_edge_len_std'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

## Get the number of networks present at each timepoint:
num_ntwrks = ntwrk_df.groupby(cols).apply(lambda df: len(df.network_label))
d = num_ntwrks.to_dict()
ntwrk_df['num_ntwrks'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)


# ## NOTE that the max ntwrk label number can change over time depending on how pieces break off
# ## (eg. if a piece breaks off in the upper left then networks further down could get a network
# ## label with a larger value, since labeling is done in a rasterized fashion).
# ## Also, there can be 2 major components if there is a tie at a particular timepoint...
# ## Therefore they are not very useful right now for grouping and plotting data.

grps = ntwrk_df.groupby(cols)

ntwrk_df['ntwrk_fluor_median_minmaxnorm'] = grps.ntwrk_fluor_median.transform(preprocessing.minmax_scale)
ntwrk_df['ntwrk_fluor_mean_minmaxnorm'] = grps.ntwrk_fluor_mean.transform(preprocessing.minmax_scale)
ntwrk_df['ntwrk_fluor_max_minmaxnorm'] = grps.ntwrk_fluor_max.transform(preprocessing.minmax_scale)

## Just checking that the normalized columns actually range from 0 to 1:
# grps = ntwrk_df_2D.groupby(cols)
# for grp in grps:
#     nm, df = grp
#     print(df.ntwrk_fluor_median_minmaxnorm.min(), df.ntwrk_fluor_median_minmaxnorm.max())

## Find the total network size of the graph at timeframe == 0 for all ['expt_condition', 'barcode', 'position_well'] groups
cols = ['expt_condition', 'barcode', 'position_well']
ntwrk_sizes = ntwrk_df.groupby([*cols, 'timeframe']).apply(lambda df: df.total_edge_len.max())
d = ntwrk_sizes.to_dict()
ntwrk_df['total_edge_len_max'] = ntwrk_df[[*cols, 'timeframe']].apply(lambda x: d[*x], axis=1)

ntwrk_sizes = ntwrk_df.groupby([*cols, 'timeframe']).apply(lambda df: df.total_edge_len.sum())
d = ntwrk_sizes.to_dict()
ntwrk_df['total_edge_len_sum'] = ntwrk_df[[*cols, 'timeframe']].apply(lambda x: d[*x], axis=1)

ntwrk_sizes_init = ntwrk_df.groupby(cols).apply(lambda df: df[df['timeframe'] == 0].total_edge_len.sum())
d = ntwrk_sizes_init.to_dict()
ntwrk_df['total_edge_len_init'] = ntwrk_df[cols].apply(lambda x: d[*x], axis=1)

## Find the network size of the major component at timeframe == 0 for all ['expt_condition', 'barcode', 'position_well'] groups
ntwrk_sizes_init = ntwrk_df.groupby(cols).apply(lambda df: df[df['timeframe'] == 0].total_edge_len.max())
d = ntwrk_sizes_init.to_dict()
ntwrk_df['ntwrk_edge_len_max_init'] = ntwrk_df[cols].apply(lambda x: d[*x], axis=1)

## Normalize size of major component to initial size of the graph and the initial size of the major component
ntwrk_df['ntwrk_edge_len_max_norm_maxinit'] = ntwrk_df['ntwrk_edge_len_max'] / ntwrk_df['ntwrk_edge_len_max_init']
ntwrk_df['ntwrk_edge_len_max_norm_totalinit'] = ntwrk_df['ntwrk_edge_len_max'] / ntwrk_df['total_edge_len_init']

ntwrk_df['total_edge_len_sum_norm'] =  ntwrk_df['total_edge_len_sum'] / ntwrk_df['total_edge_len_init']

cols = ['barcode', 'position_well']
ntwrk_df['timelapse_id'] = ntwrk_df[cols].apply(lambda x: tuple((*x,)), axis=1)
ntwrk_df['timelapse_id'] = ntwrk_df[cols].apply(lambda x: tuple((*x,)), axis=1)

## zero time around migration timing:
ntwrk_df['Time since migration onset (hours)'] = ntwrk_df[['Time (hours)', 'migration onset (hours)']].apply(lambda x: x['Time (hours)'] - x['migration onset (hours)'], axis=1)

## filter nodes_df so that it only includes timepoints that are found in the filtered ntwrk_df dataset
cols = ['expt_condition', 'barcode', 'position_well']

tpoints_in_ntwrk_df = ntwrk_df.groupby(cols).apply(lambda df: set(df.timeframe.to_list()))
d = tpoints_in_ntwrk_df.to_dict()
nodes_df['valid_tpoints'] = nodes_df[cols].apply(lambda df: d[*df], axis=1)

nodes_df = nodes_df[nodes_df[['timeframe', 'valid_tpoints']].apply(lambda df: df.timeframe in df.valid_tpoints, axis=1)]

## filter nodes_df so it only includes nodes that are found in the filtered ntwrk_df dataset
cols = ['expt_condition', 'barcode', 'position_well', 'timeframe']

filtered_ntwrk_labels = ntwrk_df.groupby(cols).apply(lambda df: df.network_label.to_list())
d = filtered_ntwrk_labels.to_dict()
nodes_df['cln_ntwrks'] = nodes_df[cols].apply(lambda df: d[*df], axis=1)

nodes_df_cln = nodes_df[nodes_df[['network_label', 'cln_ntwrks']].apply(lambda df: df.network_label in df.cln_ntwrks, axis=1)]

nodes_df_cln['node_centroids_subpx'] = nodes_df_cln.node_centroids_subpx.transform(lambda x: stringified_floatlist_to_floatlist(x))

nodes_df_cln['node_centroids_subpx_Z'] = nodes_df_cln['node_centroids_subpx'].transform(lambda x: x[0])
nodes_df_cln['node_centroids_subpx_Y'] = nodes_df_cln['node_centroids_subpx'].transform(lambda x: x[1])
nodes_df_cln['node_centroids_subpx_X'] = nodes_df_cln['node_centroids_subpx'].transform(lambda x: x[2])



cols = ['expt_condition', 'barcode', 'position_well', 'timeframe']

node_Z_mean = nodes_df_cln.groupby(cols)['node_centroids_subpx_Z'].apply(lambda s: s.mean())
d = node_Z_mean.to_dict()
nodes_df_cln['node_centroids_subpx_Z_mean'] = nodes_df_cln[cols].apply(lambda df: d[*df], axis=1)

ntwrk_df['node_centroids_subpx_Z_mean'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

node_Z_mean = nodes_df_cln.groupby(cols + ['network_label'])['node_centroids_subpx_Z'].apply(lambda s: s.mean())
d = node_Z_mean.to_dict()
nodes_df_cln['node_centroids_subpx_Z_mean_per_network'] = nodes_df_cln[cols + ['network_label']].apply(lambda df: d[*df], axis=1)

ntwrk_df['node_centroids_subpx_Z_mean_per_network'] = ntwrk_df[cols + ['network_label']].apply(lambda df: d[*df] if tuple(df) in d else np.nan, axis=1)


## THIS FILTER CORRESPONDS TO THE 26mar630pm PLOTS.
ntwrk_df_filt = ntwrk_df.query('((num_nodes > 11) and (mean_edge_len >= ntwrk_edge_len_mean)) or (ntwrk_fluor_median >= (init_6hrs_total_ntwrk_fluor_median_median + 0*init_6hrs_total_ntwrk_fluor_median_std))')
ntwrk_df_filt = ntwrk_df.query('((num_edges > 10) and (mean_edge_len >= ntwrk_edge_len_mean)) or (ntwrk_fluor_median >= (init_6hrs_total_ntwrk_fluor_median_median + 0*init_6hrs_total_ntwrk_fluor_median_std))')

ntwrk_df_filt_no_sm_pcs = ntwrk_df.query('total_edge_len > (last_6hrs_ntwrk_len_mean + last_6hrs_ntwrk_len_std)')

## Need to update num_ntwrks in network_df_filt now that filtering has been done
cols = ['expt_condition', 'barcode', 'position_well', 'timeframe']
num_ntwrks = ntwrk_df_filt.groupby(cols).apply(lambda df: len(df.network_label))
d = num_ntwrks.to_dict()
ntwrk_df_filt['num_ntwrks'] = ntwrk_df_filt[cols].apply(lambda df: d[*df], axis=1)

## filter nodes_df so that it only includes timepoints that are found in the filtered ntwrk_df dataset
cols = ['expt_condition', 'barcode', 'position_well']

tpoints_in_ntwrk_df_filt = ntwrk_df_filt.groupby(cols).apply(lambda df: set(df.timeframe.to_list()))
d = tpoints_in_ntwrk_df_filt.to_dict()
nodes_df['valid_tpoints'] = nodes_df[cols].apply(lambda df: d[*df], axis=1)

nodes_df_filt = nodes_df[nodes_df[['timeframe', 'valid_tpoints']].apply(lambda df: df.timeframe in df.valid_tpoints, axis=1)]


## filter nodes_df so it only includes nodes that are found in the filtered ntwrk_df dataset
cols = ['expt_condition', 'barcode', 'position_well', 'timeframe']

filtered_ntwrk_labels = ntwrk_df_filt.groupby(cols).apply(lambda df: df.network_label.to_list())
d = filtered_ntwrk_labels.to_dict()
nodes_df_filt['cln_ntwrks'] = nodes_df_filt[cols].apply(lambda df: d[*df], axis=1)

nodes_df_cln_filt = nodes_df_filt[nodes_df_filt[['network_label', 'cln_ntwrks']].apply(lambda df: df.network_label in df.cln_ntwrks, axis=1)]


nodes_df_cln_filt['node_centroids_subpx'] = nodes_df_cln_filt.node_centroids_subpx.transform(lambda x: stringified_floatlist_to_floatlist(x))

nodes_df_cln_filt['node_centroids_subpx_Z'] = nodes_df_cln_filt['node_centroids_subpx'].transform(lambda x: x[0])
nodes_df_cln_filt['node_centroids_subpx_Y'] = nodes_df_cln_filt['node_centroids_subpx'].transform(lambda x: x[1])
nodes_df_cln_filt['node_centroids_subpx_X'] = nodes_df_cln_filt['node_centroids_subpx'].transform(lambda x: x[2])



cols = ['expt_condition', 'barcode', 'position_well', 'timeframe']

node_Z_mean = nodes_df_cln_filt.groupby(cols)['node_centroids_subpx_Z'].apply(lambda s: s.mean())
d = node_Z_mean.to_dict()
nodes_df_cln_filt['node_centroids_subpx_Z_mean'] = nodes_df_cln_filt[cols].apply(lambda df: d[*df], axis=1)

ntwrk_df_filt['node_centroids_subpx_Z_mean'] = ntwrk_df_filt[cols].apply(lambda df: d[*df], axis=1)



## Save these plots:
if PLOT_DATA:
    print('Starting to plot and save...')

    plt.close('all')
    prj_dir = 'unfilt'
    Path.mkdir(out_dir/prj_dir, parents=True, exist_ok=True)
    plot_and_save(ntwrk_df, out_dir/prj_dir)
    plt.close('all')

    plt.close('all')
    prj_dir = 'filt'
    Path.mkdir(out_dir/prj_dir, parents=True, exist_ok=True)
    plot_and_save(ntwrk_df_filt, out_dir/prj_dir)
    plt.close('all')

    plt.close('all')
    prj_dir = 'filt_no_sm_pcs'
    Path.mkdir(out_dir/prj_dir, parents=True, exist_ok=True)
    plot_and_save(ntwrk_df_filt_no_sm_pcs, out_dir/prj_dir)
    plt.close('all')
else:
    pass


## Plot the migration timings for the full set of data:
barcode_annotations['migration onset (hours)'] = barcode_annotations['migration onset (timeframe)'].transform(lambda s: s*t_res)
annot_full_path = Path('//allen/aics/assay-dev/users/Serge/zo1_analysis')
barcode_annotations_full = pd.read_csv(Path.joinpath(annot_full_path, 'annotations/zo1_to_seg_done.csv'))
barcode_annotations_full['migration onset (hours)'] = barcode_annotations_full['migration onset (timeframe)'].transform(lambda s: s*t_res)


if PLOT_DATA:
    fig, ax = plt.subplots()
    sns.stripplot(data=barcode_annotations, x='expt_condition', y='migration onset (hours)',
                hue='Plate Barcode', palette='bright', ax=ax)
    ax.tick_params(axis='x', rotation=10)
    fig.savefig(out_dir/'migration_onsets.pdf', bbox_inches='tight', dpi=300)

    plt.close('all')
    for nm,grp in barcode_annotations_full.groupby('Plate Barcode'):
        fig, ax = plt.subplots()
        sns.stripplot(data=grp.sort_values('expt_condition'),
                      x='expt_condition', y='migration onset (hours)',
                      c='k', ax=ax)
        ax.tick_params(axis='x', rotation=10)
        ax.set_xlabel('Colony Condition')
        ax.set_ylabel('Migration Onset (hours)')
        ax.set_title(f'Plate Barcode: {nm}')
    plots_to_multipdf(out_dir/'migration_onsets_full_by_barcode.pdf', tight=True, dpi=300)

    fig, ax = plt.subplots()
    box_colors = {'boxprops':{'facecolor':'none', 'edgecolor':'k'},
                  'medianprops':{'color':'k'},
                  'whiskerprops':{'color':'k'},
                  'capprops':{'color':'k'},
                  'flierprops':{'markerfacecolor':'k', 'markeredgecolor':'k'}}
    sns.boxplot(data=barcode_annotations_full.sort_values('expt_condition'), 
                    x='expt_condition', y='migration onset (hours)',
                    ax=ax, **box_colors)
    ax.tick_params(axis='x', rotation=10)
    ax.set_xlabel('Colony Condition')
    ax.set_ylabel('Migration Onset (hours)')
    fig.savefig(out_dir/f'migration_onsets_full.pdf', bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots()
    box_colors = {'boxprops':{'facecolor':'none', 'edgecolor':'k'},
                  'medianprops':{'color':'k'},
                  'whiskerprops':{'color':'k'},
                  'capprops':{'color':'k'},
                  'flierprops':{'markerfacecolor':'k', 'markeredgecolor':'k'}}
    sns.boxplot(data=barcode_annotations_full.sort_values('expt_condition'), 
                    x='expt_condition', y='migration onset (hours)',
                    ax=ax, **box_colors)
    sns.swarmplot(data=barcode_annotations_full.sort_values('expt_condition'), 
                  x='expt_condition', y='migration onset (hours)', marker='.', size=10,
                  hue='Plate Barcode', palette='bright', 
                  legend=False, ax=ax)
    ax.tick_params(axis='x', rotation=10)
    ax.set_xlabel('Colony Condition')
    ax.set_ylabel('Migration Onset (hours)')
    fig.savefig(out_dir/f'migration_onsets_full_v2.pdf', bbox_inches='tight', dpi=300)

    plt.show()
    plt.close('all')

else:
    pass



## Run a statistical tests on the migration timings of migrating groups for the full dataset
mig_timings_nmgrp = dict([(nm, grp.to_list()) for nm, grp in barcode_annotations_full.groupby('expt_condition')['migration onset (hours)']])
mig_timings = list(mig_timings_nmgrp.values())

ftest = f_oneway(*mig_timings)
tuk = tukey_hsd(*mig_timings)
print(tuk.statistic, '\n', tuk.pvalue)
[print(k, len(mig_timings_nmgrp[k])) for k in mig_timings_nmgrp]

barcode_annotations_full.groupby('expt_condition')['migration onset (hours)'].describe()

print('Done.')



## Test plots:
# SAC_subset = ntwrk_df.query('expt_condition == "2D-MG-EMT-1-60-MG" or expt_condition == "2D-PLF-EMT-1-60-MG" or expt_condition == "3D-MG-EMT-1-60-MG"')
