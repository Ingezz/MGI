import os
import matplotlib
#matplotlib.use('Agg')
import seaborn
import matplotlib.pyplot as plt
import numpy
import scipy
import scipy.stats
import pandas
import itertools

from . import toolbox

FIG_EXTS = ('png', 'pdf')
PPT_16x9_FULL_BOX = (11.5, 4.75)  # size to fill a content box for default layout of a 16x9 powerpoint slide
MARKERS = ['o', '^', 's', '*', 'D', 'v', 'p', 'h', '+', '<', '>', '8']
COLORS = ['k', 'w', 'r', 'b', 'y', 'g', 'violet', 'orange', 'gray', 'cyan', 'olive', 'pink']

def make_plot_series(sample_info_df, column_mappings):
    """
    Given a dataframe of sample information (sample names in rows, attributes in columns),
    and a dictionary mapping the color and marker properties to lists of 1 or more columns,
    
    e.g. {'marker':['Tissue'], 'color':['IL4', 'JQ1', 'KO']}
    
    return a list of dicitonaries containing plot attributes (as integers) and sample members.
    
    e.g.
    
    [{'color': 0,
      'label': 'Tissue=BMDM, IL4=0, JQ1=0.0, KO=Rac2',
      'marker': 0,
      'members': {'C57-BMDM-Rac2KO1-RNA-PolyA-DurdenLab-DGO-15-05-08',
      'C57-BMDM-Rac2KO2-RNA-PolyA-DurdenLab-DGO-15-05-08'}},
    {'color': 1,
      'label': 'Tissue=BMDM, IL4=0, JQ1=0.0, KO=Syk',
      'marker': 0,
      'members': {'C57_BMDM_SykKO1_Cre_RNA_PolyA_DurdenLab_DGO_15_10_09',
      'C57_BMDM_SykKO2_Cre_RNA_PolyA_DurdenLab_DGO_1_12_14'}}]
    """

    factor_subclasses = {}
    plot_properties = column_mappings.keys()
    for plot_property in column_mappings:
        factor_subclasses[plot_property] = {}
        for i, (key, group) in enumerate(sample_info_df.groupby(column_mappings[plot_property])):
            if len(column_mappings[plot_property]) == 1:
                key = [key]
            key = [str(k) for k in key]
            subclass_name = ', '.join(['='.join((k,v)) for k,v in zip(column_mappings[plot_property], key)])
            factor_subclasses[plot_property][i] = {'name':subclass_name, 'members':group.index}

    plot_properties = factor_subclasses.keys()

    subclass_indices = list(itertools.product(*[property_subclasses.keys() for property_subclasses in factor_subclasses.values()]))
    subclass_names = [', '.join([list(factor_subclasses.values())[i][t]['name'] for i, t in enumerate(subclass_tup)]) for subclass_tup in subclass_indices]

    series_properties = [dict([p, t] for p,t in zip(plot_properties, tup)) for tup in subclass_indices]
    for series, series_name in zip(series_properties, subclass_names):
        series['members'] = list(set.intersection(*[set(factor_subclasses[prop_name][prop_index]['members']) for prop_name, prop_index in series.items()]))
        series['label'] = series_name
       
    return [series for series in series_properties if series['members']]

    
def annotated_scatter(dataframe, x, y, z, sample_annotations, factor_mappings, fname_prefix='', markers=MARKERS, colors=COLORS,
            series_labels=[], fig_size=(10,5), legend_loc=(1.1,0), marker_size=60, first_two_only=False):
    """
    Creates a scatterplot using the data in the  :param:`x` and :param:`y` columns of :param:`dataframe`,
    setting colors and markers according to :param:`sample_annotations` and
    :param:`factor_mappings`.

    :param:`dataframe` should have row indices that match the sample_annotations.
    :sample_annotations: should be a dataframe with samples as rows and attributes as columns
    :factor_mappings: is a dictionary that maps the "marker" and "color" properties to lists of 1 or more columns.
    """
    seaborn.set_style('white')
    
    plot_series = make_plot_series(sample_annotations, factor_mappings)
    color_set = set([])
    marker_set = set([])
    
    for series in plot_series:
        color_set.add(series['color'])
        marker_set.add(series['marker'])
    
    # if we don't have enough colors then it's time to taste the rainbow
    if len(color_set) > len(colors):
        # we do the extra conversion to hex to avoid RGB triples being interpreted as a sequence of grayscale singletons when there are 3 elements to be plotted.
        colors = [matplotlib.colors.rgb2hex(matplotlib.colors.hsv_to_rgb([c/float(len(color_set)),1,1])) for c in range(len(color_set))]
    # make sure we have enough markers
    assert len(marker_set) <= len(MARKERS), 'Not enough markers defined to enumerate all the requested classes. Please pass a custom list of at least {} markers to use'.format(len(marker_set))

    dataframe = dataframe.loc[sample_annotations.index]

    fig, ax = plt.subplots(1,(2,1)[first_two_only], figsize=fig_size, sharey=True)
    
    if first_two_only:
        ax = [ax] # allows us to use the same indexing whether ax is a singleton or array

    for series_idx, series in enumerate(plot_series):
        if series['members']:
            if not series_labels: # kludgy way of overriding the legends for now
                label = series['label']
            else:
                label=series_labels[series_idx] 
            
            ax[0].scatter(dataframe.loc[series['members'], x], dataframe.loc[series['members'], y], marker=markers[series['marker']], c=colors[series['color']], s=marker_size, label=label)
            if not first_two_only:
                ax[1].scatter(dataframe.loc[series['members'], z], dataframe.loc[series['members'], y], marker=markers[series['marker']], c=colors[series['color']], s=marker_size, label=label)
    
    ax[0].set_xlabel('{}'.format(x))
    ax[0].set_ylabel('{}'.format(y))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    if first_two_only:
        ax[0].legend(loc=legend_loc, frameon=1, )
    else:
        ax[1].set_xlabel('{}'.format(z))
        ax[1].set_ylabel('{}'.format(y))
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].legend(loc=legend_loc, frameon=1, )
        
    if first_two_only:
        dataframe.drop(z, axis=1, inplace=True)
        
    dataframe.to_csv('{}.csv'.format(fname_prefix))
    if fname_prefix:
        for fig_ext in FIG_EXTS:
            fig.savefig('{}.{}'.format(fname_prefix, fig_ext), bbox_inches='tight', dpi=600)
            
    return fig    
    

def my_pca(dataframe, sample_annotations, factor_mappings, fname_prefix='', markers=MARKERS, colors=COLORS,
            series_labels=[], fig_size=(12,6), legend_loc=(1.1,0), marker_size=60, first_two_only=False,
            sklearn_pca=None):
    """
    Plots the first three PCs of :dataframe:
    
    :sample_annotations: should be a dataframe with samples as rows and attributes as columns
    :factor_mappings: is a dictionary that maps the "marker" and "color" properties to lists of 1 or more columns.
    :param:`sklearn_pca` will use this to transform the data if provided, otherwise will generate one training on all the data
    """
    import sklearn.decomposition
    seaborn.set_style('white')
    
    plot_series = make_plot_series(sample_annotations, factor_mappings)
    color_set = set([])
    marker_set = set([])
    
    for series in plot_series:
        color_set.add(series['color'])
        marker_set.add(series['marker'])
    
    # if we don't have enough colors then it's time to taste the rainbow
    if len(color_set) > len(colors):
        # we do the extra conversion to hex to avoid RGB triples being interpreted as a sequence of grayscale singletons when there are 3 elements to be plotted.
        colors = [matplotlib.colors.rgb2hex(matplotlib.colors.hsv_to_rgb([c/float(len(color_set)),1,1])) for c in range(len(color_set))]
    # make sure we have enough markers
    assert len(marker_set) <= len(MARKERS), 'Not enough markers defined to enumerate all the requested classes. Please pass a custom list of at least {} markers to use'.format(len(marker_set))

    dataframe = dataframe.loc[:, sample_annotations.index]
    
    if not sklearn_pca:
        sklearn_pca = sklearn.decomposition.PCA(n_components=3)
        sklearn_pca.fit(dataframe.T)
       
    sklearn_transf = sklearn_pca.transform(dataframe.T)

    explained_variance = sklearn_pca.explained_variance_ratio_

    pca_df = pandas.DataFrame(sklearn_transf, columns=['PC_1', 'PC_2', 'PC_3'], index=dataframe.columns)

    fig, ax = plt.subplots(1,(2,1)[first_two_only], figsize=fig_size, sharey=True)
    
    if first_two_only:
        ax = [ax] # allows us to use the same indexing whether ax is a singleton or array

    for series_idx, series in enumerate(plot_series):
        if series['members']:
            if not series_labels: # kludgy way of overriding the legends for now
                label = series['label']
            else:
                label=series_labels[series_idx] 
            
            ax[0].scatter(pca_df.loc[series['members'], 'PC_1'], pca_df.loc[series['members'], 'PC_2'], marker=markers[series['marker']], c=colors[series['color']], s=marker_size, label=label)
            if not first_two_only:
                ax[1].scatter(pca_df.loc[series['members'], 'PC_3'], pca_df.loc[series['members'], 'PC_2'], marker=markers[series['marker']], c=colors[series['color']], s=marker_size, label=label)
    
    ax[0].set_xlabel('PC 1 ({:.2})'.format(explained_variance[0]))
    ax[0].set_ylabel('PC 2 ({:.2})'.format(explained_variance[1]))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    if first_two_only:
        ax[0].legend(loc=legend_loc, frameon=1, )
    else:
        ax[1].set_xlabel('PC 3 ({:.2})'.format(explained_variance[2]))
        ax[1].set_ylabel('PC 2 ({:.2})'.format(explained_variance[1]))
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].legend(loc=legend_loc, frameon=1, )
    if first_two_only:
        pca_df.drop('PC_3', axis=1, inplace=True)
    pca_df.to_csv('{}.csv'.format(fname_prefix))
    if fname_prefix:
        for fig_ext in FIG_EXTS:
            fig.savefig('{}.{}'.format(fname_prefix, fig_ext), bbox_inches='tight', dpi=600)
            
    return fig
    

def my_nmf(dataframe, sample_annotations, factor_mappings, fname_prefix='', markers=MARKERS, colors=COLORS, series_labels=[], fig_size=(12,6), legend_loc=(1.1,0), marker_size=60):
    """
    Factor :dataframe: and plots the sample membership in the three eigenfeatures.
    
    :sample_annotations: should be a dataframe with samples as rows and attributes as columns
    :factor_mappings: is a dictionary that maps the "marker" and "color" properties to lists of 1 or more columns.
    """
    import sklearn.decomposition
    seaborn.set_style('white')
    
    plot_series = make_plot_series(sample_annotations, factor_mappings)
    color_set = set([])
    marker_set = set([])
    
    for series in plot_series:
        color_set.add(series['color'])
        marker_set.add(series['marker'])
    
    # if we don't have enough colors, time to taste the rainbow
    if len(color_set) > len(colors):
        # we do the extra conversion to hex to avoid RGB triples being interpreted as a sequence of grayscale singletons when there are 3 elements to be plotted.
        colors = [matplotlib.colors.rgb2hex(matplotlib.colors.hsv_to_rgb([c/float(len(color_set)),1,1])) for c in range(len(color_set))]

    dataframe = dataframe.loc[:, sample_annotations.index]
    sklearn_nmf = sklearn.decomposition.NMF(n_components=3)
    sklearn_transf = sklearn_nmf.fit_transform(dataframe.T)

    #explained_variance = sklearn_nmf.explained_variance_ratio_

    nmf_df = pandas.DataFrame(sklearn_transf, columns=[1,2,3], index=dataframe.columns)

    fig, ax = plt.subplots(1,2, figsize=fig_size, sharey=True)

    for series_idx, series in enumerate(plot_series):
        if series['members']:
            if not series_labels: # super-kludgy way of fixing up the legends for now
                label = series['label']
            else:
                label=series_labels[series_idx] 
            ax[0].scatter(nmf_df.loc[series['members'], 1], nmf_df.loc[series['members'], 2], marker=markers[series['marker']], c=colors[series['color']], s=marker_size, label=label)
            ax[1].scatter(nmf_df.loc[series['members'], 3], nmf_df.loc[series['members'], 2], marker=markers[series['marker']], c=colors[series['color']], s=marker_size, label=label)
    
    ax[0].set_xlabel('Metafeature 1')
    ax[0].set_ylabel('Metafeature 2')
    ax[1].set_xlabel('Metafeature 3')
    ax[1].set_ylabel('Metafeature 2')
       
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
       
    ax[1].legend(loc=legend_loc, frameon=1, )
    
    if fname_prefix:
        for fig_ext in FIG_EXTS:
            fig.savefig('{}.{}'.format(fname_prefix, fig_ext), bbox_inches='tight', dpi=600)
            
    return fig   


def my_mds(dataframe, sample_annotations, factor_mappings, fname_prefix='', markers=MARKERS, colors=COLORS, series_labels=[], fig_size=(6,6), legend_loc=(1.1,0), marker_size=60):
    """
    Plots the first three PCs of :dataframe:
    
    :sample_annotations: should be a dataframe with samples as rows and attributes as columns
    :factor_mappings: is a dictionary that maps the "marker" and "color" properties to lists of 1 or more columns.
    """
    import sklearn.manifold
    seaborn.set_style('white')
    
    plot_series = make_plot_series(sample_annotations, factor_mappings)
    color_set = set([])
    marker_set = set([])
    
    for series in plot_series:
        color_set.add(series['color'])
        marker_set.add(series['marker'])
    
    # if we don't have enough colors, time to taste the rainbow
    if len(color_set) > len(colors):
        # we do the extra conversion to hex to avoid RGB triples being interpreted as a sequence of grayscale singletons when there are 3 elements to be plotted.
        colors = [matplotlib.colors.rgb2hex(matplotlib.colors.hsv_to_rgb([c/float(len(color_set)),1,1])) for c in range(len(color_set))]

    dataframe = dataframe.loc[:, sample_annotations.index]
    sklearn_mds = sklearn.manifold.MDS(n_components=2)
    sklearn_transf = sklearn_mds.fit_transform(dataframe.T)

    mds_df = pandas.DataFrame(sklearn_transf, columns=['X', 'Y'], index=dataframe.columns)

    fig, ax = plt.subplots(1,1, figsize=fig_size, sharey=True)

    for series_idx, series in enumerate(plot_series):
        if series['members']:
            if not series_labels: # super-kludgy way of fixing up the legends for now
                label = series['label']
            else:
                label=series_labels[series_idx] 
            ax.scatter(mds_df.loc[series['members'], 'X'], mds_df.loc[series['members'], 'Y'], marker=markers[series['marker']], c=colors[series['color']], s=marker_size, label=label)
    
       
    ax.set_xticks([])
    ax.set_yticks([])

       
    ax.legend(loc=legend_loc, frameon=1, )
    
    if fname_prefix:
        for fig_ext in FIG_EXTS:
            fig.savefig('{}.{}'.format(fname_prefix, fig_ext), bbox_inches='tight', dpi=600)
            
    return fig    
 

def my_violinplot(data, data_names, ax=None, **kwargs):
    """
    Wrapper for seaborn's violinplot that allows unequal-length vectors to be compared.
    """
    row_size = max([len(x) for x in data])
    combined_df = pandas.DataFrame(numpy.full(shape=(row_size, len(data)), fill_value=numpy.nan), columns=data_names)
    for col, data_set in enumerate(data):
        combined_df.iloc[:len(data_set),col] = list(data_set) # need to convert to list because it will attempt to align indexes for pandas.Series and all we care about are the values
    seaborn.violinplot(data=combined_df, ax=ax, **kwargs)
            
            
def plot_cc(start, signal, peak_locations=[], fname=''):
    """
    Designed for plotting cross-correlation values. Given a segment of the cross-correlation signal and the coordinate of its
    start point, it will plot the signal strength and denote the peak location.
    """
    end = start + len(signal)
    seaborn.set_style('whitegrid')
    fig, ax = plt.subplots(1)
    xs = range(start + 1, end + 1)
    ax.plot(xs, signal)
    ax.set_xlim(start, end)
    ax.set_xlabel('Fragment length (bp)', fontsize=14)
    ax.set_ylabel('Cross-correlation', fontsize=14)

    if not peak_locations:
        peak_locations = [numpy.argmax(signal) + start + 1]

    for peak in peak_locations:
        peak_line = matplotlib.lines.Line2D(xdata=(peak, peak), ydata=(0, ax.get_ylim()[1]), linestyle="--", color='r')
        ax.add_line(peak_line)

    text_x = start + (end - start) * 0.95
    text_y = ax.get_ylim()[1] * 0.9

    ax.text(x=text_x, y=text_y, s='Peaks: {}'.format(', '.join([str(p + 1) for p in peak_locations])),
            horizontalalignment='right', fontsize=14)
    if fname:
        for fig_ext in FIG_EXTS:
            fig.savefig('{}.{}'.format(fname, fig_ext), bbox_inches='tight', dpi=600)


def my_bars(df, name_column='', fname_prefix='', title='', ylim=None, ylabel=''):
    """
    Given a DataFrame, will produce a bar graph for each column
    Bars will be labeled with the values in <name_column>. If <name_column>
    is not given, use the index
    """

    X_PAD = 0.05
    bar_width = (1 - X_PAD * len(df.index)) / len(df.index)
    TICKS_FONT = matplotlib.font_manager.FontProperties(family='arial', style='italic', size=12, weight='normal',
                                                        stretch='normal')

    col_list = [col for col in df.columns if col != name_column]

    fig, ax = plt.subplots(1, len(col_list), figsize=(3 * len(col_list), 3))

    max_value = max([df[col].max() for col in df.columns])
    min_value = min([df[col].min() for col in df.columns])

    if min_value >= 0 and max_value <= 1:
        y_lim = (0, 1)
    else:
        if min_value < 0:
            y_lim = (min_value * 1.1, max_value * 1.1)
        else:
            y_lim = (0, max_value * 1.1)

    for i, col in enumerate(sorted(col_list)):
        data_series = df[col].dropna()

        centers = numpy.linspace(X_PAD, 1 - X_PAD, num=len(data_series) + 2)[1:-1]
        lefts = [c - (bar_width / 2) for c in centers]

        ax[i].bar(lefts, data_series, width=bar_width)

        ax[i].set_ylim(*y_lim)
        ax[i].set_xlim(0, 1)
        ax[i].set_xticks(centers)

        if name_column:
            ax[i].set_xticklabels([df.ix[d, name_column] for d in data_series.index])
        else:
            ax[i].set_xticklabels(list(data_series.index))

        for l in ax[i].get_xticklabels():
            l.set_rotation(270)
            l.set_font_properties(TICKS_FONT)

        ax[i].grid(b=0, which='major', axis='x')

        if i == 0:
            ax[i].set_ylabel(ylabel)
        else:
            ax[i].set_yticklabels([])

        ax[i].set_title(col.split('_')[0])

    fig.text(x=0.5, y=1.2, s=title, ha='center', fontsize=20)

    if fname_prefix:
        for fig_ext in FIG_EXTS:
            fig.savefig('{}.{}'.format(fname_prefix, fig_ext), bbox_inches='tight', dpi=600)
    return fig


def kde_plot(data, ax=None):
    """
    Not really needed now that seaborn includes a similar function
    """
    xs = numpy.linspace(data.min(), data.max())
    if not ax:
        fig, ax = plt.subplots(1)
    ax.plot(xs, scipy.stats.gaussian_kde(data)(xs))
    return ax


def my_hist2d(xs, ys, x_title='', y_title='', binsizes=[], x_lim=None, y_lim=None, norm=matplotlib.colors.LogNorm(),
              cmap='binary',
              fname_prefix='', PCC=None, overall_title='', show_PCC=True, show_title=True, fig_size=(8, 6),
              fig_exts=FIG_EXTS):
    seaborn.set_style('white')
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    max_x = max(xs)
    max_y = max(ys)
    min_x = min(xs)
    min_y = min(ys)

    if not x_lim:
        x_lim = (min_x, max_x)
    if not y_lim:
        y_lim = (min_y, max_y)

    x_span = x_lim[1] - x_lim[0]
    y_span = y_lim[1] - y_lim[0]

    if not binsizes:
        binsizes = (toolbox.iround(max_x), toolbox.iround(max_y))

    ax.hist2d(xs, ys, bins=binsizes, range=(x_lim, y_lim), norm=norm, cmap=cmap)
    ax.set_xlabel(x_title, fontsize=16)
    ax.set_ylabel(y_title, fontsize=16)

    if show_PCC:
        if not PCC:
            PCC = scipy.stats.pearsonr(xs, ys)[0]
        ax.text(x_span * 0.75 + min_x, y_span * 0.1 + min_y, s='PCC: {:.3}'.format(PCC), fontsize=16)

    if show_title:
        if overall_title:
            fig.suptitle(overall_title, fontsize=18)
        elif x_title and y_title:
            fig.suptitle('{} vs. {}'.format(x_title, y_title), fontsize=18)

    if fname_prefix:
        for fig_ext in fig_exts:
            print('Saving figure as {}'.format('{}.{}'.format(fname_prefix, fig_ext)))
            fig.savefig('{}.{}'.format(fname_prefix, fig_ext), bbox_inches='tight', dpi=600)
    return fig


def simple_profile(profile, fname_prefix='', color='k', fig_exts=('png', 'pdf')):
    # seaborn.set_style('white')
    fig, ax = plt.subplots(1, 1, figsize=(10, 1.2), facecolor='white', frameon=False)
    ax.bar(list(range(len(profile))), profile, width=1.0, linewidth=0, color=color)
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if fname_prefix:
        for fig_ext in fig_exts:
            fig.savefig('{}.{}'.format(fname_prefix, fig_ext), bbox_inches='tight', dpi=600)
    return fig


def wig_plot(input_data, fname='', figsize=(6, 2), frame_on=False, color='k'):
    seaborn.set_style('white')
    size = len(input_data)
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white', frameon=frame_on)
    max_height = max(input_data)
    ax.bar(list(range(size)), input_data, width=1.0, linewidth=0, color=color)
    # ax.set_frame_on(frame_on)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if fname:
        for fig_ext in FIG_EXTS:
            fig.savefig(os.path.join(IMG_OUTPATH, '{}.{}'.format(fname, fig_ext)), bbox_inches='tight', dpi=600)
            
def my_scatter_matrix(df):
    axs = pandas.scatter_matrix( df, alpha=0.2, diagonal='kde')
    n = len(df.columns)
    for x in range(n):
        for y in range(n):
            # to get the axis of subplots
            ax = axs[x, y]
            # to make x axis name vertical  
            ax.xaxis.label.set_rotation(90)
#             ax.xaxis.labelpad = 50
            # to make y axis name horizontal 
            ax.yaxis.label.set_rotation(0)
            # to make sure y axis names are outside the plot area
            ax.yaxis.labelpad = 50
            
def forward_selection_evaluation(best_features, feature_scores, score_label='R^2'):
    feature_s = pandas.Series(feature_scores)
    feature_s.name=score_label
    feature_s.sort()

    fig, ax = plt.subplots(1, figsize=(10,5))
    seaborn.barplot(x=best_features, y=feature_s, ax=ax, color=(0.4,0.4,0.9))
    d = ax.set_xticklabels(best_features, rotation=45)
    return fig
    

def q_q_normal(data):
    """
    Given a vector of observations, generate a q-q plot of the observed
    values against the matching quantiles from a normal distribution having
    a mean and standard deviation matching the sample.
    """
    data=pandas.Series(sorted(data), name='Observed data')
    this_normal = scipy.stats.norm(loc=data.mean(), scale=data.std()) # should we use sample SD here?
    quantiles = (numpy.arange(len(data)) + 1) / (len(data) + 1)
    norm_vals = pandas.Series(this_normal.ppf(quantiles), name='Normal distribution')
    
    seaborn.jointplot(norm_vals, data, kind='regplot' )