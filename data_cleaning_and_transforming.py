import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('figure', figsize=(15.0, 5.0))

def string_to_numeric(value, replace_nan = np.nan, br_format = False):
    '''
    Casts string value to a numeric type using pandas.to_numeric() function.
    If string value is in brazilian format (ex: 1.234,56), first transform it to international format (ex: 1234.56).
    If value cannot be cast to numeric format, return a replacement value.
    '''
    
    if pd.isnull(value):
    	return replace_nan

    elif br_format:
    	value = value.replace('.', '')
    	value = value.replace(',', '.')

    	try:
    		value =  pd.to_numeric(value)
    	except:
    		value = -1

    return value

def custom_to_datetime(series):
	dates_hash = {}
	unique_dates = series.unique()

	for date in unique_dates:
		try:
			transformed_datetime = pd.to_datetime(date, dayfirst=True)
		except:
			print("Não foi possível converter o valor '{}' para formato datetime.".format(date))
			transformed_datetime = date

		dates_hash[date] = transformed_datetime

	return series.map(dates_hash)

def custom_to_ordered_category(series):
	series_in_numerical_format = series.apply(string_to_numeric, args = (np.nan, True))
	ordered_categs = series_in_numerical_format.unique().astype(int)
	ordered_categs.sort()

	cat_type = CategoricalDtype(categories=ordered_categs, ordered=True)
	return series_in_numerical_format.astype(cat_type)

# Seaborn distplot template
def upper_rugplot(data, height=.05, ax=None, **kwargs):
    from matplotlib.collections import LineCollection
    ax = ax or plt.gca()
    kwargs.setdefault("linewidth", 1)
    segs = np.stack((np.c_[data, data],
                     np.c_[np.ones_like(data), np.ones_like(data)-height]),
                    axis=-1)
    lc = LineCollection(segs, transform=ax.get_xaxis_transform(), **kwargs)
    ax.add_collection(lc)

def custom_distplot(data, color, bin_step = 1, rug=False):
    #fig, ax = plt.subplots()
    ax = sns.distplot(
        a = data,
        bins=np.arange(int(data.min()), int(data.max()), bin_step),
        hist=True,
        kde=False,
        rug=rug,
        fit=None,
        hist_kws=None,
        kde_kws=None,
        rug_kws=None,
        fit_kws=None,
        color=color,
        vertical=False,
        norm_hist=True,
        axlabel=None,
        label=None,
        ax=None
    )
    
    upper_rugplot(data, ax=ax, color=color, alpha=0.6)
