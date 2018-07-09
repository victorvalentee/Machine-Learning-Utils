# Biblioteca de funções compartilhadas

# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
# --------

# Formatting (cleaning and casting)
def montante_to_numeric(x):
    '''String to numeric.'''
    x = x.strip()
    x = x.replace('.','')
    x = x.replace(',','.')

    if '-' in x:
        x = x.replace('-','')
        x = '-' + x
        
    try:
        x = float(x)
    except:
        x = np.nan

    return x

def lookup_dates(s, format = '%d.%m.%Y'):
    '''uses .map() to apply changes'''
    dates = {date:pd.to_datetime(date, errors='coerce', format=format) for date in s.unique()}
    return s.map(dates)
# --------

# Splitting
	# train_test_split
	# train_test_split_by_date
# --------

# Preprocessing
def scale_0_1(series):
	# scale any value between 0 and 1
    series_min = series.min()
    series_max = series.max()
    return (series - series_min) / (series_max - series_min)

	# Feature Generation
	# Feature Selection
# --------

# Encoding
	# LabelEnconder
	# One Hot Encoding
# --------

# Training / Testing
	# sklearn.model_selection
	# LGB
	# XGB
	# NN
# --------

# Evaluating
def find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr+(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[-1:]]
    
    return roc_t['threshold']

def plotRocCurve(y_test, classifiers):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for (name, probs) in classifiers:
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %s (area = %0.2f)' % (name, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CV ROC curve')
    plt.legend(loc="lower right")
    fig = plt.gcf()
    fig.set_size_inches(15,5)

    plt.show()

def ks(pred, real, thr = 0.5):
	pred = pred.reshape((-1))
    scorecard_all = np.vstack((pred, real)).T
    scorecard_all = scorecard_all[scorecard_all[:, 0].argsort()]
    count_badclient = 0
    count_goodclient = 0
    percent_badclient = np.zeros(scorecard_all.shape[0])
    percent_goodclient = np.zeros(scorecard_all.shape[0])
    score_range = []
    score_std = scorecard_all[:,0].std()
    
    for i in range(scorecard_all.shape[0]):
        if scorecard_all[i,1] > thr:
            count_goodclient +=1
        else: 
            count_badclient +=1
        percent_badclient[i] = count_badclient
        percent_goodclient[i] = count_goodclient
    ks = 0
    ks_pos = 0
    diffs = np.zeros(scorecard_all.shape[0])
    for i in range(scorecard_all.shape[0]):
        percent_badclient[i]  /= count_badclient
        percent_goodclient[i] /= count_goodclient
        diffs[i] = percent_badclient[i] - percent_goodclient[i]
        if diffs[i] > ks:
            ks = diffs[i]
            ks_pos = i
    print('Scorecard standard deviation:', score_std)
    fig2 = plt.figure(2,figsize=(10, 8))
    ax = fig2.add_subplot(111)
    plt.plot(list(scorecard_all[:, 0]),percent_goodclient, 'b', 
             list(scorecard_all[:, 0]),diffs, 'g',
             list(scorecard_all[:, 0]),percent_badclient, 'r',)
    plt.plot([scorecard_all[ks_pos,0],scorecard_all[ks_pos,0]], 
             [percent_badclient[ks_pos], percent_goodclient[ks_pos]], 'g', linestyle = '--')
    ax.text(0.05, 0.95,'Greenline: KS = '+ str(ks), ha='left', va='center', color='green', transform=ax.transAxes)
    ax.text(0.05, 0.90,'Blueline: Good', ha='left', va='center', color='blue', transform=ax.transAxes)
    ax.text(0.05, 0.85,'Redline: Bad', ha='left', va='center', color='red', transform=ax.transAxes)
    plt.ylabel('Cumulative percentage')
    plt.xlabel('Score')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    fig2.suptitle('KS graph')
    result_str = str(int(round(ks*1000)))
    plt.draw()

def generate_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion_matrix(y_test, y_pred):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #plt.figure()
    fig, ax1 = plt.subplots()
    ax1.xaxis.grid(False)
    ax1.yaxis.grid(False)
    generate_confusion_matrix(cnf_matrix, classes=['MAU', 'BOM'],
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    #plt.figure()
    fig, ax1 = plt.subplots()
    ax1.xaxis.grid(False)
    ax1.yaxis.grid(False)
    generate_confusion_matrix(cnf_matrix, classes=['MAU', 'BOM'], normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

def debtor_distribution(scores, real):
    if max(scores) >= 10:
        scores = [float(s) / 100 for s in scores]

    debtors_per_range = [0. for i in range(10)]
    clients_per_range = [0. for i in range(10)]
    clients_total = len(real)
    
    for i in range(clients_total):
        score_range = min(int(10 * scores[i]), 9)
        if real[i] == 0:
            debtors_per_range[score_range] += 1 
        clients_per_range[score_range] += 1
        
    importance_list = []
    noncomplice_rate_list = []
        
    acm = 0.
    print '%d clients, (%.2f %% debtors)' % (clients_total, 100.*sum(debtors_per_range)/clients_total)
    print 'score range\tdebtors(%)\tfrequency(%)' 
    for i in range(10):
        rel_debtors_per_range = 0 if clients_per_range[i] == 0 else debtors_per_range[i]/clients_per_range[i]
        rel_clients_per_range = 0 if clients_total == 0 else clients_per_range[i]/clients_total
        acm += rel_clients_per_range
        print '%3d - %3d\t%10.2f\t%12.2f\t%2.2f' % (i*10, (i+1)*10, 
                                           100.*rel_debtors_per_range, 
                                           100.*rel_clients_per_range, acm)
        noncomplice_rate_list.append(100.*rel_debtors_per_range)
        importance_list.append(100.*rel_clients_per_range)
    
    return importance_list, noncomplice_rate_list

def create_distribution_plot(importance_list, noncomplice_rate_list, 
	titulo = u'Distribuição de inadimplência', fontsize=20, 
	y_max_imp=0, y_max_rate=0):
    
    fig, ax1 = plt.subplots()
    t = np.arange(0.01, 10.0, 1)
    s1 = importance_list
    ax1.bar(t, s1, align='center')
    ax1.set_xlabel('Faixas de scores', fontsize=fontsize)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel(u'Representatividade (%)', color='b', fontsize=fontsize)
    ax1.tick_params('y', colors='b', labelsize=fontsize)
    labels = ['0-10',  '10-20', '20-30', '30-40', '40-50', 
              '50-60', '60-70', '70-80', '80-90', '90-100']

    rects = ax1.patches
    for rect, label in zip(rects, s1):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2, height-2, '%.2f'%label, ha='center', va='bottom', 
                 color='white', fontsize=20)

    plt.xticks(t, labels, rotation='horizontal', fontsize=fontsize)
    plt.text(0.5, 1.04, titulo, horizontalalignment='center', fontsize=(fontsize+10), transform = ax1.transAxes)
    
    #ax1.set_xlim([xmin,xmax])
    if y_max_imp!=0:
        ax1.set_ylim([0,y_max_imp+2])

    ax2 = ax1.twinx()
    s2 = noncomplice_rate_list
    ax2.plot(t, s2, ls=':', marker='o', c='red', ms=1)
    ax2.set_ylabel(u'Taxa de inadimplência (%)', color='r', fontsize=fontsize)
    ax2.tick_params('y', colors='r', labelsize=fontsize)
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)
    
    if y_max_rate!=0:
        ax2.set_ylim([0,y_max_rate+2])

    for xy in zip(t, s2):
        x = xy[0]
        y = xy[1]
        ax2.annotate('%.2f' % xy[1], xy=(x, y), textcoords='data', fontsize=fontsize, ha='center', 
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.6)) # <--

    #fig.tight_layout()
    fig.set_size_inches(18.5, 10.5)
    #plt.tight_layout()
    plt.margins(x=0.02, y=0.1)
    plt.show()

    fig.savefig('inad.png', dpi=100)
# --------

# Misc
from sklearn.externals import joblib

# Save
def save_model(model, model_name):
    joblib.dump(model, model_name + ".pkl")

# Load
def load_model(path_to_saved_model):
    loaded_model = joblib.load(path_to_saved_model)
    return loaded_model
# --------
