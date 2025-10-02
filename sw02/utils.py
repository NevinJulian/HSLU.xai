import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, norm

def standardize_residuals(residuals):
    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals)
    return (residuals - mean_resid) / std_resid

def residual_analysis_plots(standardized_residuals, fitted_values=None):
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 2, figure=fig)
    ax_time = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_qq = fig.add_subplot(gs[1, 1])
    axs = [ax_time, ax_hist, ax_qq]
    for ax in axs:
        ax.grid(True)

    if fitted_values is None:
        fitted_values = np.arange(len(standardized_residuals))
    else:
        ax_time.set_xlabel("Fitted values")
    ax_time.scatter(
        fitted_values,
        standardized_residuals,
    )
    ax_time.set_title('Standardized Residuals')

    ax_hist.hist(standardized_residuals, density=True, label='Hist', edgecolor='#FFFFFF')
    kde = gaussian_kde(standardized_residuals)
    xlim = (-1.96 * 2, 1.96 * 2)
    x = np.linspace(xlim[0], xlim[1])
    ax_hist.plot(x, kde(x), label='KDE')
    ax_hist.plot(x, norm.pdf(x), label='N(0,1)')
    ax_hist.legend()
    ax_hist.set_title('Histogram with estimated density')

    sm.qqplot(standardized_residuals, line='s', ax=ax_qq)
    ax_qq.set_title(f'Normal Q-Q')

    fig.tight_layout()

