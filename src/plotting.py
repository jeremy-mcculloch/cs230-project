import matplotlib.pyplot as plt
from src.CANN.cont_mech import *
from src.CANN.models import *
import matplotlib
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator


## Override default matplotlib setting so plots look better
# plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.family"] = "Source Sans 3"

# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['xtick.labelsize'] = 40
plt.rcParams['ytick.labelsize'] = 40
plt.rcParams['xtick.minor.size'] = 7
plt.rcParams['ytick.minor.size'] = 7
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.labelsize'] = 40

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.pad'] = 14
plt.rcParams['ytick.major.pad'] = 14
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.titlesize'] = 40
plt.rcParams["axes.titleweight"] = "bold"

plt.rcParams["figure.titlesize"] = 40
plt.rcParams["figure.titleweight"] = "extra bold"
matplotlib.rcParams['legend.handlelength'] = 1
matplotlib.rcParams['legend.handleheight'] = 1
# matplotlib.rcParams['legend.fontsize'] = 60


def plot_bcann(stretches, stresses, model_given, lam_ut_all, terms, id, modelFit_mode, plot_dist=True, blank=False):

    #### Get model Predictions ####
    # Means and variances
    model_weights = model_given.get_weights()
    if blank:
        model_given.set_weights([x * 0.0 for x in model_weights])
    Stress_predict_UT = model_given.predict(lam_ut_all).reshape((-1, 2, 2, 2))  # N x 8 (45, xy, mean/var)

    stress_pred_mean = np.array(Stress_predict_UT[:, :, :, 0]).transpose((1, 0, 2)).reshape((2, 5, 5, -1, 2))[:, 0, :, :,
                      :]  # 2 x 5 x 100 x 2( x/y)
    stress_pred_std = np.sqrt(
        np.array(Stress_predict_UT[:, :, :, 1]).transpose((1, 0, 2)).reshape((2, 5, 5, -1, 2)))[:, 0, :, :,
                          :]  # 2 x 5 x 100 x 2( x/y)

    # Predictions by term
    n_terms = len(model_weights) // 2
    stress_pred_terms = np.zeros(shape = stress_pred_mean.shape + (n_terms + 1,))
    for i in range(n_terms):
        temp_weights = [model_weights[j] * (i == j // 2 or j==n_terms* 2) for j in range(n_terms * 2 + 1)]
        model_given.set_weights(temp_weights)
        temp_preds = model_given.predict(lam_ut_all).reshape((-1, 2, 2, 2))
        stress_pred_terms[:, :, :, :, i + 1] = stress_pred_terms[:, :, :, :, i] + np.array(temp_preds[:, :, :, 0]).transpose((1, 0, 2)).reshape((2, 5, 5, -1, 2))[:, 0, :, :, :]
    model_given.set_weights(model_weights)

    plot_bcann_raw_data(stretches, stresses, stress_pred_mean, stress_pred_std, stress_pred_terms, terms, id, modelFit_mode, blank, plot_dist)

def plot_bcann_raw_data(stretches, stresses, stress_pred_mean, stress_pred_std, stress_pred_terms, terms, id, modelFit_mode, blank, plot_dist):
    i_dev = 9
    i_test = 13

    stretch_plot = stretches.reshape((2, 5, 5, -1, 2))[0, 0, :, :, :]  # 5 x 100 x 2
    stress_true = stresses.reshape((2, 5, 5, -1, 2)) # 2 x 5(n_ex) x 5 x 100 x 2( x/y)

    # Plot Best Fit
    plt.rcParams['text.usetex'] = False
    plt.rcParams['figure.figsize'] = [30, 20 if terms else 15]
    plt.rcParams['figure.constrained_layout.use'] = True



    fig, axes = plt.subplots(3, 5)
    axes = flatten([list(x) for x in zip(*axes[0:2])]) + list(axes[2])
    direction_strings = ["w", "s"] * 5 + ["x", "y", "x", "y", "x"]
    n_spaces = 55
    titles = ["strip-w", None, "off-w", None, "equibiax-ws", None, "off-s", None, "strip-s", None, " " * n_spaces + "strip-x", None," " * n_spaces + "off-x", None, "equibiax-xy"]
    inputs = [[stretch_plot[j, :, k] for k in range(2)] for i in range(2) for j in range(5) ]
    inputs = [[x[i] if np.max(x[i]) > 1.0 else ((x[1 - i] - 1) * 1e-9 + 1) for i in range(2)] for x in inputs]
    inputs = flatten(inputs)[0:15] # 100 len each
    outputs = [stress_true[i, :, j, :, k] for i in range(2) for j in range(5) for k in range(2)][0:15] # 5 x 100 each
    if terms:
        pred_terms = [stress_pred_terms[i, j, :, k, :] for i in range(2) for j in range(5) for k in range(2)][0:15]
    pred_mean = [stress_pred_mean[i, j, :, k] for i in range(2) for j in range(5) for k in range(2)][0:15]
    pred_std = [stress_pred_std[i, j, :, k] for i in range(2) for j in range(5) for k in range(2)][0:15]
    cmap = plt.cm.get_cmap('jet_r', 17)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]

    num_points = 17
    train_loss = 0
    for i in range(len(axes)):
        if terms:
            n_terms = stress_pred_terms.shape[-1] - 1
            for j in range(n_terms):
                # Create plot that fills between the lower and upper bound
                axes[i].fill_between(inputs[i], pred_terms[i][:, j], pred_terms[i][:, j + 1], lw=0,
                                     zorder=j + 1, color=cmaplist[j],
                                     label=j + 1)
                axes[i].plot(inputs[i], pred_terms[i][:, j + 1],  lw=0.4, zorder=23, color='k')
        else:
            # Create plot that fills between the lower and upper bound
            axes[i].fill_between(inputs[i], pred_mean[i] - pred_std[i], pred_mean[i] + pred_std[i], lw=0, zorder=0, color="#384ebc", alpha = 0.25,
                             label=i + 1)
            eps = 1e-6
            # Compute negative log likelihood for a normal distribution
            errors = 0.5 * (np.log(2 * np.pi * (pred_std[i] ** 2 + eps)) + (outputs[i][:, :] - pred_mean[i]) ** 2 / (
                        pred_std[i] ** 2 + eps)) # Result should be 5 x 100
            nll = np.mean(errors)
            if i == i_dev or i == i_test:
                print(("Test" if i == i_test else "Dev") + f" Loss: {nll}")
            else:
                train_loss += nll

        # Create plot that draws a line on the upper bound
        axes[i].plot(inputs[i], pred_mean[i], lw=4, zorder=24, color='k')

        min_P = np.min(outputs[i])
        max_P = np.max(outputs[i])
        min_x = np.min(inputs[i])
        max_x = np.max(inputs[i])
        if np.max(inputs[i]) - np.min(inputs[i]) < 1e-6:
            axes[i].set_xticks([np.min(inputs[i]), np.max(inputs[i])])
            axes[i].set_xticklabels(['1', '1'])

        axes[i].set_xlim([min_x, max_x])
        axes[i].set_ylim([0.0, max_P])


        if inputs[i].shape[0] > num_points:
            input_old = inputs[i]
            inputs[i] = np.linspace(np.min(input_old), np.max(input_old), num_points)
            outputs[i] = np.array([np.interp(inputs[i], input_old, outputs[i][j, :]) for j in range(5)])

        if terms or not plot_dist:
            for j in range(5):
                scatterplot = axes[i].scatter(inputs[i], outputs[i][j, :], s=300, zorder=25, lw=3, facecolors='w', edgecolors='k',
                                            clip_on=False)
        else:
            data_mean = np.mean(outputs[i], axis=0)
            data_std = np.std(outputs[i], axis=0)
            data_std_sample = np.std(outputs[i], axis=0, ddof=1)
            errors = 0.5 * (np.log(2 * np.pi * (data_std ** 2 + eps)) + (outputs[i][:, :] - data_mean) ** 2 / (
                    data_std ** 2 + eps))  # Result should be 5 x 100
            nll_min = np.mean(errors)
            axes[i].fill_between(inputs[i], data_mean - data_std_sample, data_mean + data_std_sample, lw=0, zorder=1,
                                 color="#FF0000", alpha = 0.25, label=i + 1)
            axes[i].plot(inputs[i], data_mean, lw=4, zorder=24, color='#FF0000')

        axes[i].set_xlabel(direction_strings[i] + " stretch [-]", labelpad=-40)
        axes[i].set_ylabel(direction_strings[i] + " stress [kPa]", labelpad=-40)
        axes[i].minorticks_on()

        xt = [np.min(inputs[i]), np.max(inputs[i])]
        axes[i].set_xticks(xt)
        yt = [np.min(outputs[i]), np.max(outputs[i])]
        axes[i].set_yticks(yt)

        axes[i].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        axes[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i'))

        # secax = ax.secondary_xaxis('top')
        secax = axes[i].twiny()
        secax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,
            labelbottom=False, labeltop=False)
        # secax.set_xlabel("Temp", labelpad=-50)

        if not terms:
            secax.set_xlabel(f"Extra NLL = {nll - nll_min:.2f}", labelpad=-50)
            # axes[i].get_shared_y_axes().get_siblings(axes[i])[0].set_xlabel(f"$R^2$ = {r2:.4f}",
            #                                                                             labelpad=-50)

        if titles[i] is not None:
            axes[i].set_title(titles[i], y=1.05, usetex=False)

    print(f"Train Loss: {train_loss}")

    if terms:
        labels = [x for In in range(1, 3) for x in
                  [f"$(I_{In} - 3)$", f"exp$( (I_{In} - 3))$", f"$(I_{In} - 3)^2$", f"exp$( (I_{In} - 3)^2)$"]]
        labels = labels + [x for dir in ["I_{4w}", "I_{4s}", "I_{4s_{I, II}}"] for x in
                           [f"exp$({dir}) -  {dir}$", f"$({dir} - 1)^2$", f"exp$( ({dir} - 1)^2)$", ]]
        legend_handles = [Patch(color=c) for c in cmaplist] + [scatterplot]
        labels += ["data"]
        leg = fig.legend(loc="lower center", ncols=4, handles=legend_handles, labels=labels,
                         mode="expand", fontsize=40)
        leg.get_frame().set_alpha(0)



    if terms:
        leg_height = 0.24
        rect_height = 0.263
        x_offset = 0.0
        engine = fig.get_layout_engine()
        engine.set(rect=(0.005, leg_height, 0.99, 0.995 - leg_height), wspace=0.04)
        for j in range(5):
            rec = plt.Rectangle((0.2 * j, leg_height + rect_height), 0.2, 1 - leg_height - rect_height, fill=False,
                                lw=2)
            rec.set_zorder(1000)
            rec = fig.add_artist(rec)
        for j in range(2):
            rec = plt.Rectangle((0.4 * j, leg_height), 0.4, rect_height, fill=False, lw=2)
            rec.set_zorder(1000)
            rec = fig.add_artist(rec)
        rec = plt.Rectangle((0.8, leg_height), 0.2, rect_height, fill=False, lw=2)
        rec.set_zorder(1000)
        rec = fig.add_artist(rec)
    else:
        rect_height = 0.349
        x_offset = 0.005
        engine = fig.get_layout_engine()
        engine.set(rect=(0.005, 0.005, 0.99, 0.99), wspace=0.04)
        for j in range(5):
            rec = plt.Rectangle((0.2 * j-x_offset * (j < 3), rect_height), 0.2 + x_offset * (j == 2), 1 - rect_height, fill=False, lw=2)
            rec.set_zorder(1000)
            rec = fig.add_artist(rec)
        for j in range(2):
            rec = plt.Rectangle((0.4 * j-x_offset, 0), 0.4 + x_offset * (j==1), rect_height, fill=False, lw=2)
            rec.set_zorder(1000)
            rec = fig.add_artist(rec)
        rec = plt.Rectangle((0.8, 0), 0.2, rect_height, fill=False, lw=2)
        rec.set_zorder(1000)
        rec = fig.add_artist(rec)
    # Render and save plot
    name = "terms" if terms else ("variance2" if plot_dist else "variance1")
    if blank:
        plt.savefig(f"../Results/{modelFit_mode}/raw_data.pdf",
                    transparent=False,
                    facecolor='white')
    else:
        plt.savefig(f"../Results/{modelFit_mode}/bcann_{id}_{name}.pdf",
                    transparent=False,
                    facecolor='white')

    plt.close()





