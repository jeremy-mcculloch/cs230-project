import matplotlib.pyplot as plt
from src.CANN.cont_mech import *
from src.CANN.models import *
import matplotlib

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
def plot_bcann(stretch, stress_true, stress_pred, stress_std_pred, kevins_version):
    # already reshaped
    # stretch: 5 x 100 x 2
    # stress_true: 2 x 5(n_ex) x 5 x 100 x 2( x/y)
    # stress_pred: 2 x 5 x 100 x 2( x/y)

    # Plot Best Fit
    plt.rcParams['text.usetex'] = False
    plt.rcParams['figure.figsize'] = [30, 15]
    plt.rcParams['figure.constrained_layout.use'] = True



    fig, axes = plt.subplots(3, 5)
    axes = flatten([list(x) for x in zip(*axes[0:2])]) + list(axes[2])
    direction_strings = ["w", "s"] * 5 + ["x", "y", "x", "y", "x"]
    n_spaces = 55
    titles = ["strip-w", None, "off-w", None, "equibiax-ws", None, "off-s", None, "strip-s", None, " " * n_spaces + "strip-x", None," " * n_spaces + "off-x", None, "equibiax-xy"]
    inputs = [[stretch[j, :, k] for k in range(2)] for i in range(2) for j in range(5) ]
    inputs = [[x[i] if np.max(x[i]) > 1.0 else ((x[1 - i] - 1) * 1e-9 + 1) for i in range(2)] for x in inputs]
    inputs = flatten(inputs)[0:15] # 100 len each
    outputs = [stress_true[i, :, j, :, k] for i in range(2) for j in range(5) for k in range(2)][0:15] # 5 x 100 each
    pred_mean = [stress_pred[i, j, :, k] for i in range(2) for j in range(5) for k in range(2)][0:15]
    pred_std = [stress_std_pred[i, j, :, k] for i in range(2) for j in range(5) for k in range(2)][0:15]
    # print(pred_mean)
    # print(pred_std)

    num_points = 17
    for i in range(len(axes)):
        # Create plot that fills between the lower and upper bound
        axes[i].fill_between(inputs[i], pred_mean[i] - pred_std[i], pred_mean[i] + pred_std[i], lw=0, zorder=i + 1, color="#C0C7EC",
                         label=i + 1)
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

        for j in range(5):
            axes[i].scatter(inputs[i], outputs[i][j, :], s=300, zorder=25, lw=3, facecolors='w', edgecolors='k',
                                        clip_on=False)
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

        if titles[i] is not None:
            axes[i].set_title(titles[i], y=1.05, usetex=False)



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
    name = "kevin" if kevins_version else "jeremy"
    plt.savefig(f"../Results/bcann_{name}.pdf",
                transparent=False,
                facecolor='white')

    plt.close()

