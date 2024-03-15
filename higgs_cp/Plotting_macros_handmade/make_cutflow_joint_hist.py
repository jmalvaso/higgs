import matplotlib.pyplot as plt
import mplhep
import pickle
import numpy as np


def create_cutflow_histogram(cuts, data, xlabel="Selections", ylabel="Selection efficiency", title="", save_path=None):
    """
    Create a cutflow histogram using Matplotlib with CMS style and save it to a PDF file.

    Parameters:
    - cuts: List of strings representing the names of the cuts.
    - data: List of integers representing the corresponding event counts for each cut.
    - xlabel: Label for the x-axis (default is "Cuts").
    - ylabel: Label for the y-axis (default is "Events").
    - title: Title of the plot (default is "Cutflow Histogram").
    - save_path: Path to save the PDF file. If None, the plot will be displayed but not saved.

    Returns:
    - fig: Matplotlib figure object.
    - ax: Matplotlib axis object.
    """

    # Set CMS style
    plt.style.use(mplhep.style.CMS)

    # Create Matplotlib figure and axis
    fig, ax = plt.subplots()
    
    # Set log scale on the y-axis
    ax.set_yscale('log')

    # Plotting the cutflow histogram
    color = ['black','red']
    for i, (name, values) in enumerate(data.items()):
        values = np.array(values)
        ax.scatter(cuts, values /values[0] , color=color[i], marker='o', alpha=0.8, label=name)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(cuts, rotation=45, ha='right')
    ax.set_ylim(0,1.1)
    # Add legend
    ax.legend()
    ax.grid(True)
    label_options = {
    "wip": "Work in progress",
    "pre": "Preliminary",
    "pw": "Private work",
    "sim": "Simulation",
    "simwip": "Simulation work in progress",
    "simpre": "Simulation preliminary",
    "simpw": "Simulation private work",
    "od": "OpenData",
    "odwip": "OpenData work in progress",
    "odpw": "OpenData private work",
    "public": "",
    }
    cms_label_kwargs = {
        "ax": ax,
        "llabel": label_options.get("pw"),
        "fontsize": 22,
        "data": True,
    }
    mplhep.cms.label(**cms_label_kwargs)
   
    plt.tight_layout()
    
     # Save to PDF if save_path is provided
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        # Show the plot if save_path is not provided
        plt.show()


    return fig, ax

def get_hist_values(pickle_file):
    file_obj = open(pickle_file, 'rb')
    data = pickle.load(file_obj)
    hist = data.profile(axis=0)
    cuts = []
    values = []
    print(hist)
    for cut_name in hist.axes[1]:
        cuts.append(f'{cut_name}')
        values.append(hist[0,f'{cut_name}',0,0].count)
        print(f'{cut_name}',hist[0,f'{cut_name}',0,0].count)
    return cuts, values

 

path22 = "/afs/cern.ch/user/j/jmalvaso/higgs_cp/data/cf_store/analysis_higgs_cp/cf.CreateCutflowHistograms/run3_2022_postEE_nano_tau_v12_limited/data_egamma_f/nominal/calib__example/sel__default__steps_trigger_e_pt_33_e_eta_2p4_e_dxy_0p045_e_dz_0p2_e_iso_0p15_e_mvaIso_WP90_Tau_pt_30_Tau_eta_2p3_Tau_dz_0p2_Tau_vs_e_Tau_vs_mu_Tau_vs_jet_Etau_couples_mT_less_70_veto_mu_veto_e_veto_os/dev/cutflow_hist__event.pickle"
path18 = "/afs/cern.ch/user/j/jmalvaso/higgs_cp/data/cf_store/analysis_higgs_cp/cf.CreateCutflowHistograms/run2_UL2018_nano_tau_v10_limited/data_egamma_ul2018_a/nominal/calib__example/sel__default__steps_trigger_e_pt_33_e_eta_2p4_e_dxy_0p045_e_dz_0p2_e_iso_0p15_e_mvaIso_WP90_Tau_pt_30_Tau_eta_2p3_Tau_dz_0p2_Tau_vs_e_Tau_vs_mu_Tau_vs_jet_Etau_couples_mT_less_70_veto_mu_veto_e_veto_os/dev/cutflow_hist__event.pickle"
cuts, values22 = get_hist_values(path22)
cuts, values18 = get_hist_values(path18)

create_cutflow_histogram(cuts, 
                         data={"2022_postEE": values22,
                             "UL2018": values18},
                         save_path="cutflow_histogram_2018vs2022.pdf")