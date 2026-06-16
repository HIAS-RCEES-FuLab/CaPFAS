from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import matplotlib.patches as patches

def venn2_3d_like(subsets, colors, alphas, save_path=None):
    plt.figure(figsize=(6, 6))
    v = venn2(subsets=subsets, set_labels=('', ''))

    # 隐藏数字
    for text_id in ['10', '01', '11']:
        label = v.get_label_by_id(text_id)
        if label:
            label.set_text('')

    for id in ['10', '01', '11']:
        patch = v.get_patch_by_id(id)
        if patch:
            patch.set_facecolor(colors[id])
            patch.set_alpha(alphas[id])
            patch.set_edgecolor('black')
            patch.set_linewidth(0.5)

    for id in ['10', '01', '11']:
        patch = v.get_patch_by_id(id)
        if patch:
            shadow = patches.PathPatch(
                patch.get_path(),
                facecolor='grey', alpha=0.25,
                transform=patch.get_data_transform() +
                plt.matplotlib.transforms.Affine2D().translate(0.015, -0.015),
                zorder=0
            )
            plt.gca().add_patch(shadow)

    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

    plt.show()

influent_colors = {
    '10': '#4C72B0',
    '01': '#DD8452',
    '11': '#55A868'
}
influent_alphas = {'10': 0.8, '01': 0.7, '11': 0.85}

venn2_3d_like(subsets=(281 - 262, 1760 - 262, 262),
              colors=influent_colors,
              alphas=influent_alphas,
              save_path='influent_venn.png')

effluent_colors = {
    '10': '#8172B3',
    '01': '#937860',
    '11': '#DA8BC3'
}
effluent_alphas = {'10': 0.8, '01': 0.7, '11': 0.85}

venn2_3d_like(subsets=(296 - 251, 1732 - 251, 251),
              colors=effluent_colors,
              alphas=effluent_alphas,
              save_path='effluent_venn.png')
