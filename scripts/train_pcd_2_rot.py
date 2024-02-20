import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from hitchhiking_rotations import utils
from hitchhiking_rotations.models import MLPNetPCD
from hitchhiking_rotations.datasets import PointCloudDataset


from collections import namedtuple

Config = namedtuple('Config', ['training_loss', 'postprocess_pred', 'to_rot', 'preprocess_target', 'in_shape', 'out_size'])

cases = {
    "r9_l1": Config(utils.metrics.l1, utils.procrustes_to_rotmat, utils.procrustes_to_rotmat, utils.passthrough, (6, 3000), 9),
    "r9_l2": Config(utils.metrics.l2, utils.procrustes_to_rotmat, utils.procrustes_to_rotmat, utils.passthrough, (6, 3000), 9),
    "r9_geodesic": Config(utils.metrics.geodesic_distance, utils.procrustes_to_rotmat, utils.procrustes_to_rotmat, utils.passthrough, (6, 3000), 9),

    "r6_l1": Config(utils.metrics.l1, utils.gramschmidt_to_rotmat, utils.gramschmidt_to_rotmat, utils.passthrough, (6, 3000), 6),
    "r6_l2": Config(utils.metrics.l2, utils.gramschmidt_to_rotmat, utils.gramschmidt_to_rotmat, utils.passthrough, (6, 3000), 6),
    "r6_chordal": Config(utils.metrics.chordal_distance, utils.gramschmidt_to_rotmat, utils.gramschmidt_to_rotmat, utils.passthrough, (6, 3000), 6),
    "r6_geodesic": Config(utils.metrics.geodesic_distance, utils.gramschmidt_to_rotmat, utils.gramschmidt_to_rotmat, utils.passthrough, (6, 3000), 6),

    "quat_l1": Config(utils.metrics.l1, utils.passthrough, utils.quaternion_to_rotmat, utils.rotmat_to_quaternion, (6, 3000), 4),
    "quat_l2": Config(utils.metrics.l2, utils.passthrough, utils.quaternion_to_rotmat, utils.rotmat_to_quaternion, (6, 3000), 4),
    "quat_dp": Config(utils.metrics.l2_dp, utils.passthrough, utils.quaternion_to_rotmat, utils.rotmat_to_quaternion, (6, 3000), 4),
    "quat_cp": Config(utils.metrics.cosine_distance, utils.passthrough, utils.quaternion_to_rotmat, utils.rotmat_to_quaternion, (6, 3000), 4),
    "quat_chordal": Config(utils.metrics.chordal_distance, utils.quaternion_to_rotmat, utils.quaternion_to_rotmat, utils.passthrough, (6, 3000), 4),
    "quat_geodesic": Config(utils.metrics.geodesic_distance, utils.quaternion_to_rotmat, utils.quaternion_to_rotmat, utils.passthrough, (6, 3000), 4),

    # "quataug_l1": Config(utils.metrics.l1, utils.passthrough, utils.quaternion_to_rotmat, utils.rotmat_to_quaternion_rand_flip, (6, 3000), 4),
    # "quataug_l2": Config(utils.metrics.l2, utils.passthrough, utils.quaternion_to_rotmat, utils.rotmat_to_quaternion_rand_flip, (6, 3000), 4),
    "quataug_dp": Config(utils.metrics.l2_dp, utils.passthrough, utils.quaternion_to_rotmat, utils.rotmat_to_quaternion_rand_flip, (6, 3000), 4),
    # "quataug_cp": Config(utils.metrics.cosine_distance, utils.passthrough, utils.quaternion_to_rotmat, utils.rotmat_to_quaternion_rand_flip, (6, 3000), 4),

    "quatcanonical_l1": Config(utils.metrics.l1, utils.passthrough, utils.quaternion_to_rotmat, utils.rotmat_to_quaternion_canonical, (6, 3000), 4),
    "quatcanonical_l2": Config(utils.metrics.l2, utils.passthrough, utils.quaternion_to_rotmat, utils.rotmat_to_quaternion_canonical, (6, 3000), 4),
    "quatcanonical_dp": Config(utils.metrics.l2_dp, utils.passthrough, utils.quaternion_to_rotmat, utils.rotmat_to_quaternion_canonical, (6, 3000), 4),
    "quatcanonical_cp": Config(utils.metrics.cosine_distance, utils.passthrough, utils.quaternion_to_rotmat, utils.rotmat_to_quaternion_canonical, (6, 3000), 4),
    "quatcanonical_chordal": Config(utils.metrics.chordal_distance, utils.quaternion_to_rotmat, utils.quaternion_to_rotmat, utils.passthrough, (6, 3000), 4),
    "quatcanonical_geodesic": Config(utils.metrics.geodesic_distance, utils.quaternion_to_rotmat, utils.quaternion_to_rotmat, utils.passthrough, (6, 3000), 4),

    "axisangle_l1": Config(utils.metrics.l1, utils.passthrough, utils.rotvec_to_rotmat, utils.rotmat_to_rotvec, (6, 3000), 3),
    "axisangle_l2": Config(utils.metrics.l2, utils.passthrough, utils.rotvec_to_rotmat, utils.rotmat_to_rotvec, (6, 3000), 3),
    "axisangle_chordal": Config(utils.metrics.chordal_distance, utils.rotvec_to_rotmat, utils.rotvec_to_rotmat, utils.passthrough, (6, 3000), 3),
    "axisangle_geodesic": Config(utils.metrics.geodesic_distance, utils.rotvec_to_rotmat, utils.rotvec_to_rotmat, utils.passthrough, (6, 3000), 3),

    "euler_l1": Config(utils.metrics.l1, utils.passthrough, utils.euler_to_rotmat, utils.rotmat_to_euler, (6, 3000), 3),
    "euler_l2": Config(utils.metrics.l2, utils.passthrough, utils.euler_to_rotmat, utils.rotmat_to_euler, (6, 3000), 3),
    "euler_chordal": Config(utils.metrics.chordal_distance, utils.euler_to_rotmat, utils.euler_to_rotmat, utils.passthrough, (6, 3000), 3),
    "euler_geodesic": Config(utils.metrics.geodesic_distance, utils.euler_to_rotmat, utils.euler_to_rotmat, utils.passthrough, (6, 3000), 3),
}

loss_name = {
    "l1": 'MAE',
    "l2": 'MSE',
    "dp": 'DP',
    "cp": 'CD',
    "chordal": 'Chordal',
    "geodesic": 'Geodesic',
}
map_names ={"r9": r'$\mathbb{R}^9+$SVD',
            "r6":  r'$\mathbb{R}^6$+GSO',
            "quat": "Quat",
            "quataug": r'Quat$^+$',
            "quatcanonical": 'Quat + canonical',
            "axisangle": "Exp",
            "euler": "Euler"}

config_names = {k: f"{map_names[k.split('_')[0]]} - {loss_name[k.split('_')[1]]}" for k in cases.keys()}

def train(model: MLPNetPCD, dataset_train: PointCloudDataset, conf: Config, nb_epoch=100, batch_size=32, out_path=''):

    if os.path.exists(f'./{out_path}.pt'):
        print(f"Model {out_path} already trained")
        return
    else:
        print(f"\nTraining {out_path} model\n\n")

    class EarlyStopper:
        def __init__(self, patience=1, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = float('inf')

        def early_stop(self, validation_loss):
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False


    early_stopper = EarlyStopper(patience=10, min_delta=0)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7, amsgrad=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses, val_losses = [], []

    running_loss = 0.0
    for epoch in range(nb_epoch):
        model.train()

        for in_pcd, target_rot in (bar:= tqdm(dataloader_train)):
            with torch.no_grad():
                target_rot = conf.preprocess_target(target_rot)

            optimizer.zero_grad()
            outputs = model(in_pcd)
            outputs = conf.postprocess_pred(outputs)

            l = conf.training_loss(outputs, target_rot)
            l.backward()
            optimizer.step()
            running_loss = l.item()

            bar.set_description(f'Epoch [{epoch + 1}/{nb_epoch}], Train Loss: {running_loss}')

        train_losses.append(running_loss)

        with torch.no_grad():
            model.eval()
            val_in_pcd, val_out_rot = dataset_train[-100:]
            val_out_rot = conf.preprocess_target(val_out_rot)

            val_output = model(val_in_pcd)
            val_output = conf.postprocess_pred(val_output)
            val_loss = conf.training_loss(val_output, val_out_rot).item()
            val_losses.append(val_loss)

        # if early_stopper.early_stop(val_loss):
        #     break

    # Save
    torch.save(model.state_dict(), f'./{out_path}.pt')

    print("\n\n")
    ##############################################################################


@torch.no_grad()
def test(model: MLPNetPCD, dataset_test: PointCloudDataset, to_rot, out_path):
    result_path = f'./{out_path}_results.pkl'
    if os.path.exists(result_path):
        print(f"Results {result_path} already computed")
        return pickle.load(open(result_path, 'rb'))

    test_in_pcd, test_out_rot = dataset_test[:]

    # test
    model.load_state_dict(torch.load(f'./{out_path}.pt'))
    model.eval()

    test_pred = model(test_in_pcd)
    test_pred_rot = to_rot(test_pred)

    # metrics
    metrics = {
        "CHORDAL": torch.norm(test_pred_rot.view(-1,3,3) - test_out_rot, p='fro', dim=[1, 2]).cpu().numpy(),
        "GEODESIC": utils.roma.rotmat_geodesic_distance(test_pred_rot.view(-1,3,3), test_out_rot).cpu().numpy(),
    }
    pickle.dump(metrics, open(result_path, 'wb'))

    return metrics


def plot_results(results_path, outputdir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use(f'{os.path.dirname(os.path.realpath(__file__))}/prettyplots.mplstyle')
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 11})
    #plt.rcParams['figure.figsize'] = [8, 8]
    colors = [(0.368, 0.507, 0.71), (0.881, 0.611, 0.142), (0.923, 0.386,0.209),
            (0.56, 0.692, 0.195),(0.528, 0.471, 0.701), (0.772, 0.432,0.102),
            (0.572, 0.586, 0.) ]
    current_size = (5.6,2.1)

    results = {f.split('_results')[0]: pickle.load(open(f"{results_path}/{f}", "rb")) for f in os.listdir(results_path) if f.endswith('.pkl')}

    datas = {}
    for f in os.listdir(results_path):
        if f.endswith('.pkl'):
            metrics = pickle.load(open(f"{results_path}/{f}", "rb"))
            for metric, value in metrics.items():
                if metric in datas:
                    datas[metric][config_names[f.split('_results')[0]]] = value
                else:
                    datas[metric] = {config_names[f.split('_results')[0]]: value}


    os.makedirs(outputdir, exist_ok=True)
    for metric, data in datas.items():
            # reset fig
            plt.figure()
            sns.boxplot(data=data, palette="Blues", orient='h', width=0.5, linewidth=1.5, fliersize=2.5, showmeans=False)
            plt.xlabel(f'{metric} distance (rad)')
            plt.tight_layout()

            plt.xscale('log')
            plt.savefig(f'./{outputdir}/{metric}_pcd_results_logx.png', dpi=300)
            plt.savefig(f'./{outputdir}/{metric}_pcd_results_logx.pdf')

            plt.xscale('linear')
            plt.savefig(f'./{outputdir}/{metric}_pcd_results.png', dpi=300)
            plt.savefig(f'./{outputdir}/{metric}_pcd_results.pdf')


            # line
            plt.axvline(x=np.pi, color='k', linestyle='--', linewidth=1)

            plt.xscale('log')
            plt.savefig(f'./{outputdir}/{metric}_pcd_results_line_logx.png', dpi=300)
            plt.savefig(f'./{outputdir}/{metric}_pcd_results_line_logx.pdf')

            plt.xscale('linear')
            plt.savefig(f'./{outputdir}/{metric}_pcd_results_line.png', dpi=300)
            plt.savefig(f'./{outputdir}/{metric}_pcd_results_line.pdf')

            # close
            plt.close()

if __name__ == "__main__":

    out_dir = f'output/pcd_exp'
    os.makedirs(out_dir, exist_ok=True)
    num_epochs = 100
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for seed in range(10):
        for key, config in cases.items():
            print(f"\n\nTraining {key}")
            # SAME AS IN TRAIN.py
            training_loss, postprocess_pred, to_rot, preprocess_target, in_shape, out_size = config



            out_path = f'{out_dir}/{seed}/{key}'
            os.makedirs(out_path, exist_ok=True)

            # Train
            dataset_train = PointCloudDataset(mode="train", device=device)
            model = MLPNetPCD(in_size=in_shape, out_size=out_size).to(device, dtype=torch.float32)
            train(model, dataset_train, config, nb_epoch=num_epochs, batch_size=batch_size, out_path=out_path)


            # Test
            dataset_test = PointCloudDataset(mode="test", device=device)
            model = MLPNetPCD(in_size=in_shape, out_size=out_size).cuda()

            results = test(model, dataset_test, config.to_rot, out_path)



        # Plot
        plot_results(results_path=f"{out_dir}/{seed}", outputdir='assets/pcd_results')