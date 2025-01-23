from pytorch3d.transforms import Transform3d
import torch
import numpy as np
import pytorch3d
from tqdm import tqdm
import helper
import time

# matplotlib.use('TkAgg')



# start_time = time.time()
''' load mesh '''
# filename_in = 'objs/dolphin.obj'
# filename_in = 'outputs/hq_small/clown/clown_untextured.obj'
# filename_in = 'outputs/hq_small/stone/stone_untextured.obj'
# filename_in = 'outputs/hq_small/house/house_untextured.obj'
# filename_in = 'outputs/hq_small/pineapple/pineapple_untextured.obj'

''' setup '''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

def optimize_candidates(mesh_tri, box3d_verts_trans, text3d_verts_trans):
    start_time = time.time()
    num_waters = box3d_verts_trans.__len__()
    # box3d_verts_trans, sampled_normals = box3d_verts_trans.float().to(device), sampled_normals.float().to(device)
    box3d_verts_trans = box3d_verts_trans.float().to(device)
    text3d_verts_trans = text3d_verts_trans.float().to(device)
    mesh_py3d = helper.trimesh_to_py3d(mesh_tri)
    mesh_py3d = mesh_py3d.to(device)
    rot_params = torch.zeros(num_waters, 3).to(device).requires_grad_(True)
    trans_params = torch.zeros(num_waters, 3).to(device).requires_grad_(True)
    losses = []
    grad_trans = []
    grad_rot = []

    initial_loss = helper.mesh_boxes_loss(mesh_py3d, box3d_verts_trans).unsqueeze(0)
    losses.append(initial_loss.detach())

    # optimizer = torch.optim.SGD([rot_params, trans_params], lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam([rot_params, trans_params], lr=0.1)

    Niter = 200
    loop = tqdm(range(Niter))
    for idx in loop:
        optimizer.zero_grad()

        ''' ROTATE + TRANSLATE '''
        # translate the box
        t = Transform3d().translate(trans_params).to(device)
        box3d_verts_trans__ = t.transform_points(box3d_verts_trans)

        # rotate the box around its axis
        rot_matrices = pytorch3d.transforms.euler_angles_to_matrix(rot_params, 'XYZ')
        box_centroids = box3d_verts_trans__.mean(1)
        t = Transform3d().translate(-box_centroids).rotate(rot_matrices).translate(box_centroids).to(device)
        box3d_verts_trans_ = t.transform_points(box3d_verts_trans__)

        # ''' ROTATE ONLY '''
        # rot_matrices = pytorch3d.transforms.euler_angles_to_matrix(rot_params, 'XYZ')
        # box_centroids = box3d_verts_trans.mean(1)
        # t = Transform3d().translate(-box_centroids).rotate(rot_matrices).translate(box_centroids).to(device)
        # box3d_verts_trans_ = t.transform_points(box3d_verts_trans)

        # compute loss and back propagate
        loss = helper.mesh_boxes_loss(mesh_py3d, box3d_verts_trans_).unsqueeze(0)
        losses.append(loss.detach())
        loss = loss.mean() # MEAN OR SUM?
        if loss < 0.005:
            break
        old_loss = losses[-2].mean()
        percent_change = (old_loss-loss)/old_loss
        if percent_change>0 and percent_change<0.005:
            break

        loop.set_description('mean_loss = %.6f' % loss.detach())
        loss.backward(retain_graph=True)
        grad_trans.append(torch.linalg.norm(trans_params.grad, dim=1).mean().detach())
        grad_rot.append(torch.linalg.norm(rot_params.grad, dim=1).mean().detach())
        optimizer.step()

    # print(f'\n Cenrtroid Intial {box3d_verts_trans.squeeze().mean(0).unsqueeze(0)} Loop {box3d_verts_trans_.squeeze().mean(0).unsqueeze(0)}')
    losses = torch.cat(losses,0).cpu()
    box3d_verts_trans_ = box3d_verts_trans_.detach()
    box3d_verts_trans = box3d_verts_trans.detach()
    # print(f'Losses Initi {losses[0]}')
    # print(f'Losses Final {losses[-1]}')
    print(f'Mean Loss (Old -> New) {losses[0].mean()} -> {losses[-1].mean()}')
    print(f'Mean Loss diff (Ini - Final) +ve better - {losses[0].mean() - losses[-1].mean()}')
    box_distance = torch.square(box3d_verts_trans_.mean(1) - box3d_verts_trans.mean(1)).mean().cpu()
    print(f'Optimization complete. Steps {idx}/{Niter}. Take taken {round(time.time()-start_time,2)}s')
    print(f'Percent last box goodness {np.mean((losses.min(0)[1]==idx+1).numpy())}')
    print(f'Box distance {box_distance}')
    print(f'Mean losses {torch.mean(losses[-1])}')

    # plt.plot(losses.mean(1))
    # plt.title('Aggregate loss (Boxes-Mesh)')
    # plt.show()

    with torch.no_grad():
        t = Transform3d().translate(trans_params).to(device)
        text3d_verts_trans__ = t.transform_points(text3d_verts_trans)
        rot_matrices = pytorch3d.transforms.euler_angles_to_matrix(rot_params, 'XYZ')
        text_centroids = text3d_verts_trans__.mean(1)
        t = Transform3d().translate(-text_centroids).rotate(rot_matrices).translate(text_centroids).to(device)
        text3d_verts_trans_ = t.transform_points(text3d_verts_trans__)
    return box3d_verts_trans_, text3d_verts_trans_, losses, box3d_verts_trans, mesh_py3d
