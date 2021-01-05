
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the APAC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


import nibabel as nib
import numpy as np

def make_funcgii(dummy_file, input_arr, out_file):
    r"""Transfer array to func.gii file

    Parameters
    ----------

    dummy_file: str 
        template func file, the dimension of the file must be same with input_arr
        func gifti file (extension: '.func.gii')

    input_arr: list | numpy.array
        input array to plot
        must be same dimension with dummy_file

    out_file: str
        file name for output func.gii file
        Extension ".func.gii" is no matter
    """

    dummy = nib.load(dummy_file)
    dummy.darrays[0].data = input_arr.astype(np.float32)

    if '.'.join(out_file.split('.')[-2:]) == 'func.gii':
        pass
    else:
        out_file = out_file + '.func.gii'
    nib.save(dummy, out_file)
    #print('Saved : {}'.format(out_file))
    

def sphere_clustering(sphere_file, cluster, dist=3):
    # cluster = 1-D array
    # coords = shpere surface corrdinate
    sphere = nib.load(sphere_file).darrays[0].data
    sphere_coords = sphere[cluster>0]
    Npoint = len(sphere_coords)
    dist_map = np.zeros((Npoint, Npoint))
    for i in range(Npoint):
        for j in range(Npoint):
            if i > j:
                dist_map[i,j] = np.linalg.norm(sphere_coords[i] - sphere_coords[j])
    dist_map = dist_map + dist_map.T
    labels = np.zeros(Npoint)

    def dfs(idx):
        adjacents = np.where(dist_map[idx] < dist)[0]
        for adj_idx in adjacents:
            if labels[adj_idx] == 0:
                labels[adj_idx] = labels[idx]
                dfs(adj_idx)

    for idx, label in enumerate(labels):
        # first >> label
        # second >> recur adjacent labels
        if label == 0:
            labels[idx] = labels.max()+1
            dfs(idx)
    return labels


def surf_morph(self, input_arr, hemi, mode='dilation', iteration=1, plot=False):
    hemi_val = self.hemi_dict[hemi]
    sphere = self.file_dict['sphere_surf'][hemi_val]
    surf = nib.load(sphere)
    x,y,z = surf.darrays[0].data.T
    faces = surf.darrays[1].data
    pos, = np.where(input_arr>0)
    out_arr = input_arr.copy()

    for _ in range(iteration):
        if mode == 'dilation':
            for idx in pos:
                idx_faces = np.unique(faces[np.where(faces==idx)[0]].ravel())
                out_arr[idx_faces] = 1

        elif mode == 'erosion':
            frame = out_arr.copy()
            for idx in pos:
                idx_faces = np.unique(faces[np.where(faces==idx)[0]].ravel())
                if frame[idx_faces].min() == 0:
                    out_arr[idx] = 0
        else:
            print('Reset the mode!')
        pos, = np.where(out_arr>0)

    if plot==True:
        representation = ['surface', 'wireframe', 'points', 'mesh', 'fancymesh']
        mlab.figure(figure=mode, size=(1280,960))
        mlab.triangular_mesh(x,y,z, faces, scalars=out_arr, representation=representation[4])
    return out_arr
