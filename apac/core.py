### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the APAC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


from . import util
from nilearn import decomposition
from sklearn import mixture
from scipy import stats
import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

class core:
    def __init__(self, OutDir):
        self.file_dict = dict()

        if not os.path.exists(OutDir):
            os.makedirs(OutDir)

        OutDir = os.path.join(OutDir, 'core')
        if not os.path.exists(OutDir):
            os.makedirs(OutDir)

        self.OutDir = OutDir
        self.hemi_dict = {'L':0,'R':1}
        
    def call(self, DataDir, AtlasDir):
        self.fs_path = DataDir
        self.file_dict['sphere_surf'] = sorted(glob.glob(DataDir + '/*.?.sphere.*.surf.gii'))
        self.file_dict['myelin'] = sorted(glob.glob(DataDir + '/*.?.SmoothedMyelinMap_BC.*.func.gii'))
        self.file_dict['curvature'] = sorted(glob.glob(DataDir + '/*.?.curvature.*.shape.gii'))
        self.file_dict['MMP'] = sorted(glob.glob(AtlasDir + '/HCPMMP.?.32k_fs_LR.label.gii'))
        

    def initial_roi(self, hemi):
        # hemi = 0 (lh) or 1 (rh)
        hemi_val = self.hemi_dict[hemi]
        MMP = nib.load(self.file_dict['MMP'][hemi_val]).darrays[0].data
        early_aud = [24, 103, 104, 105, 124, 173, 174]
        self.initial_roi = np.isin(MMP, early_aud)


	def def_pcore(self, min_out):
        for hemi in ['L', 'R']:            
            hemi_val = self.hemi_dict[hemi]
            
            ### define initial ROI
            self.initial_roi(hemi)
            if min_out == 0:
                util.make_funcgii(
                    dummy_file = self.file_dict['myelin'][hemi_val], 
                    input_arr = self.initial_roi, 
                    out_file = os.path.join(
                                    self.OutDir, 
                                    '{}.initial_roi.func.gii'.format(hemi)))
            
            ### clustering
            # load myelin and take values only within the initial ROI
            myelin = nib.load(self.file_dict['myelin'][hemi_val]).darrays[0].data
            initial_roi = self.initial_roi
            myelin[initial_roi==0] = 0
            valid_myelin = myelin[initial_roi == 1]
            
            # GMM with k=3
            n_comp = 3
            gmm = GaussianMixture(n_components=n_comp)
            gmm.fit(valid_myelin.reshape(-1, 1))
            gmm_label = gmm.predict(valid_myelin.reshape(-1,1))
            
            # take the highest myelinated cluster
            myelin_idx = np.argmax([myelin[initial_roi==1][gmm_label==idx].mean() for idx in range(n_comp)])
            
            # make cluster image
            pcore_img = np.zeros_like(myelin)
            pcore_img[initial_roi==1] = (gmm_label==myelin_idx)
            
            sp_clust = util.sphere_clustering(self.file_dict['sphere_surf'][hemi_val], pcore_img)
            clustK = np.zeros_like(myelin)
            clustK[initial_roi==1] = gmm_label + 1
            if min_out == 0:
                util.make_funcgii(
                        dummy_file = self.file_dict['myelin'][hemi_val], 
                        input_arr = clustK,
                        out_file = os.path.join(
                                        self.OutDir, 
                                        '{}.clustK'.format(hemi) + '{}.func.gii'.format(n_comp)))
            
            # make a clear cluster (remove small redundant noise clusters)
            # by only taking the largest region
            list_idx, count_idx = np.unique(sp_clust, return_counts=True)
            largest_idx = list_idx[np.argmax(count_idx)]
            
            clust = np.zeros_like(pcore_img)
            clust[pcore_img == 1] = np.where(sp_clust == largest_idx, 1, 0)
            util.make_funcgii(
                    dummy_file = self.file_dict['myelin'][hemi_val], 
                    input_arr = clust,
                    out_file = os.path.join(
                                    self.OutDir, 
                                    '{}.self.func.gii'.format(hemi)))
            
            
            
            ### adjust for curvature (pCore-c)
            # load curvature and take negative values (sulci)
            curv = nib.load(self.file_dict['curvature'][hemi_val]).darrays[0].data
            sulc_line = np.where(curv < 0, 1, 0)
            sulc_line[initial_roi != 1] = 0

            # make suli border (limit of pCore expansion)
            # and remove overlapped region (pCore & border)
            border = np.where((pcore_img == 1) & (sulc_line == 1), 1, 0)
            A1A2 = np.where((pcore_img == 1) & (border == 0), 1, 0)
            
            clust = np.zeros_like(A1A2)
            clust[A1A2 == 1] = util.sphere_clustering(self.file_dict['sphere_surf'][hemi_val], A1A2)

            clust_labels = np.arange(1, clust.max()+1)
            counts = np.array([np.count_nonzero(clust==idx) for idx in clust_labels])
            pcore_c = np.isin(clust, np.argmax(counts)+1)
            p_celse = np.isin(clust, clust_labels[counts != counts[np.argmax(counts)]])

            if min_out == 0:
                util.make_funcgii(
                    dummy_file = self.file_dict['myelin'][hemi_val], 
                    input_arr = border, 
                    out_file = os.path.join(
                                    self.OutDir, 
                                    '{}.curv_border.func.gii'.format(hemi)))
            
            # expand the remained region until it touch the border
            # if there are >=2 remained regions, expand until they touch each other
            while True:
                pcore_c = pcore_c | util.surf_morph(pcore, pcore_c, hemi) * border
                if (pcore_c*p_celse).max() == 1:
                    break
                p_celse = p_celse | util.surf_morph(pcore, p_celse, hemi) * border
                if (pcore_c*p_celse).max() == 1:
                    break
                if (pcore_c*border).sum() == border.sum():
                    break
                prev_pcore_c = pcore_c
                if pcore_c.sum() == prev_pcore_c.sum():
                    break
            self.pcore_c = pcore_c

            util.make_funcgii(
                dummy_file = self.file_dict['myelin'][hemi_val], 
                input_arr = self.pcore_c, 
                out_file = os.path.join(
                                self.OutDir, 
                                '{}.pcore_c.func.gii'.format(hemi)))
