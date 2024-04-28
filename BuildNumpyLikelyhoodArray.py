from __future__ import print_function
import os, sys
import ipympl
import ROOT
ROOT.PyConfig.DisableRootLogon = True
ROOT.PyConfig.IgnoreCommandLineOptions = False
import numpy as np
from ROOT import TCanvas, TGraph, gROOT, TFile, TObject
from math import sin
from array import array
import ROOT as r
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.optimize as opt
from array import array
import numpy as np
import scipy.optimize as opt
from glob import glob
import re

xbin = np.arange(0.5,33.5,1)


path_to_Unpolarized_Data = "~/xlc_sim_outputs/flight_like/UnPolarized.root"

unpol_File = TFile.Open(path_to_Unpolarized_Data,"read")
PD0_Tree = unpol_File.Get(f"tree")
unpol_hist = r.TH1D('unpol_hist','unpol_hist',len(xbin)-1,xbin)
for event in PD0_Tree:
    if PD0_Tree.horizontal_pixel < 32:
        unpol_hist.Fill(PD0_Tree.horizontal_pixel)


probs = np.zeros(40)
unpol_prob_dataframe = pd.DataFrame(index=np.arange(1, 41, 1))


unpol_hist.Scale(1 / unpol_hist.Integral(), "width")

for i in range(40):
        j = i + 1
        probs[i] = unpol_hist.GetBinContent(j)
unpol_hist.Delete()
unpol_prob_dataframe['Probs'] = probs
unpol_prob_dataframe = unpol_prob_dataframe.loc[unpol_prob_dataframe.index < 32]
unpol_prob_dataframe['pixel'] = np.arange(1, 32, 1)
print(probs)


probs = np.zeros(40)
def build_prob_array(File,row):
        Prob_DataFrame = pd.DataFrame(index=np.arange(1, 41, 1))
        for tree in range(len(File.GetListOfKeys())):
                fTree = File.Get(f"tree_{tree}")
                prob_hist = r.TH1D('prob_hist', 'prob_hist', len(xbin) - 1, xbin)
                for event in fTree:
                        if fTree.horizontal_pixel < 32 & fTree.pixel_row == row:
                                prob_hist.Fill(fTree.horizontal_pixel)
                prob_hist.Scale(1 / prob_hist.Integral(), "width")
                for i in range(40):
                        j = i + 1
                        probs[i] = prob_hist.GetBinContent(j)
                prob_hist.Delete()
                df = pd.DataFrame(index=np.arange(1, 41, 1))
                df[f'{tree}'] = probs

                Prob_DataFrame = pd.concat([Prob_DataFrame, df], axis=1).copy()
        return Prob_DataFrame




def get_numbers_from_filename(filename):
        return re.search(r'\d+', filename).group(0)


likelihoodfiles = sorted(glob(f'/mnt/d/VaryingRotationAngle/SimData_Histograms_RotAng*.root'))
likelihood_list = []
full_likelihood_tensor = []
#for i, f in enumerate(likelihoodfiles):
for i in range(0,10):
        for row in range(0,40):
	
                likelihood_list = []
                full_likelihood_tensor = []
                for rot_ang in np.arange(0+36*i,36+36*i,1):
                #rot_ang = get_numbers_from_filename(f)

                        File = ROOT.TFile.Open(f'/mnt/d/VaryingRotationAngle/SimData_Histograms_RotAng{rot_ang}.root', "read")
                        pol_prob_dataframe = build_prob_array(File,row)
                        pol_prob_dataframe = pol_prob_dataframe.loc[pol_prob_dataframe.index < 32]
                        pol_prob_dataframe['pixel'] = np.arange(1, 32, 1)

                        pol_likelihoods = []
                        likelihoods_array_2D = np.empty_like(unpol_prob_dataframe.to_numpy())
                        unpol_likelihoods = np.zeros([181, 31])
                        pol_prob_dataframe['180'] = pol_prob_dataframe['0']
                        for j in range(0, 181):
                                unpol_likelihoods[j] = unpol_prob_dataframe['Probs'].to_numpy()
                        pol_likelihoods = unpol_likelihoods.T
                        for x in np.arange(0, 1, 0.01):
                                pol = pol_prob_dataframe.drop('pixel', axis=1).to_numpy()
                                unpol = unpol_prob_dataframe.copy()
                                likelihoods = ((1 - x) * unpol_likelihoods.T) + ((x) * pol)
                                pol_likelihoods = np.dstack((pol_likelihoods, likelihoods))
                        likelihood_list.append(pol_likelihoods)
                        print(rot_ang)
                full_likelihood_tensor[row] = np.stack(likelihood_list, axis=3)
        full_likelihood_tensor = np.stack(full_likelihood_tensor,axis=4)
        print(full_likelihood_tensor.shape)
      	np.save(f"/home/jacob/xlc_sim_outputs/notebook_outputs/likelihood_tensor_{36*i}_{36*i}.npy", -np.log(full_likelihood_tensor))
