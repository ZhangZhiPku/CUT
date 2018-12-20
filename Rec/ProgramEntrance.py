"""
    Implementation of Matrix Factorization in Recommendation system.

    All algorithm are implemented with tensorflow on gpu and tested.

    Author Phoenix.z 2007 - 2018, all rights reserved.
"""

from Rec.Configurations import *
from Rec.Common.Utilites import *
from Rec.Models.SequentialModels import UPS_Cliassifler

if __name__ == '__main__':
    sequential_data, n_items, n_users = data_pre_process(mount_dataset())
    model = UPS_Cliassifler(n_items)
    model.pretrain(sequential_data)