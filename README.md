# [NTIRE 2025 the First Challenge on Event-Based Deblurring](https://codalab.lisn.upsaclay.fr/competitions/21498) @ [CVPR 2025](https://cvlai.net/ntire/2025/)

This is a simple introduction to the dataset and basic codes.



## How to start testing?
1. The 'num_bins' of the voxel dataset is 30.
2. test command:
Example:
```
python3 basicsr/test.py -opt options/test/HighREV/Model22_HighREV_Deblur_voxel.yml
```

Be sure to modify the 'dataroot' and 'dataroot_voxel' configurations in yml file.
