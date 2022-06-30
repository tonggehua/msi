
from OCTOPUS.fieldmap.unwrap import fsl_prep
import nibabel as nib
from scipy.io import savemat
if __name__ == '__main__':
    #
    # fsl_fprep(data_path_raw='b0map_071521.mat',
    #                   data_path_dicom='./flywheel/gre_field_mapping_071521/1.3.12.2.1107.5.2.43.166011.2021071509201242210315592.0.0.0.dicom',
    #                   dst_folder='./fieldmap_prep', dTE=2.46e-3)
    # data = nib.load('./fieldmap_prep/fmap_rads.nii.gz')
    # b0maps = data.get_fdata()
    # print(data.shape)
    # print(type(data))
    # savemat('b0_rads_from_FSL.mat',{'data':b0maps})


    data = nib.load('./fieldmap_prep/phase_diff.nii.gz')
    data1 = data.get_fdata()
    savemat('phase_diff_dicom.mat',{'data':data1})


    data = nib.load('./fieldmap_prep/mag_vol_extracted.nii.gz')
    data1 = data.get_fdata()
    savemat('mag_vol_dat.mat',{'data':data1})