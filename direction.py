import nibabel
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    #img = nibabel.load(r'D:/Datasets/PRoVe/ProVe-IT-01-001/MNI_mCTA1_brain.nii.gz')
    img = sitk.ReadImage(r'D:/Datasets/MyRegistration/ProVe-IT-01-001/MCA_mCTA1.nii.gz')
    data = sitk.GetArrayFromImage(img)
    #data = img.get_fdata()
    print(data.shape)
    flipped = np.flip(data, axis=2)
    slice0 = flipped[:, 128, :]
    slice1 = data[:, 128, :]
    #subplot
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(slice0, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(slice1, cmap='gray')
    plt.show()
    
    