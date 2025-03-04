import nibabel as nib
import numpy as np
import pathlib as Path
import matplotlib.pyplot as plt

'''
View the contents of a processed NII file (for example after segmentation)
'''
base_folder = 'Z:/Hospital/Michele/'
# Load the predicted volume
pred_img = nib.load(base_folder + 'sub-30_T1w_predicted_volume.nii.gz')
pred_data = pred_img.get_fdata()

# Check the data type and value range
print(f"Data type: {pred_data.dtype}")
print(f"Value range: {np.min(pred_data)} to {np.max(pred_data)}")

# Count unique values (for the predicted volume)
unique_vals = np.unique(pred_data)
print(f"Unique values: {unique_vals}")

# For the probability map
prob_img = nib.load(base_folder + 'sub-30_T1w_prob_map_volume.nii.gz')
prob_data = prob_img.get_fdata()
print(f"Probability map shape: {prob_data.shape}")

# Display a middle slice from each dimension
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
slice_x = pred_data.shape[0] // 2
slice_y = pred_data.shape[1] // 2
slice_z = pred_data.shape[2] // 2

axes[0].imshow(pred_data[slice_x, :, :], cmap='nipy_spectral')
axes[0].set_title('Sagittal')
axes[1].imshow(pred_data[:, slice_y, :], cmap='nipy_spectral')
axes[1].set_title('Coronal')
axes[2].imshow(pred_data[:, :, slice_z], cmap='nipy_spectral')
axes[2].set_title('Axial')

plt.tight_layout()
plt.show()
# %%
