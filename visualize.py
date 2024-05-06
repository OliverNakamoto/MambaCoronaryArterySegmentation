import napari
import SimpleITK as sitk

image_path = "Annotations/Diseased_1.nrrd"
image_path2 = "Normal/Normal/Annotations/Normal_20.nrrd"



#image_path = "Diseased_7.nrrd"


image = sitk.ReadImage(image_path)
img_array = sitk.GetArrayFromImage(image)

image2 = sitk.ReadImage(image_path2)
img_array2 = sitk.GetArrayFromImage(image2)

with napari.gui_qt():
    #viewer = napari.view_image(img_array, colormap='gray')
    viewer = napari.view_image(img_array2, colormap='gray')
