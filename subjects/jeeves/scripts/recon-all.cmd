

#---------------------------------
# New invocation of recon-all Wed Aug 16 16:10:47 EDT 2023 
#--------------------------------------------
#@# Mask BFS Wed Aug 16 16:10:49 EDT 2023

 mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz 

#--------------------------------------------
#@# Fill Wed Aug 16 16:10:54 EDT 2023

 mri_fill -a ../scripts/ponscc.cut.log -segmentation aseg.presurf.mgz -ctab /usr/local/freesurfer/7.3.0//SubCorticalMassLUT.txt wm.mgz filled.mgz 



#---------------------------------
# New invocation of recon-all Wed Aug 16 16:12:22 EDT 2023 
#--------------------------------------------
#@# Mask BFS Wed Aug 16 16:12:22 EDT 2023

 mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz 

#--------------------------------------------
#@# Fill Wed Aug 16 16:12:27 EDT 2023

 mri_fill -a ../scripts/ponscc.cut.log -segmentation aseg.presurf.mgz -ctab /usr/local/freesurfer/7.3.0//SubCorticalMassLUT.txt wm.mgz filled.mgz 

