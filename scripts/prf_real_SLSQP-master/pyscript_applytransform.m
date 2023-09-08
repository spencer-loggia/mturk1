fprintf(1,'Executing %s at %s:\n',mfilename(),datestr(now));
ver,
try,
        infile = '/Users/braunlichkr/Documents/python/clarte/atlases/HarvardOxford-maxprob-thr0_Atlas_Map.nii';
        outfile = '/Users/braunlichkr/Documents/experiments/macman_align/bin/first_lev/prf/HarvardOxford-maxprob-thr0_Atlas_Map_trans.nii'
        transform = load('/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/nii/sub-001/anat_clinical/iy_4_t1_mprage_sag_p2_0.75mm.nii');

        V = spm_vol(infile);
        X = spm_read_vols(V);
        [p n e v] = spm_fileparts(V.fname);
        V.mat = transform.M * V.mat;
        V.fname = fullfile(outfile);
        spm_write_vol(V,X);

        
,catch ME,
fprintf(2,'MATLAB code threw an exception:\n');
fprintf(2,'%s\n',ME.message);
if length(ME.stack) ~= 0, fprintf(2,'File:%s\nName:%s\nLine:%d\n',ME.stack.file,ME.stack.name,ME.stack.line);, end;
end;