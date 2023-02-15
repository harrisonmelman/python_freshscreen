project_code="17.gaj.40"
label_type="WHS_heritability";
is_windows=0;

in_dir=/Volumes/PWP-CIVM-CTX01/freshscreen_n5_library/17.gaj.40;
out_dir=/Volumes/PWP-CIVM-CTX01/freshscreen_n5_library/to_S3/labels;
code_dir=/Users/harry/scratch/neuroglancer_python_prototype;
if $is_windows; then
    in_dir=/u/freshscreen_n5_library/17.gaj.40;
    out_dir=/u/freshscreen_n5_library/to_S3/labels;
    code_dir=/k/workstation/code/display/python_freshscreen/;
fi

cd ${in_dir}
for x in BTBR C57BL_6J CAST DB2; do
    cd $x/Done/;
    for y in $(ls); do
        # everything before the first underscore is the spec_id_fresh
        # everything after is the runno (but also trim the trailing slash)
        label_dir=${in_dir}/${x}/Done/${y}/labels*/${label_type};
        spec_id_fresh=${y%_*};
        runno=${y##*_};
        runno=${runno%/*};
        in_nii=$(ls ${label_dir}/${runno}*labels*nii);
        in_nhdr=${in_nii%.nii};
        in_nhdr=${in_nhdr}.nhdr;
        if [ ! -f $in_nhdr ]; then
            mk_nhdr ${in_nii};
        fi
        in_file=$in_nhdr;
        out_file=${out_dir}/${spec_id_fresh}_${runno}_${label_type}_labels.precomputed;
        if [ ! -d $out_file ]; then
            python ${code_dir}/prototype_make_precomputed_file.py ${in_file} ${out_file} $label_type
        fi
    done;
    cd ../../;
done
