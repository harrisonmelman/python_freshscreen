project_code="19.gaj.43"
runno="N57252NLSAM"
spec_id="190415-1_1"

spec_id_fresh=$(echo $spec_id | sed "s/_/-/")
in_file="/Volumes/PWP-CIVM-CTX01/${project_code}/${spec_id}/Aligned-Data/labels/RCCF/${runno}_RCCF_labels.nhdr"
out_file="/Users/harry/scratch/neuroglancer_python_prototype/data/${spec_id_fresh}_${runno}_RCCF_labels.precomputed"
python prototype_make_precomputed_file.py ${in_file} ${out_file}

runno="N57207NLSAM"
spec_id="190415-4_1"
spec_id_fresh=$(echo $spec_id | sed "s/_/-/")
# TODO: pattern match for *_labels.nhdr or something, bc i keep getting burned on not having correct runno OR there being some extra bits in the name like _MC_
in_file="/Volumes/PWP-CIVM-CTX01/${project_code}/${spec_id}/Aligned-Data/labels/RCCF/${runno}_RCCF_labels.nhdr"
out_file="/Users/harry/scratch/neuroglancer_python_prototype/data/${spec_id_fresh}_${runno}_RCCF_labels.precomputed"
python prototype_make_precomputed_file.py ${in_file} ${out_file}

# TODO: let the above script (which handles naming, so it has the full precomputed filename) handle uploading to s3. It can then auto-start the script to make scene jsons, and auto-upload them too. 
