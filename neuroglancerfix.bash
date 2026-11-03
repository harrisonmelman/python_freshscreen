output_dir="U:/freshscreen_n5_library/to_S3/jsonfiles/19.gaj.43"
nhdr_dir="Q:/19.gaj.43/190108-5_1/Aligned-Data"

project_code="19.gaj.43"
code_dir="K:/workstation/code/display/python_freshscreen";

specimen_list="190415-4_1"

echo $specimen_list;
for specimen_id in $specimen_list; do
    nhdr_dir="Q:/19.gaj.43/${specimen_id}/Aligned-Data"
    python ${code_dir}/prototype_write_neuroglancer_scene_json.py ${project_code} ${specimen_id} ${nhdr_dir} ${output_dir};
    exit 1;
done
