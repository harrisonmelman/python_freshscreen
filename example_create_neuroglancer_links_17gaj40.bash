output_dir="U:/freshscreen_n5_library/to_S3/jsonfiles"
nhdr_dir="U:/freshscreen_n5_library/to_S3"
project_code="17.gaj.40"
code_dir="K:/workstation/code/display/python_freshscreen";

search_dir="U:/freshscreen_n5_library/17.gaj.40";
specimen_list="";
cd $search_dir;

for x in BTBR C57BL_6J CAST DB2; do
    cd $x/Done/;
    for y in $(ls); do
        # everything before the first underscore is the spec_id_fresh
        # everything after is the runno (but also trim the trailing slash)
        specimen_id=${y%_*};
        specimen_list="${specimen_list} ${specimen_id}";
    done;
    cd ../../;
done

echo $specimen_list;
#specimen_list="17GAJ40-BTBR-AF";
for specimen_id in $specimen_list; do
    python ${code_dir}/prototype_write_neuroglancer_scene_json.py ${project_code} ${specimen_id} ${nhdr_dir} ${output_dir};
done
