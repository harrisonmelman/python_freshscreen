basedir="/s/freshscreen_library/json_display_files"
cd $basedir
for d in $(ls); do
	cd $d;
	for x in $(ls *json); do
		echo $x;
		rclone copy -P --s3-no-check-bucket $x freshscreen:d3mof5o;
	done
	cd ../;
done
