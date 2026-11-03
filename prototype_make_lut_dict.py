import csv

def rgba_to_hex(r, g, b, a):
    r=int(float(r))
    g=int(float(g))
    b=int(float(b))
    a=int(float(a))
    # :02x means "2 digits, lowercase hex" (use :02X for uppercase)
    return f"#{r:02x}{g:02x}{b:02x}{a:02x}"


def convert_lut_file(label_lut_file):
    roi_dict = {}

    with open(label_lut_file, 'r') as file:
        # resolve header fields
        # Done this way to safely have 'ROI' as the key field instead of '# ROI'
        original_reader = csv.reader(file,delimiter='\t')
        original_headers = next(original_reader)
        
        # strip whitespace and replace "# ROI" with "ROI"
        cleaned_headers = [h.strip().replace('# ROI', 'ROI') for h in original_headers]
        
        # Skip the comment row
        next(file)
        
        # use dict reader with cleaned headers
        reader = csv.DictReader(file, fieldnames=cleaned_headers, delimiter='\t')
        
        # Now iterate as normal
        for row in reader:
            roi = row['ROI']
            if 'NaN' in roi:
                # TODO: i don't love this conditional. is it safe??
                continue
            hex_color = rgba_to_hex(row['c_r'],row['c_g'],row['c_b'],row['c_a'])
            roi_dict[roi] = hex_color
    return roi_dict


in_file="B:/24.chdi.01-PHASE2/stats/month_15/Scalar_and_Volume/anovan_0110/Genotype_Sex/Non_Erode/Bilateral/Complex_Figures/Genotype/ad_mean/lookup_tables/Genotype_ad_mean_pval_BH_lookup.txt"
d = convert_lut_file(in_file)
print(d)