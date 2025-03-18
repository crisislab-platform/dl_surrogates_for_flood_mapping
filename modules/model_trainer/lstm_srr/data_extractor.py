def read_rawdata_from_csv(work_dir, grid_no):
    raw_data = dict()
    with open(f"{work_dir}grid_coords_{grid_no}.csv", newline='') as ptfile:
        reader = csv.reader(ptfile)
        grids_info = []
        row_no = 0
        for row in reader:
            if row_no == 0:
                row_no += 1
                continue
            else:
                row_no += 1
                grids_info.append(tuple([float(i) for i in row[1:]]))
    with open(f"{work_dir}grid_{grid_no}_rawdata.csv", newline='') as rawdatafile:
        reader = csv.reader(rawdatafile)
        raw_data[grids_info[0]] = dict()
        for row in reader:
            raw_data[grids_info[0]][row[0]] = [float(i) for i in row[1:]]
    return raw_data