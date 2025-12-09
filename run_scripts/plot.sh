# python main.py plot --csv_file "Upstream_Flows_Run1.csv"

# python main.py plot --plot_type up_conditions --event "1"

# python main.py plot --plot_type rep_locations --run_id 20250314_141848  --file ss_300.shp

# python main.py plot --plot_type hydrograph_clean

# python main.py plot --plot_type test_event

# parallel --line-buffer -j 1 CUDA_VISIBLE_DEVICES={1} python3.11 main.py plot \
#  --plot_type bootstrap \
#  ::: 0 

python main.py plot --plot_type boundary_information