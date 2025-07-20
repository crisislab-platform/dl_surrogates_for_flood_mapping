from modules.models.srr_lstm.srr.srr_reconstruction import SDRReconstructor
from modules.models.srr_lstm.srr.srr_reduction import SDRReducer

def findRLS():
    reducer = SDRReducer()
    reducer.sdr_searching()
    # reducer.sdr_searching_mcl()
    # reducer.sdr_rl()
    