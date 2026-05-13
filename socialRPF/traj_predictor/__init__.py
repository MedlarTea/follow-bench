from traj_predictor.cv import CV
# from traj_predictor.sgan import SGAN
from traj_predictor.cvkf import CVKF

def get_predictor(predictor_type, params):
    if predictor_type == 'cv':
        predictor = CV()
        predictor.set_params(params)
        return predictor
    elif predictor_type == 'cvkf':
        predictor = CVKF()
        predictor.set_params(params)
        return predictor
    # elif predictor_type == 'sgan':
    #     predictor = SGAN()
    #     predictor.set_params(params)
    #     return predictor
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")