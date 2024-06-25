class fitter:
    #conditionally creae an instance from a class
    def __new__(cls,fit):
        if fit == 'T2':
            return t2_fitter()
        elif fit =='T1':
            return t1_fitter()

        elif fit =='ADC':
            return ADC_fitter()
        elif fit =='diffusion' or fit == 'DTI':
            return dti_fitter()



class t1_fitter:
    def __init__(self) -> None:
        print('Initialzing t1 fitter')
        
class t2_fitter:
    def __init__(self) -> None:
        print('Initialzing t2 fitter')
        
class ADC_fitter:
    def __init__(self) -> None:
        print('Initialzing adc fitter')

        
class dti_fitter:
    def __init__(self) -> None:
        print('Initialzing dti fitter')