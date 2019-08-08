import pystan
import hashlib
import os
import joblib


stan_compiled = os.path.dirname(__file__) + '/../stan_compiled/'
stan_code = os.path.dirname(__file__) + '/../stan_code/'

def compile_code(code, code_name):
    """
   Compile a stan code in the stan_code directory store in stan_compiled directory
   Avoids repeat compilation by checking hashes

    parameters 
    code_name: string
        the name of the stan code to compile
    code: string
         the code locatoin 
    """
    hash_name = hashlib.md5(code.encode('utf-8')).hexdigest()[0:10]
    if not os.path.exists(stan_compiled + hash_name + '.stanc'): #check if compilation already exists
        print('compiled ' + code_name)
        c_code = pystan.StanModel(model_code=code)
        joblib.dump(c_code, stan_compiled + code_name + '.stan')
        joblib.dump(' ', stan_compiled + hash_name + '.stanc')
    else:
        print('already compiled ' + code_name)


def compile_all():
    """
    Runs compile_code on all the stan codes in the stan code directory
    """
    print('This may take several minutes')
    files = os.listdir(stan_code)
    for f in files:
        if '.st' in f:
            with open(stan_code + f, "r") as myfile:
                data = myfile.readlines()
            code = ''.join(data)
            compile_code(code, f[:-3])
    print('all done')

