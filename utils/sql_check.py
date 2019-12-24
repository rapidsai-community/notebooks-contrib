# Notebooks-contrib test 
import os
import sys  


def bsql_start():
    """
    set up users to use SQL in the RAPIDS AI ecosystem
        > try to import BlazingContext from BlazingSQL
        > offer to install BlazingSQL if module not found
    BlazingSQL prereqs: 
        > Conda: https://docs.blazingdb.com/docs/install-via-conda#section-conda-prerequisites
        > Docker: https://docs.blazingdb.com/docs/install-via-docker#section-docker-hub-prerequisites
    latest install scripts: 
        > Conda: https://docs.blazingdb.com/docs/install-via-conda
        > Docker: https://docs.blazingdb.com/docs/install-via-docker
        > Source: https://docs.blazingdb.com/docs/build-from-source
    """
    # is BlazingSQL installed?
    try:
        # from blazingsql import BlazingContext
        print("You've got BlazingSQL set up perfectly! Let's get started with SQL in RAPIDS AI!")
    # BlazingSQL not found
    except ModuleNotFoundError:
        # install BlazingSQL?
        ask = input('Unable to locate BlazingSQL, would you like to install it now? [y/n]')
        # account for input error (extra spaces or CAPS)
        ask = ask.strip().lower()
        # yes install now (account for error input w/ strip & lower)
        if (ask == 'y') or (ask == 'yes'):      
            # tag BlazingSQL conda install script
            b = "conda install -c blazingsql/label/cuda10.0 -c blazingsql" 
            b += ' -c rapidsai -c nvidia -c conda-forge -c defaults '
            b += "blazingsql python=3.7 cudatoolkit=10.0"  # CUDA 10, Python 3.7 (BlazingSQL also supports CUDA 9.2)
            # check python version
            py = sys.version.split('.')[1]
            # are we on python 3.6?
            if py == '6':
                # adjust to 3.6 install script
                b = b.replace('python=3.7', 'python=3.6')
            # what's going on?
            print('Installing BlazingSQL, this may take a few minutes.')
            # install BlazingSQL
            os.system(b)
        # no, don't install now
        else:
            # indicate so
            docs = 'https://docs.blazingdb.com/docs/install-via-conda'
            print(f'Ok, not installing now.\nTo download later simply rerun this script or go to {docs}')
            
            
if __name__=='__main__':
    # check environment for BlazingSQL
    check = bsql_start()
    print(check)
