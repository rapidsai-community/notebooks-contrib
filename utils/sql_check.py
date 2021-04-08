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
        import blazingsql
        # yes, indicate success
        return "You've got BlazingSQL set up perfectly! Let's get started with SQL in RAPIDS AI!"
    # BlazingSQL not found
    except ModuleNotFoundError:
            # do we want to install BlazingSQL?
            print('Unable to locate BlazingSQL.  Please install it from https://rapids.ai/start.html#rapids-release-selector using "RAPIDS and BlazingSQL" with your current system configuration selected')
#             # Install JRE first
#             os.system("apt-get update")
#             os.system("apt-get -y install default-jre")
#             # tag BlazingSQL conda install script
#             b = "conda install -c blazingsql/label/cuda10.0 -c blazingsql" 
#             b += ' -c rapidsai -c nvidia -c conda-forge -c defaults '
#             b += "blazingsql python=3.7 cudatoolkit=10.0"  # CUDA 10, Python 3.7 (BlazingSQL also supports CUDA 9.2)
#             # tag python version
#             py = sys.version.split('.')  # e.g. output: ['3', '6', '7 | packaged by cond...
#             if py[0] == '3':  # make sure we're in 3
#                 py = py[1]  # focus mid version (3.?)
#             # are we on python 3.6?
#             if py == '6':
#                 # adjust to 3.6 install script
#                 b = b.replace('python=3.7', 'python=3.6')
#             # lmk what's going on?
#             print('Installing BlazingSQL, this should take some time.  This is only need to be done once')
#             # install BlazingSQL 
#             os.system(b)
#             # indicate completion
#             return f"Let's get started with SQL in RAPIDS AI!"
            return f"After you've installed RAPIDS and BlazingSQL, please try this notebook again."
    except FileNotFoundError:    
        print('Unable to locate BlazingSQL.  Please install it from https://rapids.ai/start.html#rapids-release-selector using "RAPIDS and BlazingSQL" with your current system configuration selected')
        return f"After you've installed RAPIDS and BlazingSQL, please try this notebook again."            
            
if __name__=='__main__':
    # check environment for BlazingSQL
    check = bsql_start()
    print(check)
