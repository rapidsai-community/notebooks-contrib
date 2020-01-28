# **TESTING.MD**

# Testing your notebook in Notebooks CI:
1. Restart your Docker container to a new pull state:
- `docker ps` to see all running docker images
- `docker restart < selected docker image >`
 
2. In terminal, navigate to the folder containing your target notebook

3. Fill in `notebook name` and run this command- `bash /rapids/utils/nbtest.sh <notebook name>`

4. If it returns `EXIT CODE 0`, you're good to PR!  
  If it returns `EXIT CODE 1`, please fix the issues found.

Please note:
1. This test just ensures that the notebook will execute.  This does NOT ensure that your notebooks output are what you believe they should be.  Adding print statements will help you diagnose those types of issue. 
1. This test does not work with magic functions, like (functions starting with `%`).  Please don't let your notebook's progression depend on these
1. Our notebook tests aren't compatible with Dask based notebooks at this time
