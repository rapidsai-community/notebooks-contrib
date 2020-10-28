import sys
import argparse
import time
import threading
import subprocess
import socket

from mpi4py import MPI
from azureml.core import Run
from notebook.notebookapp import list_running_servers

def flush(proc, proc_log):
  while True:
    proc_out = proc.stdout.readline()
    if proc_out == '' and proc.poll() is not None:
      proc_log.close()
      break
    elif proc_out:
      sys.stdout.write(proc_out)
      proc_log.write(proc_out)
      proc_log.flush()


if __name__ == '__main__':
  
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  ip = socket.gethostbyname(socket.gethostname())

  parser = argparse.ArgumentParser()
  parser.add_argument("--datastore")
  parser.add_argument("--n_gpus_per_node")
  parser.add_argument("--jupyter_token")
  args = parser.parse_args()

  n_gpus_per_node = eval(args.n_gpus_per_node)

  print("number of GPUs per node:", n_gpus_per_node)
  print("- my rank is ", rank)
  print("- my ip is ", ip)
  
  if rank == 0:
    cluster = {
      "scheduler"  : ip + ":8786",
      "dashboard"  : ip + ":8787"
    }
    scheduler = cluster["scheduler"]
    dashboard = cluster["dashboard"]
  else:
    cluster = None
  
  cluster = comm.bcast(cluster, root=0)
  scheduler = cluster["scheduler"]
  dashboard = cluster["dashboard"]
  
  if rank == 0:
    Run.get_context().log("headnode", ip)
    Run.get_context().log("cluster",
                          "scheduler: {scheduler}, dashboard: {dashboard}".format(scheduler=cluster["scheduler"],
                                                                                  dashboard=cluster["dashboard"]))
    Run.get_context().log("datastore", args.datastore)

    cmd = ("jupyter lab --ip 0.0.0.0 --port 8888" + \
                      " --NotebookApp.token={token}" + \
                      " --allow-root --no-browser").format(token=args.jupyter_token)
    jupyter_log = open("jupyter_log.txt", "a")
    jupyter_proc = subprocess.Popen(cmd.split(), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    jupyter_flush = threading.Thread(target=flush, args=(jupyter_proc, jupyter_log))
    jupyter_flush.start()

    while not list(list_running_servers()):
      time.sleep(5)

    jupyter_servers = list(list_running_servers())
    assert (len(jupyter_servers) == 1), "more than one jupyter server is running"

    Run.get_context().log("jupyter",
                          "ip: {ip_addr}, port: {port}".format(ip_addr=ip, port=jupyter_servers[0]["port"]))
    Run.get_context().log("jupyter-token", jupyter_servers[0]["token"])

    cmd = "dask-scheduler " + "--port " + scheduler.split(":")[1] + " --dashboard-address " + dashboard
    scheduler_log = open("scheduler_log.txt", "w")
    scheduler_proc = subprocess.Popen(cmd.split(), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(list(range(n_gpus_per_node))).strip("[]")
    cmd = "dask-cuda-worker " + scheduler + " --memory-limit 0"
    worker_log = open("worker_{rank}_log.txt".format(rank=rank), "w")
    worker_proc = subprocess.Popen(cmd.split(), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    worker_flush = threading.Thread(target=flush, args=(worker_proc, worker_log))
    worker_flush.start()

    flush(scheduler_proc, scheduler_log)
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(list(range(n_gpus_per_node))).strip("[]")
    cmd = "dask-cuda-worker " + scheduler + " --memory-limit 0"
    worker_log = open("worker_{rank}_log.txt".format(rank=rank), "w")
    worker_proc = subprocess.Popen(cmd.split(), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    flush(worker_proc, worker_log)

