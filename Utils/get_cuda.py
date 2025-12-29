import torch
import re
from typing import Optional


def get_gpuinfo(
        device,
        min_free_mem_mb: Optional[int] = None,
):

    # If CPU is used, GPU info is not applicable
    if device.type != "cuda":
        print("Execution device set to CPU.")
        return

    idx = device.index

    print('####################  GPU INFO  ##########################')
    print('Available GPU count :', torch.cuda.device_count())
    print('Selected GPU        :', torch.cuda.get_device_name(idx))

    # ---- GPU properties ----
    props = torch.cuda.get_device_properties(idx)
    total_mem_mb = props.total_memory / 1024 ** 2

    # ---- This process usage ----
    allocated_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
    reserved_mb = torch.cuda.memory_reserved(device) / 1024 ** 2

    # ---- REAL GPU usage (like nvidia-smi) ----
    free_b, total_b = torch.cuda.mem_get_info(device)
    free_real_mb = free_b / 1024 ** 2
    total_real_mb = total_b / 1024 ** 2
    used_real_mb = total_real_mb - free_real_mb

    print(f"[Device] Using {device}")
    print(f"  ├─ Name                : {props.name}")
    print(f"  ├─ Total memory        : {total_mem_mb:.1f} MB")
    print(f"  ├─ Used (GPU total)    : {used_real_mb:.1f} MB")
    print(f"  ├─ Free  (GPU total)   : {free_real_mb:.1f} MB")
    print(f"  ├─ Reserved (process)  : {reserved_mb:.1f} MB")
    print(f"  └─ Allocated(process) : {allocated_mb:.1f} MB")

    if min_free_mem_mb is not None:
        if free_real_mb < min_free_mem_mb:
            print(f"[Warning] Not enough free GPU memory "
                  f"({free_real_mb:.1f} MB < {min_free_mem_mb} MB)")

    print('###########################################################')



def is_valid_cuda_device_str(device_str: str) -> bool:
    """
        Validate CUDA device string format.

        Valid format:
            - 'cuda:N' where N is a non-negative integer
    """
    pattern = r"^cuda:\d+$"
    return device_str is not None and re.match(pattern, device_str.lower()) is not None



def get_cuda(
        gpu_num: str = None,
        verbose: bool = False,
):
    """
    Select and return a torch.device based on user input.

    Args:
        gpu_num (str): CUDA device string (e.g., 'cuda:0')
        verbose (bool): If True, print GPU information

    Returns:
        torch.device
    """

    # If CUDA is not available, fallback to CPU
    if torch.cuda.is_available():

        # Validate CUDA device string
        if is_valid_cuda_device_str(gpu_num):
            # gpu_num is None and
            gpu_index = int(gpu_num.split(':')[-1])
            gpu_count = torch.cuda.device_count()

            # Check GPU index range
            if gpu_index <= gpu_count:
                print("Selected GPU is : ", gpu_num)
                device = torch.device(gpu_num)

                if verbose:
                    get_gpuinfo(device)

                return device
            else:
                print('Available GPU count is : ', gpu_count)
                print('Your gpu_num :', gpu_num)
                print("Please edit the CUDA index you provided.")

        # Invalid format → fallback to cuda:0
        else:
            print('Check value or type of *gpu_num* in get_cuda()')
            print('Running on cuda:0')

            return torch.device("cuda:0")

    else:
        print('GPU not available............')
        print('Falling back to CPU!!!!!!!!!!!!!!!!!!')

        return torch.device('cpu')



