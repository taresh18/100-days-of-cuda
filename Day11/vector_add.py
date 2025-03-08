import torch
import triton
import triton.language as tl

# define the device
DEVICE = 'cuda'

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  # program id ~ block id
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  # calculate offsets for each thread to access array elements
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  # out of bounds bound check
  mask = offsets < n_elements
  # load X and Y
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  # write output back to hbm
  tl.store(output_ptr + offsets, output, mask=mask)


def add(x, y):
  # preallocate the output
  output = torch.empty_like(x)
  n_elements = output.numel()
  # defing the grid dimension
  grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
  # call the kernel indexed with the grid element
  add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
  return output

# compare the outputs 
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
