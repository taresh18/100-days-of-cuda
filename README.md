# 100-days-of-cuda
100 days of learning cuda

### Day 01
Read chapter 2 of PMPP, wrote cuda program to add two 1d vectors

### Day 02
Read chapter 3 of PMPP, wrote cuda programs to convert rgb image to greyscale, blur a 2d image, 2d matrix multiplication

### Day 03
Read chapter 4 of PMPP, learnt about warps, blocks, SMs, how threads are assigned to a SM, occupancy.

### Day 04
Read chapter 5 of PMPP, wrote cuda program to do matrix multiplication using shared mamory to reduce number of global memory access

### Day 05
Read chapter 6 of PMPP, wrote cuda programs to do matrix multiplication (A * B.T) using corner turning and matrix multiplication using thread coarsening

### Day 06
Read chapter 7 of PMPP, wrote cuda program to do a 2d convolution operation reducing global memory access using shread memory and local memory

### Day 07
Read chapter 9 of PMPP, wrote cuda program to compute histogram. Applied shared memory usage, thread coarsening and memory coalescing to optimise the vanilla implementation.

### Day 08
Read chapter 10 of PMPP, wrote cuda program to do reduction (addition) on a 1d array. Applied memory coalescing, thread coarsening, heirarchial reduction etc. to optimise.

### Day 09
Read half of the chapter 11 of PMPP. wrote cuda prgram to do prefix sum via kogg-stone algorithm.

### Day 10
Read rest of the chapter 11 of PMPP. wrote cuda prgram to do prefix sum via brent-kung algorithm plus with its thread coarsened version.
