#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Legend
// GX X GY is the dimension of spatial grid
// img is the image 
// m is the row size of image
// n is column size of image
// hist is the output histogram

__global__ void cs_lbp(int *img, int *hist, int T, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int p1[4][2] = {{0, 1}, {1, 1}, {1, 0}, {1, -1}};
    int p2[4][2] = {{0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
    int val = 0;
    for (int i = 0; i < 4; i++)
    {
      val <<= 1;
      if (img[(row + p1[i][0]) * n + (col + p1[i][1])] - img[(row + p2[i][0]) * n + (col + p2[i][1])] > T)
        val |= 1;
    }
    hist[blockIdx.y * gridDim.x * 16 + blockIdx.x * 16 + val]++;
}

int GX = 3, GY = 3, m = 5, n = 5, threshold = 0;

int main()
{
  int img[5][5] = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, hist[GX][GY][16];
  for (int i = 0; i < GX; i++)
    for (int j = 0; j < GY; j++)
      for (int k = 0; k < 16; k++)
          hist[i][j][k] = 0;
  int *d_img, *d_hist;
  int img_size = m * n * sizeof(int), hist_size = GX * GY * sizeof(int) * 16; 
  cudaMalloc(&d_img, img_size);
  cudaMalloc(&d_hist, hist_size);
  cudaMemcpy(d_img, img, img_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_hist, hist, hist_size, cudaMemcpyHostToDevice);
  dim3 gridDim(GX, GY, 1), blockDim((m - 2) / GX, (n -2) / GY, 1);
  cs_lbp<<<gridDim, blockDim>>>(d_img, d_hist, threshold, n);
  cudaMemcpy(hist, d_hist, hist_size, cudaMemcpyDeviceToHost);
  printf("Histogram: \n");
  for (int i = 0; i < GX; i++)
    for (int j = 0; j < GY; j++)
      for (int k = 0; k < 16; k++)
        printf("%d ", hist[i][j][k]);
  printf("\n");
  cudaFree(d_img);
  cudaFree(d_hist);
  return 0;
}
