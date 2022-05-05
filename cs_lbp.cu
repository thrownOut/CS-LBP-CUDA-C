%%cu
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Legend
// GX X GY is the dimension of spatial grid
// img is the image 
// m is the row size of image
// n is column size of image
// hist is the output histogram

__global__ void cs_lbp(int *img, int *hist, float T, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int p1[4][2] = {{0, 1}, {1, 1}, {1, 0}, {1, -1}};
    int p2[4][2] = {{0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
    int val = 0;
    for (int i = 0; i < 4; i++)
      val = (val<<1) | (img[(row + p1[i][0]) * n + (col + p1[i][1])] - img[(row + p2[i][0]) * n + (col + p2[i][1])] > T);
    hist[blockIdx.y * gridDim.x * 16 + blockIdx.x * 16 + val]++;
}

float CS_LBP(int **img, int ***hist, int m, int n, int GX, int GY, float threshold)
{
  int *d_img, *d_hist;
  int img_size = m * n * sizeof(int), hist_size = GX * GY * sizeof(int) * 16; 
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  cudaMalloc(&d_img, img_size);
  cudaMalloc(&d_hist, hist_size);
  cudaMemcpy(d_img, img, img_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_hist, hist, hist_size, cudaMemcpyHostToDevice);
  dim3 gridDim(GX, GY, 1), blockDim((n - 2) / GX, (m -2) / GY, 1);
  cs_lbp<<<gridDim, blockDim>>>(d_img, d_hist, threshold, n);
  cudaMemcpy(hist, d_hist, hist_size, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaFree(d_img);
  cudaFree(d_hist);
  return elapsedTime;
}

int m = 5, n = 5, GX = 3, GY = 3;
float threshold = 0.01;

int main()
{
  int img[5][5] = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, hist[GX][GY][16];
  float time, totalTime = 0, avgTime; 
  for (int i = 0; i < 100; i++)
  {
    for (int i = 0; i < GX; i++)
      for (int j = 0; j < GY; j++)
        for (int k = 0; k < 16; k++)
          hist[i][j][k] = 0;
    time = CS_LBP((int **)img, (int ***)hist, m, n, GX, GY, threshold);
    totalTime += time; 
  }
  avgTime = totalTime / 100;
  printf("Average Time: %f\n", avgTime);
  printf("Histogram: \n");
  for (int i = 0; i < GX; i++)
    for (int j = 0; j < GY; j++)
      for (int k = 0; k < 16; k++)
        printf("%d ", hist[i][j][k]);
  printf("\n");
  return 0;
}
