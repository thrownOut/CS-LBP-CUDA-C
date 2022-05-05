%%cu
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* 
  Legend
  GX X GY is the dimension of spatial grid
  img is the image 
  M is the row size of image
  N is column size of image
  hist is the output histogram
*/

#define GX 3
#define GY 3
#define M 5
#define N 5
#define threshold 0.01

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

float CS_LBP_Parallel(int img[][N], int hist[][GY][16], int m, int n, int gx, int gy, float T)
{
  int *d_img, *d_hist;
  int img_size = m * n * sizeof(int), hist_size = gx * gy * sizeof(int) * 16; 
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  cudaMalloc(&d_img, img_size);
  cudaMalloc(&d_hist, hist_size);
  cudaMemcpy(d_img, img, img_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_hist, hist, hist_size, cudaMemcpyHostToDevice);
  dim3 gridDim(gx, gy, 1), blockDim((n - 2) / gx, (m -2) / gy, 1);
  cs_lbp<<<gridDim, blockDim>>>(d_img, d_hist, T, n);
  cudaMemcpy(hist, d_hist, hist_size, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaFree(d_img);
  cudaFree(d_hist);
  return elapsedTime;
}

float CS_LBP_Sequential(int img[][N], int hist[][GY][16], int m, int n, int gx, int gy, float T)
{
  int p1[4][2] = {{0, 1}, {1, 1}, {1, 0}, {1, -1}};
  int p2[4][2] = {{0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
  clock_t start, end;
  start = clock();
  for (int row = 1; row <= (m - 2) / gy * gy; row++)
  {
    for (int col = 1; col <= (n - 2) / gx * gx; col++)
    {
      int val = 0;
      for (int i = 0; i < 4; i++)
        val = (val<<1) | (img[row + p1[i][0]][col + p1[i][1]] - img[row + p2[i][0]][col + p2[i][1]] > T);
      hist[(row - 1) / ((m - 2) / gy)][(col - 1) / ((n - 2) / gx)][val]++;
    }
  }
  end = clock();
  float timeElapsed = ((float)(end - start)) / CLOCKS_PER_SEC;
  return timeElapsed;
}

void clear(int hist[][GY][16])
{
  for (int i = 0; i < GX; i++)
    for (int j = 0; j < GY; j++)
      for (int k = 0; k < 16; k++)
        hist[i][j][k] = 0;
}

void print(int hist[][GY][16])
{
  printf("Histogram: \n");
  for (int i = 0; i < GX; i++)
    for (int j = 0; j < GY; j++)
      for (int k = 0; k < 16; k++)
        printf("%d ", hist[i][j][k]);
  printf("\n");
}

int main()
{
  int img[M][N] = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, hist[GX][GY][16];
  float time, totalTime = 0, avgTime; 
  for (int i = 0; i < 100; i++)
  {
    clear(hist);
    time = CS_LBP_Parallel(img, hist, M, N, GX, GY, threshold);
    totalTime += time; 
  }
  avgTime = totalTime / 100;
  printf("Average Time of Parallel Execution: %f\n", avgTime);
  clear(hist);
  time = CS_LBP_Sequential(img, hist, M, N, GX, GY, threshold);
  printf("Time of Sequential Execution: %f\n", time);
  print(hist);
  return 0;
}

/*
Output :- 
Parallel - 0.115621
Sequential - 0.000002
*/