#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

using namespace std;

//*******************************************

// functors and device functions ===============================================

struct ThresholdToBinary {
  __device__
  int operator()(const int& value) const {
    return value > 0 ? 1 : 0;
  }
};

__device__ unsigned int closest_power_of_2(unsigned int val) {
    // Subtract 1 from val
    // and then do bitwise OR of all bits right of the leftmost 1 bit
    // This operation sets all bits to the right of the leftmost 1 bit to 1
    val--;

    // Bitwise OR operation with right-shift operation finds the next power of 2
    val |= val >> 1;
    val |= val >> 2;
    val |= val >> 4;
    val |= val >> 8;
    val |= val >> 16;

    // Increment the result to get the next power of 2
    return val + 1;
}

__device__ bool check_collinear(int x1 ,int y1 ,int x2 ,int y2 ,int x3 ,int y3)
{
  long long int a1 = x1,b1=y1,a2=x2,b2=y2,a3=x3,b3=y3;
  long long int az = a1 * (b2 - b3) + a2 * (b3 - b1) + a3 * (b1 - b2);
  if(az == 0) return 1;
  else return 0;


}

__device__ int distancecalc(int x1 ,int y1 ,int x2 ,int y2)
{
    int a1 = x1,b1=y1,a2=x2,b2=y2;
    int dist1 = (a1-a2);
    int dist2 = (b1-b2);
    if(dist1 < 0) dist1 *= -1;
    if(dist2 < 0) dist2 *= -1;
    return dist1+dist2;
}

__device__ int direction(int x1 ,int y1 ,int x2 ,int y2)
{
    if (x1 >= x2)
    {
        if (y1 > y2)
            return 1;
        return 2;
    }
    else
    {
        if (y1 > y2)
            return 3;
        return 4;
    }
}

// kernels ===================================================


__global__
void target_setting(int* tankdir ,int *xcoord,int *ycoord,int* HP , int T , int i)
{

    __shared__ int distance_matrix[1024];

    int tankid = blockIdx.x;
    int target = (tankid+i)%T;
    int inlinetank = threadIdx.x;
    int x1 = xcoord[tankid]  ,y1 = ycoord[tankid] ,x2 =xcoord[target] ,y2 =ycoord[target] ,x3 = xcoord[inlinetank] ,y3 = ycoord[inlinetank];
    distance_matrix[inlinetank] = INT_MAX;
    int distid = INT_MAX;


    // filtering the valid tanks and calculating their distance

    if(tankid != inlinetank && HP[inlinetank]>0 && check_collinear(x1,y1,x2,y2,x3,y3) && (direction(x1,y1,x2,y2) == direction(x1,y1,x3,y3)) )
    {

          target = inlinetank;
          distance_matrix[inlinetank] = distancecalc(x1,y1,x3,y3);
          distid = distancecalc(x1,y1,x3,y3);



    }
    
    __syncthreads();


  int T_pad = closest_power_of_2(T);

  for(int off = T_pad/2 ; off ; off /= 2)
  {
      if(inlinetank < off){

      if(inlinetank + off >= T)
      {
        atomicMin(&distance_matrix[inlinetank],INT_MAX);
      }
      else
      {
        atomicMin(&distance_matrix[inlinetank],distance_matrix[inlinetank+off]);

      }


    }
      __syncthreads();

  }



    if(distance_matrix[0] == INT_MAX)
    {
         tankdir[tankid] = -1;
    }
    else if(distance_matrix[0]  ==  distid)
    {
      
      tankdir[tankid] = inlinetank;
    }


}

__global__
void updatescore(int *score,int* HP ,int* tankdir)
{
  int tank = threadIdx.x;
  int target = tankdir[threadIdx.x];

  if(HP[tank] <= 0)
  {
      tankdir[tank] = -1;
  }

  if(tankdir[tank] != -1 )
  {
    atomicAdd(&score[tank],1);
    atomicSub(&HP[target],1);
  }

}


// Write down the kernels here


//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;


    FILE *inputfilepointer;

    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0;
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }


    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

// initial setup ===============================================================
  int *xcoord_d ,*ycoord_d, *score_d;

  thrust::device_vector<int> HP(T,H);
  thrust::device_vector<int> tankdir(T);

  // allocating cuda memory and copying it

  cudaMalloc(&xcoord_d , sizeof(int)*T);
  cudaMalloc(&ycoord_d , sizeof(int)*T);
  cudaMalloc(&score_d , sizeof(int)*T);


  cudaMemcpy(xcoord_d, xcoord, sizeof(int)*T, cudaMemcpyHostToDevice);
  cudaMemcpy(ycoord_d, ycoord, sizeof(int)*T, cudaMemcpyHostToDevice);

  // converting to raw pointer to pass in kernel


  int* tankdir_ptr = thrust::raw_pointer_cast(tankdir.data());
  int* HP_ptr = thrust::raw_pointer_cast(HP.data());


  // setup for looping

  
  //transform Reduce to get the sum of transformed elements
  
  int sum = thrust::transform_reduce(HP.begin(), HP.end(), ThresholdToBinary(), 0, thrust::plus<int>());
  int i=0;

  // starting loop

  while(sum >1 && ++i)
  {
      if(i % T == 0)
        continue;

      target_setting<<<T,T>>>(tankdir_ptr,xcoord_d,ycoord_d,HP_ptr,T,i);

      updatescore<<<1,T>>>(score_d, HP_ptr , tankdir_ptr );

     // Reduce to get the sum of transformed elements

      sum = thrust::transform_reduce(HP.begin(), HP.end(), ThresholdToBinary(), 0, thrust::plus<int>());
      cudaDeviceSynchronize();

  }

  cudaMemcpy(score, score_d, sizeof(int)*T, cudaMemcpyDeviceToHost);



    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}