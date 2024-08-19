/*
	CS 6023 Assignment 3. 
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>


void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input.
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;


	int meshX, meshY ;
	int globalPositionX, globalPositionY; 
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ;
		scenes.push_back (scene) ;
	}

	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ;
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL;
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}

	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}


__global__ void doTranslation(int * allTranslations, int* gpuX, int* gpuY,int n){

	if(blockDim.x*blockIdx.x + threadIdx.x < n){
		int translationId = (blockDim.x*blockIdx.x + threadIdx.x) * 3;
		int node = allTranslations[translationId + 0];
		int dir = allTranslations[translationId + 1];
		int amount = allTranslations[translationId + 2];
		int nexX = 0;
		int newY = 0;
		switch (dir)
		{
			case 0:nexX = -amount;break;
			case 1:nexX = amount;break;
			case 3:newY = amount;break;
			default:newY = -amount;
		}
		atomicAdd(&gpuY[node], updateY);
		atomicAdd(&gpuX[node], nexX);
			
	}
}

__global__ void additiveUpdate(int i, int j, int* dCsr, int node, int* gpuX, int* gpuY) {
    if ((blockIdx.x * blockDim.x + threadIdx.x) < j) {
		if(i <= (blockIdx.x * blockDim.x + threadIdx.x)){
			gpuX[dCsr[(blockIdx.x * blockDim.x + threadIdx.x)]] = gpuX[dCsr[(blockIdx.x * blockDim.x + threadIdx.x)]] + gpuX[node];
			gpuY[dCsr[(blockIdx.x * blockDim.x + threadIdx.x)]] = gpuY[dCsr[(blockIdx.x * blockDim.x + threadIdx.x)]] + gpuY[node];
		}
    }
}

__global__ void printMesh( int* grid, int* X, int* Y, int* dUpdateX, int* dUpdateY, int *GlobalScene, int* globalOpacity,int i, int j, int m, int n, int o, int gridIdx){

	int gridId = blockIdx.x * blockDim.x;
    int translationID = gridId + threadIdx.x;
    int col = translationID % j;
    int row = translationID / j;


    if( row >= 0 && row < i && col >= 0 && col < j){
      int globalX = row + X[gridIdx] + dUpdateX[gridIdx];
      int globalY = col + Y[gridIdx] + dUpdateY[gridIdx];
    
      if (globalX >= 0 && globalX < m && globalY >= 0 && globalY < n)
      {
		int rowNum = globalX * n
		if (globalOpacity[rowNum + globalY] < o)
		{
			GlobalScene[rowNum + globalY] = grid[row * j + col];
			globalOpacity[rowNum + globalY] = o;
		}
      }
    }

}


int main (int argc, char **argv) {

	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ;

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;

	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;


	// Code begins here.
	// Do not change anything above this comment.

	int tempSize =sizeof(int) * V;
	int *gpuCsr;
	cudaMalloc(&gpuCsr,tempSize);
	cudaMemcpy(gpuCsr, hCsr,tempSize, cudaMemcpyHostToDevice);

    int *gpuMatrix;
	cudaMalloc(&gpuMatrix, sizeof(int) * frameSizeX * frameSizeY);
    cudaMemset(&gpuMatrix, 0, sizeof(int) * frameSizeX * frameSizeY);
    
    int *gpuMatrixOpacity;
	cudaMalloc(&gpuMatrixOpacity, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemset(&gpuMatrixOpacity, -1, sizeof(int) * frameSizeX * frameSizeY);



	int *gpuX;
	cudaMalloc(&gpuX,tempSize);
	cudaMemcpy(gpuX, hGlobalCoordinatesX,tempSize, cudaMemcpyHostToDevice);

	int *gpuY;
	cudaMalloc(&gpuY,tempSize);
	cudaMemcpy(gpuY, hGlobalCoordinatesY,tempSize, cudaMemcpyHostToDevice);

	int *dUpdateX;
	cudaMalloc(&dUpdateX,tempSize);
	cudaMemset(dUpdateX, 0, V*sizeof(int));

	int *dUpdateY;
  	cudaMalloc(&dUpdateY,tempSize);
	cudaMemset(dUpdateY, 0, V*sizeof(int));
	
	int* dTranslations;
	cudaMalloc(&dTranslations, sizeof(int) * numTranslations * 3);
	int index = 0; 
	for (;index < numTranslations; index+=1) {
		cudaMemcpy(dTranslations + index * 3, translations[index].data(), sizeof(int) * 3, cudaMemcpyHostToDevice);
	}
	int gridSize = ceil(float(numTranslations)/1024);
    doTranslation<<<gridSize, 1024>>>(dTranslations, dUpdateX, dUpdateY, numTranslations);


	std::queue<int> queu = {0};
  while (!queu.empty()) {
        int current = queu.front();
		int next = current + 1;
        queu.pop();
		int i = hOffset[current];
		int j = hOffset[next];
        int v = i;
		for (; v < j;) {
            queu.push(hCsr[v]);
			v+=1;
        }

		if(j - i> 0){
				int gridSize = ceil(float(V) / 1024);
				additiveUpdate<<<gridSize, 1024>>>(i, j, gpuCsr, current, dUpdateX, dUpdateY);
			}
		}

	int v = 0;
	for ( ;v < V; v+=1) {
		vector<int> meshCord = { hFrameSizeX[v], hFrameSizeY[v]};
		int opacity = hOpacity[v];
		int* dMesh;
		int meshSize = 
		cudaMalloc(&dMesh, sizeof(int) * meshCord[0] * meshCord[1]);
		cudaMemcpy(dMesh, hMesh[v], sizeof(int) * meshCord[0] * meshCord[1], cudaMemcpyHostToDevice);
		float threads = 1024.0;
		float temp = meshCord[0]*meshCord[1];
		int gd = ceil(temp/ threads);
		printMesh<<<gd, (int)threads>>>(dMesh, gpuX, gpuY, dUpdateX, dUpdateY, gpuMatrix, gpuMatrixOpacity, meshCord[0], meshCord[1], frameSizeX, frameSizeY, opacity, v);

	}
	cudaMemcpy(hFinalPng, gpuMatrix, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);
  // Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;

}
