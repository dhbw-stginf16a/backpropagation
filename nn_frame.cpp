#include <cstdlib>
#include <iostream>
#include <ctime>
#include <math.h>

#include "backpropagation.h"
//---------------------------------------------------------------------------
// Example :  Assignment A
//---------------------------------------------------------------------------

#pragma argsused

int main(int argc, char* argv[])
{
  double in[5][2];
  double teach[5][4];

  feedForwardNetwork NN(2,5,4);

  int correctClassifications = 0;
  int lastCorrect = 0;
  int i,j;

  static float last_error = 1000.0f;

  double o[MAX_OUTPUT_LAYER_SIZE];
  double t[MAX_OUTPUT_LAYER_SIZE];
  double error,total_error=0.0f;
  bool  learned=false;
  char  buffer[50];
  int number = 5;
  int numberOfTestcases = 5;
  int iterations = 0;
  double old = 0.0f;
  int bps = 0;

  int    inputDim,hiddenDim,outputDim;
  double testIn0, testIn1, testOut;

  // The network is configured with 2 input neurons, 5 hidden
  // neurons and 4 output neurons (one for each class).

  inputDim  = 2;
  hiddenDim = 5;
  outputDim = 4;
  NN.configure(inputDim,hiddenDim,outputDim);
  NN.init();
  NN.setEpsilon(0.0001f);
  NN.setLearningRate(0.3f);

  fprintf(stderr, "Generate training dataset:\n");

    in[0][0] = (float)6/15.0f;
    in[0][1] = (float)9/15.0f;
    teach[0][0] = 0;
    teach[0][1] = 1;
    teach[0][2] = 0;
    teach[0][3] = 0;

    in[1][0] = (float)13/15.0f;
    in[1][1] = (float)12/15.0f;
    teach[1][0] = 1;
    teach[1][1] = 0;
    teach[1][2] = 0;
    teach[1][3] = 0;

    in[2][0] = (float)4/15.0f;
    in[2][1] = (float)4/15.0f;
    teach[2][0] = 0;
    teach[2][1] = 0;
    teach[2][2] = 0;
    teach[2][3] = 1;

    in[3][0] = (float)7/15.0f;
    in[3][1] = (float)5/15.0f;
    teach[3][0] = 0;
    teach[3][1] = 0;
    teach[3][2] = 1;
    teach[3][3] = 0;

    in[4][0] = (float)14/15.0f;
    in[4][1] = (float)15/15.0f;
    teach[4][0] = 1;
    teach[4][1] = 0;
    teach[4][2] = 0;
    teach[4][3] = 0;

  // note: input converted to [0,1] range (neuron netinput)

  for(i=0;i<number;i++)
  {
    fprintf(stderr, "[%2d] %2.0f %2.0f -> (%1.0f %1.0f %1.0f %1.0f)\n",i,in[i][0]*15,in[i][1]*15,teach[i][0],teach[i][1],teach[i][2],teach[i][3]);
  }

  fprintf(stderr, "Press enter to continue");
  getchar();

  fprintf(stderr, "\nStarting:\n");

  while (correctClassifications < number)
  {
    for (i = 0; i< number; i++)
    {
      iterations++;

      for (j=0; j<inputDim;j++)
      {
        NN.setInput(j,in[i][j]);
      }

      learned = false;
      bps = 0;
      while (!learned)
      {
        NN.apply();

        for (j=0;j<outputDim;j++)
        {
          o[j] = NN.getOutput(j);
        }

        for (j=0;j<outputDim;j++)
         t[j] = teach[i][j];

        error = NN.energy(t,o,outputDim);

        if (error > NN.getEpsilon())
        {
          NN.backpropagate(t);
          bps++;
        }
        else
          learned = true;
      }
      fprintf(stderr, "Backpropagations = %d\n",bps);

    }

    // get status of learning

    correctClassifications = 0;

    total_error = 0.0f;

    for (i = 0; i< number; i++)
    {
      for (j=0; j<inputDim;j++)
      {
        NN.setInput(j,in[i][j]);
      }

      NN.apply();

        for (j=0;j<outputDim;j++)
        {
          o[j] = NN.getOutput(j);
          t[j] = teach[i][j];
        }

      error = NN.energy(t,o,outputDim);
      total_error += error;

      if (error < NN.getEpsilon())
      {
        correctClassifications++;
      }
    }

    // total error

    last_error = total_error;
    if (lastCorrect != correctClassifications)
    {
     fprintf(stderr, "[%4d]>> Korrekte: %2d Fehler : %5.7f\n",iterations/number,correctClassifications, total_error);
     lastCorrect = correctClassifications;
    }

  }

  printf("Iterationen: %d\n", iterations/number);

  // save results in a file

  if ((fptr = fopen("test.csv","w")) == NULL)
  {
    printf("Fehler: Datei test.csv konnte nicht geöffnet werden.\n");
  }
  else
  {

    fprintf(fptr,"Epsilon %7.6f, Lernrate %7.6f,Iterationen %d \n",NN.getEpsilon(),NN.getLearningRate(),iterations/number);
    fprintf(fptr,"Istwert,Sollwert");

    // test with training cases (should be all fine ...)

    for (i=0;i<numberOfTestcases;i++)
    {

      testIn0 = in[i][0];
      testIn1 = in[i][1];

      printf("[%2.0f \t%2.0f] \t-> ",testIn0*15,testIn1*15);

      NN.setInput(0,testIn0);
      NN.setInput(1,testIn1);

      NN.apply();

      fprintf(fptr,"%5.4f;",NN.getOutput(0));
      fprintf(fptr,"%5.4f;",NN.getOutput(1));
      fprintf(fptr,"%5.4f;",NN.getOutput(2));
      fprintf(fptr,"%5.4f\n",NN.getOutput(3));

      printf("%5.4f;",NN.getOutput(0));
      printf("%5.4f;",NN.getOutput(1));
      printf("%5.4f;",NN.getOutput(2));
      printf("%5.4f\n",NN.getOutput(3));
    }
    printf("\n");
  }

  return 0;
}
//---------------------------------------------------------------------------
