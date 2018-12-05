#include <cstdlib>
#include <iostream>
#include <ctime>
#include <math.h>

#include "backpropagation.h"
//---------------------------------------------------------------------------
// Example :  Assignment A
//---------------------------------------------------------------------------

#pragma argsused

#define INPUT_NEURONS 2
#define HIDDEN_NEURONS 5
#define OUTPUT_NEURONS 4

#define TRAINING_EXAMPLES 5
#define TEST_CASES 15

int main(int argc, char* argv[])
{
  double in[TRAINING_EXAMPLES][INPUT_NEURONS];
  double teach[TRAINING_EXAMPLES][OUTPUT_NEURONS];

  feedForwardNetwork NN(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS);

  int correctClassifications = 0;
  int lastCorrect = 0;

  static float last_error = 1000.0f;

  double o[MAX_OUTPUT_LAYER_SIZE];
  double t[MAX_OUTPUT_LAYER_SIZE];
  double error,total_error=0.0f;
  bool  learned=false;
  char  buffer[50];
  int iterations = 0;
  double old = 0.0f;
  int bps = 0;

  double testIn[TEST_CASES][INPUT_NEURONS];
  double testOut[TEST_CASES][OUTPUT_NEURONS];

  // The network is configured with 2 input neurons, 5 hidden
  // neurons and 4 output neurons (one for each class).

  NN.configure(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS);
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

  for (int i = 0; i < TRAINING_EXAMPLES; i++) {
    fprintf(stderr, "[%2d] %2.0f %2.0f -> (%1.0f %1.0f %1.0f %1.0f)\n",i,in[i][0]*15,in[i][1]*15,teach[i][0],teach[i][1],teach[i][2],teach[i][3]);
  }

  //fprintf(stderr, "Press enter to continue");
  //getchar();

  fprintf(stderr, "\nStarting:\n");

  while (correctClassifications < TRAINING_EXAMPLES)
  {
    for (int i = 0; i < TRAINING_EXAMPLES; i++)
    {
      iterations++;

      for (int j = 0; j < INPUT_NEURONS;j++)
      {
        NN.setInput(j,in[i][j]);
      }

      learned = false;
      bps = 0;
      while (!learned)
      {
        NN.apply();

        for (int j = 0; j < OUTPUT_NEURONS;j++)
        {
          o[j] = NN.getOutput(j);
        }

        for (int j=0;j < OUTPUT_NEURONS;j++)
         t[j] = teach[i][j];

        error = NN.energy(t,o,OUTPUT_NEURONS);

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

    for (int i = 0; i < TRAINING_EXAMPLES; i++)
    {
      for (int j = 0; j < INPUT_NEURONS;j++)
      {
        NN.setInput(j,in[i][j]);
      }

      NN.apply();

        for (int j=0;j<OUTPUT_NEURONS;j++)
        {
          o[j] = NN.getOutput(j);
          t[j] = teach[i][j];
        }

      error = NN.energy(t,o,OUTPUT_NEURONS);
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
     fprintf(stderr, "[%4d]>> Korrekte: %2d Fehler : %5.7f\n",
             iterations / TRAINING_EXAMPLES, correctClassifications, total_error);
     lastCorrect = correctClassifications;
    }

  }

  fprintf(stderr, "Iterationen: %d\n", iterations / TRAINING_EXAMPLES);

  printf("Epsilon %7.6f, Lernrate %7.6f,Iterationen %d \n",
         NN.getEpsilon(), NN.getLearningRate(), iterations / TRAINING_EXAMPLES);
  printf("Istwert,Sollwert\n");

  // test with training cases (should be all fine ...)
  testIn[0][0] = (float) 6/15.0f;
  testIn[0][1] = (float) 9/15.0f;
  testOut[0][0] = 0;
  testOut[0][1] = 1;
  testOut[0][2] = 0;
  testOut[0][3] = 0;

  testIn[1][0] = (float) 12/15.0f;
  testIn[1][1] = (float) 11/15.0f;
  testOut[1][0] = 1;
  testOut[1][1] = 0;
  testOut[1][2] = 0;
  testOut[1][3] = 0;

  testIn[2][0] = (float) 12/15.0f;
  testIn[2][1] = (float) 4/15.0f;
  testOut[2][0] = 0;
  testOut[2][1] = 1;
  testOut[2][2] = 0;
  testOut[2][3] = 0;

  testIn[3][0] = (float) 15/15.0f;
  testIn[3][1] = (float) 10/15.0f;
  testOut[3][0] = 1;
  testOut[3][1] = 0;
  testOut[3][2] = 0;
  testOut[3][3] = 0;

  testIn[4][0] = (float) 7/15.0f;
  testIn[4][1] = (float) 4/15.0f;
  testOut[4][0] = 0;
  testOut[4][1] = 0;
  testOut[4][2] = 1;
  testOut[4][3] = 0;

  testIn[5][0] = (float) 7/15.0f;
  testIn[5][1] = (float) 9/15.0f;
  testOut[5][0] = 0;
  testOut[5][1] = 1;
  testOut[5][2] = 0;
  testOut[5][3] = 0;

  testIn[6][0] = (float) 3/15.0f;
  testIn[6][1] = (float) 1/15.0f;
  testOut[6][0] = 0;
  testOut[6][1] = 0;
  testOut[6][2] = 0;
  testOut[6][3] = 1;

  testIn[7][0] = (float) 12/15.0f;
  testIn[7][1] = (float) 2/15.0f;
  testOut[7][0] = 0;
  testOut[7][1] = 0;
  testOut[7][2] = 1;
  testOut[7][3] = 0;

  testIn[8][0] = (float) 15/15.0f;
  testIn[8][1] = (float) 15/15.0f;
  testOut[8][0] = 1;
  testOut[8][1] = 0;
  testOut[8][2] = 0;
  testOut[8][3] = 0;

  testIn[9][0] = (float) 7/15.0f;
  testIn[9][1] = (float) 6/15.0f;
  testOut[9][0] = 0;
  testOut[9][1] = 0;
  testOut[9][2] = 1;
  testOut[9][3] = 0;

  testIn[10][0] = (float) 15/15.0f;
  testIn[10][1] = (float) 6/15.0f;
  testOut[10][0] = 1;
  testOut[10][1] = 0;
  testOut[10][2] = 0;
  testOut[10][3] = 0;

  testIn[11][0] = (float) 1/15.0f;
  testIn[11][1] = (float) 10/15.0f;
  testOut[11][0] = 0;
  testOut[11][1] = 0;
  testOut[11][2] = 1;
  testOut[11][3] = 0;

  testIn[12][0] = (float) 10/15.0f;
  testIn[12][1] = (float) 7/15.0f;
  testOut[12][0] = 0;
  testOut[12][1] = 1;
  testOut[12][2] = 0;
  testOut[12][3] = 0;

  testIn[13][0] = (float) 3/15.0f;
  testIn[13][1] = (float) 6/15.0f;
  testOut[13][0] = 0;
  testOut[13][1] = 0;
  testOut[13][2] = 0;
  testOut[13][3] = 1;

  testIn[14][0] = (float) 0/15.0f;
  testIn[14][1] = (float) 4/15.0f;
  testOut[14][0] = 0;
  testOut[14][1] = 0;
  testOut[14][2] = 0;
  testOut[14][3] = 1;

  fflush(stdout);
  double testEpsilon = 0.1;
  for (int i = 0; i < TEST_CASES; i++) {
    fprintf(stderr, "[%2.0f \t%2.0f] \t-> ", testIn[i][0] * 15.0, testIn[i][1] * 15.0);
    fflush(stderr);

    NN.setInput(0, testIn[i][0]);
    NN.setInput(1, testIn[i][1]);

    NN.apply();

    fflush(stdout);

    for (int k = 0; k < 4; k++) {
      if (testOut[i][k])
        printf("\033[1m");

      if (abs(NN.getOutput(k) - testOut[i][k]) < testEpsilon)
        printf("\033[32m");
      else
        printf("\033[31m");

      printf("%5.4f;",NN.getOutput(k));
      printf("\033[0m");
    }
    printf("\n");
  }

  return 0;
}
//---------------------------------------------------------------------------
