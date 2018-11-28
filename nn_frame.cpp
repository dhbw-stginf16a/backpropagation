int main(int argc, char* argv[])
{
  FILE *fptr;

  double in[20][2];
  double teach[20][1];

  // configure net

  feedForwardNetwork *NN = new feedForwardNetwork(2,25,1);

  int correctClassifications = 0;
  int lastCorrect = 0;
  int i,j;

  static float last_error = 1000.0f;

  double o[MAX_OUTPUT_LAYER_SIZE];
  double t[MAX_OUTPUT_LAYER_SIZE];
  double error,total_error=0.0f;
  bool  learned=false;
  char  buffer[50];
  int number = 20;
  int numberOfTestcases = 20;
  int iterations = 0;
  double old = 0.0f;
  
  int    inputDim,hiddenDim,outputDim;
  double testIn0, testIn1, testOut;
  
  // The network is configured with 2 input neurons, 15 hidden
  // neurons and a single output neuron.
  
  inputDim = 2;
  hiddenDim = 25;
  outputDim = 1;
  NN->configure(inputDim,hiddenDim,outputDim);
  NN->init();
  NN->setEpsilon(0.001f);
  NN->setLearningRate(0.3f);
  
  /* provide learning data */


  /*
  for(i=0;i<number;i++)
  {
    in[i][0] = ...
    in[i][1] = ...
    teach[i][0] = ...
  }
  */
  
  // convert to [0,1] range (neuron netinput)
  
  for(i=0;i<number;i++)
  {
     in[i][0]= in[i][0]/2.0f+0.5f;
     in[i][1]= in[i][1]/2.0f+0.5f;
     teach[i][0]= teach[i][0]/2.0f+0.5f;
  }
  
  for(i=0;i<number;i++)
  {
    printf("[%2d] %5.4f %5.4f -> %5.4f\n",i,in[i][0],in[i][1],teach[i][0]);
  }
  
  printf("\nStarting:\n");

  while (correctClassifications < number)
  {
    for (i = 0; i< number; i++)
    {
      iterations++;

      for (j=0; j<inputDim;j++)
      {
        NN->setInput(j,in[i][j]);
      }

      learned = false;

      while (!learned)
      {
        NN->apply();

        for (j=0;j<outputDim;j++)
        {
          o[j] = NN->getOutput(j);
        }

        for (j=0;j<outputDim;j++)
         t[j] = teach[i][j];

        error = NN->energy(t,o,outputDim);
               
        if (error > NN->getEpsilon())
        {     
          NN->backpropagate(t);                   
        }
        else
          learned = true;
      }
      
      
    }

    // get status of learning

    correctClassifications = 0;

    total_error = 0.0f;

    for (i = 0; i< number; i++)
    {
      for (j=0; j<inputDim;j++)
      {
        NN->setInput(j,in[i][j]);
      }

      NN->apply();

        for (j=0;j<outputDim;j++)
        {
          o[j] = NN->getOutput(j);
          t[j] = teach[i][j];
        }

      error = NN->energy(t,o,outputDim);
      total_error += error;

      if (error < NN->getEpsilon())
      {
        correctClassifications++;
      }
    }

    // total error

    last_error = total_error;
    if (lastCorrect != correctClassifications)
    {
     printf("[%4d]>> Korrekte: %2d Fehler : %5.4f \n",iterations/number,correctClassifications, total_error);    
     lastCorrect = correctClassifications;
    }
    
  }

  printf("Iterationen: %d\n ", iterations/number);
  
  // convert from [0,1] range (neuron netinput)
  
  for(i=0;i<number;i++)
  {
     teach[i][0]= teach[i][0]*2.0f-1.0f;
  }
  
  // save results in a file
  
  if ((fptr = fopen("test.csv","w")) == NULL)
  {
    printf("Fehler: Datei test.csv konnte nicht ge�ffnet werden.\n");
  }
  else
  {
  
    fprintf(fptr,"Epsilon %7.6f, Lernrate %7.6f,Iterationen %d \n",NN->getEpsilon(),NN->getLearningRate(),iterations/number);
    fprintf(fptr,"Istwert,Sollwert");
    printf(" x(t-1)\t x(t)\t\tIstwert\t\tSollwert\n");
    
    // provide testcases
 
    for (i=0;i<numberOfTestcases;i++)
    {
      testIn0 = ...
      testIn1 = ...
      testOut = ...

      printf("[%5.4f \t%5.4f] \t-> ",testIn0,testIn1);

      NN->setInput(0,testIn0);
      NN->setInput(1,testIn1);

      NN->apply();

      // compare result to testresult
    
      fprintf(fptr,"%5.4f;",NN->getOutput(0));
      fprintf(fptr,"%5.4f \n",testOut);

      printf("%5.4f \t",NN->getOutput(0));
      printf("(%5.4f)\n",testOut);
    }
    fprintf(fptr,"\n");
  
    fclose(fptr);
  }
  system("PAUSE");

  return 0;
}
//---------------------------------------------------------------------------
