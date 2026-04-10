
#include <iostream>
#include <stdio.h>
#include "heat_lattice.h"

using std::cout;
using std::endl;

/*******************************************************************
functions that are commented out can serve for a provisoric mineral-
transformation.
to enable them, turn on the "previous_heat_distribution"-array
in heat_lattice.h.
 
also in min_trans.cc some functions have to be commented out.
*********************************************************************/

Heat_Lattice::Heat_Lattice()
{}



void Heat_Lattice::set_Parameters(float y_wall, float x_wall, float now)
{
  //inital_diameter of a single particle at the beginning of the run. scales the stepsize

  float factor;
  int counter2;

  time = float(now)/2.0;
  col = int(sqrt(float(numParticles)));
  delta_x = x_wall / col;
  delta_y = y_wall / col;

  alpha = (diffusivity*time)/(2.0*rho*c*pow(initial_diameter * x_wall, 2.0));
  beta = (diffusivity*time)/(2.0*rho*c*pow(initial_diameter * y_wall, 2.0));

  counter = 1;

  //just in case
  while (isinf(alpha) || isinf(beta))
  {
    time /= 2.0;
    counter *= 2;

    alpha = (diffusivity*time)/(2.0*rho*c*pow(initial_diameter * x_wall, 2.0));
    beta = (diffusivity*time)/(2.0*rho*c*pow(initial_diameter * y_wall, 2.0));
  }
  
  //for a better solution
  time /= 10.0;
  counter *= 10;

  alpha = (diffusivity*time)/(2.0*rho*c*pow(initial_diameter * x_wall, 2.0));
  beta = (diffusivity*time)/(2.0*rho*c*pow(initial_diameter * y_wall, 2.0));

  //increase accuracy
  while (alpha > 1 || beta > 1)
  {
    time /= 2.0;
    counter *= 2;

    alpha = (diffusivity*time)/(2.0*rho*c*pow(initial_diameter * x_wall, 2.0));
    beta = (diffusivity*time)/(2.0*rho*c*pow(initial_diameter * y_wall, 2.0));
  }
}



void
Heat_Lattice::heat_flow()  // 'main'-function, ̹bergibt x-schritt,
// y-schritt, gr̦�e der matrix, deform-zeitschritt, dichte,
// spezifische w�rme, w�rmeleitf�higkeit, pressure
{
  int             i,
  j,
  k,
  l,
  t,
  count,
  intbuf;
  float buf[8];

  //berechne und setze adiabatische temperatur f̹r rand-partikel
  set_boundary_condition(123);

  // if there's no value for a cell, take the average of the surrounding cells
  //   for (i=0;i<col;i++)
  //     {
  //       for (j=0;j<col;j++)
  //         {
  //           if (heat_distribution[i][j] == 0.0)
  //             {
  //               for (k=0; k<8; k++)
  //                 buf[k] = 0.0;
  //
  //               for (k=i-1; k<=i+1; k++)
  //                 {
  //                   count = (k-i)+1;
  //                   for (l=j-1; l<=j+1; l++)
  //                     {
  //                       intbuf = count*3+(l-j)+1;
  //                       if (k>=0 && l>=0)
  //                         {
  //                           if (k<col && l<col)
  //                             buf[intbuf] = heat_distribution[k][l];
  //                           else
  //                             buf[intbuf] = 0.0;
  //                         }
  //                       else
  //                         buf[intbuf] = 0.0;
  //                     }
  //                 }
  //               count = 0;
  //               for (k=0; k<8; k++)
  //                 {
  //                   heat_distribution[i][j] += buf[k];
  //                   if (buf[k] != 0.0)
  //                     count += 1;
  //                 }
  //               heat_distribution[i][j] /= count;
  //             }
  //         }
  //     }

  for (k=0; k<counter; k++)
  {
    cout << "counter: " << counter << endl;
    cout << "k: " << k << endl;
    //solution x-dir
    set_boundary_condition(123);
    transpose();
    set_matrices_1();
    invert();                   //l̦st impliziten teil
    multiplication();          //berechnet den neuen vektor (expliziter teil)
    transpose_interim_distribution();
    multiplication_two();
    //solution y-dir
    set_matrices_2();
    invert();                   //l̦st impliziten teil
    multiplication();          //berechnet den neuen vektor (expliziter teil)
    transpose_interim_distribution();
    multiplication_two();
    transpose();
  }
}


void
Heat_Lattice::invert()
{
  int             i,
  j,
  k,
  l;
  double        factor, swap;

  for (i = 0; i < (col); i++)
  {
    for (j = 0; j < col; j++)
    {
      a_inv[i][j]=(i==j)?1:0;
      a_save[i][j] = a[i][j];
    }
  }
  for (i = 0; i < (col-1); i++)
  {          //Invertierung von a (untere dreiecksmatrix), geschrieben nach u_k
    for (k=i+1; k < (col); k++)
    {
      factor = (-a[k][i]) / a[i][i];
      if (!isinf(factor) && !isnan(factor))
      {
        for (j = 0; j < col; j++)
        {
          a[k][j] += a[i][j] * factor;
          a_inv[k][j] += a_inv[i][j] * factor;
        }
      }
    }
  }
  for (i = col-1; i > 0; i--)
  {          //Invertierung von a (obere dreiecksmatrix), geschrieben nach u_k
    for (k=i-1; k >= 0; k--)
    {
      factor = (-a[k][i]) / a[i][i];
      if (!isinf(factor) && !isnan(factor))
      {
        for (j = 0; j < col; j++)
        {
          a[k][j] += a[i][j] * factor;
          a_inv[k][j] += a_inv[i][j] * factor;
        }
      }
    }
  }
  for (i=0; i<col; i++)
  {                //einsen erzeugen
    if (a[i][i] != 0.0)
    {
      factor = a[i][i];
      for (j=0; j<col; j++)
      {
        a[i][j] /= factor;           //zur �berpr̹fung
        a_inv[i][j] /= factor;
      }
    }
  }
}



void
Heat_Lattice::multiplication()
{
  int             i,
  k,
  j;
  double        swap;
  for (k=0; k<col; k++)
  {
    for (i=0; i<col; i++)
    {
      swap = 0;
      for (j=0; j<col; j++)
      {
        swap += b[k][j] * heat_distribution[j][i];
      }
      interim_heat_distribution[k][i] = swap;
    }
  }
}



void
Heat_Lattice::multiplication_two()
{
  int             i,
  k,
  j;
  double        swap;
  for (k=0; k<col; k++)
  {
    for (i=0; i<col; i++)
    {
      swap = 0;
      for (j=0; j<col; j++)
      {
        swap += a_inv[k][j] * interim_heat_distribution[j][i];
      }
      heat_distribution[k][i] = swap;
    }
  }
}



void
Heat_Lattice::set_matrices_1()
{
  int             i,
  j;

  for (i=0; i<col; i++)
  {                   //baut die matritzen f̹r glg 1
    for (j=0; j<col; j++)
    {
      if (i==j)
      {
        if (i==0 || i==col-1)
        {
          a[i][j] = 1;
          b[i][j] = 1;
        }
        else
        {
          a[i][j] = 1+2*alpha;
          b[i][j] = 1-2*beta;
        }
      }
      else if (i == j+1 || i == j-1)
      {
        a[i][j] = -alpha;
        b[i][j] = beta;
      }
      else
      {
        a[i][j] = 0;
        b[i][j] = 0;
      }
    }
  }
  a[0][1] = b[0][1] = 0;
  a[col-1][col-2] = b[col-1][col-2] = 0;
}



void
Heat_Lattice::set_matrices_2()
{
  int             i,
  j;

  for (i=0; i<col; i++)
  {                   //baut die matritzen f̹r glg 2
    for (j=0; j<col; j++)
    {
      if (i==j)
      {
        if (i==0 || i==col-1)
        {
          a[i][j] = 1;
          b[i][j] = 1;
        }
        else
        {
          a[i][j] = 1+2*beta;
          b[i][j] = 1-2*alpha;
        }
      }
      else if (i == j+1 || i == j-1)
      {
        a[i][j] = -beta;
        b[i][j] = alpha;
      }
      else
      {
        a[i][j] = 0;
        b[i][j] = 0;
      }
    }
  }
  a[0][1] = b[0][1] = 0;
  a[col-1][col-2] = b[col-1][col-2] = 0;
}



void
Heat_Lattice::transpose()
{
  double           interim_matrix[800][800];
  int                i,
  j;

  for (i=0; i<col; i++)
  {
    for (j=0; j<col; j++)
    {
      interim_matrix[i][j] = interim_heat_distribution[j][i];
    }
  }
  for (i=0; i<col; i++)
  {
    for (j=0; j<col; j++)
    {
      interim_heat_distribution[i][j] = interim_matrix[i][j] ;
    }
  }
}



void
Heat_Lattice::transpose_interim_distribution()
{
  double           interim_matrix[800][800];
  int                i,
  j;

  for (i=0; i<col; i++)
  {
    for (j=0; j<col; j++)
    {
      interim_matrix[i][j] = interim_heat_distribution[j][i];
    }
  }
  for (i=0; i<col; i++)
  {
    for (j=0; j<col; j++)
    {
      interim_heat_distribution[i][j] = interim_matrix[i][j] ;
    }
  }
}



void
Heat_Lattice::set_boundary_condition(float pressure)
{
  int             i,
  j;

  adiabatic_temperature = 1000;                                // ACHTUNG: FORMEL EINSETZEN!!!

  for (i=0; i<col; i++)
  {                                     //setzt randfelder entsprechend adiabatischer bedingungen
    heat_distribution[i][col-1] = adiabatic_temperature;
    heat_distribution[i][0] = adiabatic_temperature;
  }
  for (i=0; i<col; i++)
  {                                     //setzt randfelder entsprechend adiabatischer bedingungen
    heat_distribution[col-1][i] = adiabatic_temperature;
    heat_distribution[0][i] = adiabatic_temperature;
  }
}



float
Heat_Lattice::set_particle_temperature(float x, float y)             //ausgabe
{
  int    i,
  j;

  i = int (y/delta_y);
  j = int (x/delta_x);

  //cout << heat_distribution[i][j] << endl;

  return (heat_distribution[i][j]);
}



void
Heat_Lattice::set_cell_temperature(float x, float y, float t)           //eingabe
{
  int             i,
  j;

  i = int (y/delta_y);
  j = int (x/delta_x);

  //    cout << " jetzt!" << endl;
  //always the highest temperature, if two particles share a field
  //if (t > heat_distribution[i][j])
  //  heat_distribution[i][j] = t;

  heat_distribution[i][j] = t;
}


void
Heat_Lattice::SetHeatFlowEnabled(int yes)
{
  if (yes == 1)
    heat_flow_enabled = true;
  else
    heat_flow_enabled = false;
}



void
Heat_Lattice::SetHeatFlowParameters(float rho_n, float c_n, float diffusivity_n, float boundary_condition_n)
{
  rho = rho_n;
  c = c_n;
  diffusivity = diffusivity_n;
  boundary_condition = boundary_condition_n;

  int             i,
  j;
  for (i=0; i<800; i++)
  {
    for (j=0; j<800; j++)
    {
      heat_distribution[i][j] = boundary_condition;                           // set initial temperature
    }
  }
}


