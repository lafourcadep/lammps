/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(snap_ttm,PairSNAPTTM);
// clang-format on
#else

#ifndef LMP_PAIR_SNAPTTM_H
#define LMP_PAIR_SNAPTTM_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSNAPTTM : public Pair {
 public:
  PairSNAPTTM(class LAMMPS *);
  ~PairSNAPTTM() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  double memory_usage() override;
  void *extract(const char *, int &) override;

  double rcutfac, quadraticflag;    // declared public to workaround gcc 4.9
  int ncoeff;                       //  compiler bug, manifest in KOKKOS package

 protected:
  int whichref, indexref, ref2index;
  char *idref;
  
  int ncoeffq, ncoeffall;
  int ntelec;
  class SNA *snaptr;
  virtual void allocate();
  void read_files(char *, char *);
  void read_betas_files(char *);
  inline int equal(double *x, double *y);
  inline double dist2(double *x, double *y);

  void compute_beta();
  double* compute_electronic_temperature_dependent_coeff(double Te_input);
  double compute_electronic_temperature_dependent_betazero(double Te_input);
  void compute_bispectrum();
  void interpolate(int, double, double *, double **);

  double Te_input;  
  double rcutmax;         // max cutoff for all elements
  double *radelem;        // element radii
  double *wjelem;         // elements weights
  double **coeffelem;     // element bispectrum coefficients
  double **betas;        // kernel alphas  
  double *telec;          // list of electronic temperature
  double **beta;          // betas for all atoms in list
  double *betazero;       // beta_zero for all atoms in list
  int bzero_poly_order;  
  double *bzero_poly_coeffs;
  double **bispectrum;    // bispectrum components for all atoms in list
  double **scale;         // for thermodynamic integration
  int twojmax, switchflag, bzeroflag, bnormflag;
  int chemflag, wselfallflag;
  int switchinnerflag;    // inner cutoff switch
  double *sinnerelem;     // element inner cutoff midpoint
  double *dinnerelem;     // element inner cutoff half-width
  int chunksize, parallel_thresh;
  double rfac0, rmin0, wj1, wj2;
  int rcutfacflag, twojmaxflag;    // flags for required parameters
  int beta_max;                    // length of beta
};

}    // namespace LAMMPS_NS

#endif
#endif
