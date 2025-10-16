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

  double rcutfac;
  int quadraticflag, ncoeff;

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
  void compute_bispectrum();
  
  double compute_electronic_temperature_dependent_betazero(double Te_input);
  void evaluate_electronic_temperature_dependent_betazero(const std::string& csv_path);
  void check_read_betas(const std::string& csv_path);

  // Cache holding all precomputed spline data.
  // Layout: for each row (0..M-1) and each interval k (0..N-2), we store 4 coeffs (a,b,c,d).
  // Access pattern: coeffs[(row*(N-1) + k)*4 + {0..3}]
  struct BetaSplines {
    int N = 0;                 // number of knots
    int M = 0;                 // number of rows
    std::vector<double> x;     // telec[0..N-1]
    std::vector<double> h;     // h[k] = x[k+1]-x[k], size N-1

    // Thomas factorization terms of the tridiagonal that depends only on x/h (natural BC):
    // denom[i] = modified diagonal, cprime[i] = modified superdiag for fast solves per row.
    std::vector<double> cprime; // size N
    std::vector<double> denom;  // size N

    // Precomputed per-interval cubic coefficients for every row: a,b,c,d
    std::vector<double> coeffs; // size M*(N-1)*4
  } BetaSpl;

  // Build the cache from telec (size N, strictly increasing, non-uniform OK) and betas (M x N).
  // betas[i] must point to an array of length N: betas[i][j] = beta_i(telec[j]).
  // Throws std::invalid_argument on input errors.
  void beta_splines_build();

  // Evaluate at a single x. Writes M values to out[0..M-1].
  void beta_splines_eval(double x, double* out);

  // Convenience wrapper that returns a std::vector<double>(M) for a single x.
  std::vector<double> beta_splines_eval_vec(double x);

  // Evaluate many x’s (xs[0..K-1]). Writes into out with row-major [q*M + i] = beta_i(xs[q]).
  //  void beta_splines_eval_many(const BetaSplines& S, const double* xs, int K, double* out);

  // Utility: find interval index k with clamped extrapolation (0..N-2).
  int beta_splines_find_interval(double x);

  void factor_tridiagonal_natural(const std::vector<double>&, std::vector<double>&, std::vector<double>&);
  void solve_tridiagonal_natural(const std::vector<double>&, const std::vector<double>& cprime, const std::vector<double>&, std::vector<double>&);

  void build_row_coeffs(const double*, const std::vector<double>&,const std::vector<double>&,const std::vector<double>&,int, int, BetaSplines&);
  void evaluate_electronic_temperature_dependent_betas(const std::string& csv_path);  
  double* compute_electronic_temperature_dependent_betas(double Te_input);
  
  double Te_input;  
  double rcutmax;         // max cutoff for all elements
  double *radelem;        // element radii
  double *wjelem;         // elements weights
  double **coeffelem;     // element bispectrum coefficients
  double **betas;         // kernel alphas  
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
