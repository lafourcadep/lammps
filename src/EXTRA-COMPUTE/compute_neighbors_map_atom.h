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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(neighborsmap/atom,ComputeNeighborsMapAtom);
// clang-format on
#else

#ifndef LMP_COMPUTE_NEIGHBORSMAP_ATOM_H
#define LMP_COMPUTE_NEIGHBORSMAP_ATOM_H

#include "compute.h"

#include <cstdint>
#include <vector>

namespace LAMMPS_NS {

class ComputeNeighborsMapAtom : public Compute {
 public:
  ComputeNeighborsMapAtom(class LAMMPS *, int, char **);
  ~ComputeNeighborsMapAtom() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  double memory_usage() override;

 private:
  int nmax;
  class NeighList *list;

  // core hyperparameters (positional args): rcut, target_img_size, gamma, beta
  int target_img_size;
  double rcut;
  double gamma;
  double beta;

  // envelope keywords (from raynol)
  int envelope;    // 0 = standard (paper) envelope, 1 = DimeNet envelope
  double decay;    // rank-based decay strength, 0 = disabled
  int sortrows;    // 1 = sort rows by decreasing sum

  // type filter / weight keywords (from eliott)
  uint32_t type_mask;
  int *type_bit;
  double *type_weight;

  double **flattened_image;

  struct Node {
    double r2 = 0.0;
    int idx = 0;
    int type = 0;
  };

  struct NodeLess {
    inline bool operator()(const Node &a, const Node &b) const noexcept { return a.r2 < b.r2; }
  };

  void inplace_selection_sort(std::vector<double> &, std::vector<int> &, int);
  void apply_standard_envelope(std::vector<std::vector<double>> &, double, double);
  void dimenet_envelope(std::vector<double> &x, double);
  void apply_dimenet_envelope(std::vector<std::vector<double>> &, double, double, double);
  void apply_decay_envelope(std::vector<std::vector<double>> &, int, double);
  void sort_rows_by_decreasing_sum(std::vector<std::vector<double>> &, int);
  void apply_type_weights(std::vector<std::vector<double>> &, const std::vector<int> &row_type,
                           const std::vector<std::vector<int>> &col_type);
};

}    // namespace LAMMPS_NS

#endif
#endif
