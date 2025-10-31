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

/* ----------------------------------------------------------------------
   Contributing author: Paul Lafourcade (CEA-DAM-DIF, Arpajon, France)
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(neighborsmap/atom,ComputeNeighborsMapAtom);
// clang-format on
#else

#ifndef LMP_COMPUTE_NEIGHBORSMAP_ATOM_H
#define LMP_COMPUTE_NEIGHBORSMAP_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeNeighborsMapAtom : public Compute {
 public:
  ComputeNeighborsMapAtom(class LAMMPS *, int, char **);
  ~ComputeNeighborsMapAtom() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  //  double memory_usage() override;

 private:
  int nmax, maxneigh, nnn, ncol;
  double *distsq;
  int *nearest;
  class NeighList *list;
  int nm_target_imsize;
  double nm_rcut;
  double nm_gamma_decay;
  double nm_beta_att;
  double **nm_flattened_image;
  void select2(int, int, double *, int *);
  void selection_sort_dist_idx(std::vector<double>&, std::vector<int>&);
};

}    // namespace LAMMPS_NS

#endif
#endif
