/* ----------------------------------------------------------------------
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

#include "compute_neighbors_map_atom.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <iostream>

using namespace LAMMPS_NS;

static const char cite_compute_neighbors_map_atom_c[] =
    "compute neighborsmap/atom command: doi:XX.XXX/XXXXX\n\n"
    "@Article{XXX,\n"
    " author = {XX, XX, XX, XX},\n"
    " title = {XX},\n"
    " journal = {XX},\n"
    " year = 202X,\n"
    " volume = XXX,\n"
    " pages = XXX\n"
    "}\n\n";

/* ---------------------------------------------------------------------- */
ComputeNeighborsMapAtom::ComputeNeighborsMapAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), list(nullptr), distsq(nullptr), nearest(nullptr), nm_flattened_image(nullptr)
{

  if (lmp->citeme) lmp->citeme->add(cite_compute_neighbors_map_atom_c);

  if (narg !=7) utils::missing_cmd_args(FLERR, "compute neighborsmap/atom", error);

  nm_rcut = utils::numeric(FLERR, arg[3], false, lmp);
  nm_target_imsize = utils::inumeric(FLERR, arg[4], false, lmp);
  nm_gamma_decay = utils::numeric(FLERR, arg[5], false, lmp);
  nm_beta_att = utils::numeric(FLERR, arg[6], false, lmp);
  
  std::cout << "NM cutoff = " << nm_rcut << std::endl;
  std::cout << "NM target image size = " << nm_target_imsize << std::endl;
  //  memory->create(arraytest, ncomps, "neighborsmap/atom:test");
  //  memory->create(arraytestbis, ncomps, ncomps2, "neighborsmap/atom:testbis");

  size_peratom_cols = nm_target_imsize * nm_target_imsize;
  peratom_flag = 1;
  
  nmax = 0;
  maxneigh = 0;
}

/* ---------------------------------------------------------------------- */

ComputeNeighborsMapAtom::~ComputeNeighborsMapAtom()
{
  memory->destroy(nm_flattened_image);
}

/* ---------------------------------------------------------------------- */

void ComputeNeighborsMapAtom::init()
{

  if (force->pair == nullptr)
    error->all(FLERR, "Compute neighborsmap/atom requires a pair style be defined");

  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
  
  if (modify->get_compute_by_style(style).size() > 1)
    if (comm->me == 0) error->warning(FLERR, "More than one compute {}", style);
}

/* ---------------------------------------------------------------------- */

void ComputeNeighborsMapAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeNeighborsMapAtom::compute_peratom()
{
  int i, j, k, l, ii, jj, kk, ll, n, inum, jnum;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq, rj_lk;
  int *ilist, *jlist, *numneigh, **firstneigh;
  
  invoked_peratom = update->ntimestep;

  // grow per-atom if necessary

  if (atom->nmax > nmax) {
    memory->destroy(nm_flattened_image);
    nmax = atom->nmax;
    memory->create(nm_flattened_image, nmax, size_peratom_cols, "neighborsmap/atom:nm_flattened_image");
    array_atom = nm_flattened_image;
  }

  // invoke full neighbor list (will copy or build if necessary)
  
  neighbor->build_one(list);
  
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // compute neighbors map for each atom in group
  // use full neighbor list
  
  double **x = atom->x;
  int *mask = atom->mask;
  //  double cutsq = force->pair->cutforce * force->pair->cutforce;
  if (nm_rcut > force->pair->cutforce)
    error->all(FLERR, "Compute neighborsmap/atom requires a cutoff larger than pair style. Please consider adding a zero pair style with a cutoff equal or greater than the one required by this compute.");
  
  double cutsq = nm_rcut * nm_rcut;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (mask[i] & groupbit) {
      // position of central atom
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      
      jlist = firstneigh[i];
      jnum = numneigh[i];

      // ensure distsq and nearest arrays are long enough

      if (jnum > maxneigh) {
        memory->destroy(distsq);
        memory->destroy(nearest);
        maxneigh = jnum;
        memory->create(distsq, maxneigh, "neighborsmap/atom:distsq");
        memory->create(nearest, maxneigh, "neighborsmap/atom:nearest");
      }

      // loop over list of all neighbors within force cutoff
      // distsq[] = distance sq to each
      // nearest[] = atom indices of neighbors
      
      n = 0;
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutsq) {
          distsq[n] = rsq;
          nearest[n++] = j;
        }
      }

      ///// DEBUT
      // ------- Hyperparameters -------
      const int nG = nm_target_imsize;
      const double eps = 1e-12;           // guard for zero distances
      
      // ------- 1) Sort neighbors by central distance -> defines the row-0 order -------
      std::vector<int>   base_idx(nG);        // neighbor indices (into your global x[][]), sorted by central distance
      std::vector<double> rj0k_sorted(nG);    // corresponding central->k distances (ascending)

      {
        std::vector<double> dists(n);
        std::vector<int>    idx(n);
        for (int k = 0; k < n; ++k) {
          const int kk = nearest[k];
          const double dx = x[kk][0] - xtmp;
          const double dy = x[kk][1] - ytmp;
          const double dz = x[kk][2] - ztmp;
          double d = std::sqrt(dx*dx + dy*dy + dz*dz);
          if (d < eps) d = eps;  // guard (e.g., duplicates)
          dists[k] = d;
          idx[k]   = k;          // store position in nearest[]
        }
        // sort by central distance (ascending)
        selection_sort_dist_idx(dists, idx);

        // map to global atom indices in sorted order
        for (int k = 0; k < nG; ++k) {
          base_idx[k]      = nearest[idx[k]]; // global atom id of k-th nearest neighbor
          rj0k_sorted[k]   = dists[k];
        }
      }

      // ------- 2) Prepare weights w_j• and row weights g_j• -------
      std::vector<double> wj(nG + 1, 1.0);      // node weights: [0]=central, [1..nG]=neighbors; adjust if species-weighted
      std::vector<double> gj(nG, 1.0);          // row weights: gj[0] for row 0, gj[l] for row l (0-based)
      gj[0] = 1.0;                               // row 0 weight
      for (int l = 1; l < nG; ++l) {
        // use attention-like weight based on central->l distance (after sorting)
        gj[l] = 1.0 / std::pow(rj0k_sorted[l-1], nm_beta_att); // l-1 because row 1 anchors the 1st neighbor, etc.
      }

      // ------- 3) Allocate M_j -------
      std::vector<std::vector<double>> Mj(nG, std::vector<double>(nG, 0.0));

      // ------- 4) Row 0 (Eq. (2)): central -> k, using the sorted order -------
      for (int k = 0; k < nG; ++k) {
        const double denom = std::pow(rj0k_sorted[k], nm_gamma_decay);
        // wj[0] is for central; neighbor weight is wj[k+1] because neighbors occupy 1..nG
        Mj[0][k] = gj[0] * wj[0] * wj[k+1] / denom;
      }

      // ------- 5) Rows l+1 (Eq. (3)): distances from anchor neighbor l to other neighbors -------
      // For each l in 0..nG-2 (anchors the (l+1)-th row), we:
      //   - compute distances from base_idx[l] to every other neighbor base_idx[k!=l]
      //   - sort those neighbors by this distance
      //   - fill columns 0..nG-2 with these sorted entries, and set column nG-1 to 0 (padding, no self)
      for (int l = 0; l < nG - 1; ++l) {
        const int row = l + 1;               // row index in Mj (0-based)
        const int ll  = base_idx[l];         // global atom id of anchor neighbor
        //        const int ll = base_idx[
        const double xl = x[ll][0], yl = x[ll][1], zl = x[ll][2];

        // build arrays excluding self (size nG-1)
        std::vector<double> dists; dists.reserve(nG - 1);
        std::vector<int>    kpos;  kpos.reserve(nG - 1);   // positions q in base_idx for the other neighbors
        for (int q = 0; q < nG; ++q) {
          if (q == l) continue;                // exclude self
          const int kk = base_idx[q];
          const double dx = xl - x[kk][0];
          const double dy = yl - x[kk][1];
          const double dz = zl - x[kk][2];
          double d = std::sqrt(dx*dx + dy*dy + dz*dz);
          if (d < eps) d = eps;                // guard
          dists.push_back(d);
          kpos.push_back(q);                   // remember column partner’s position in the base ordering
        }

        // sort by distance from neighbor l
        selection_sort_dist_idx(dists, kpos);

        // fill row: first nG-1 columns from sorted pairs; last column is padding (0.0)
        for (int p = 0; p < nG - 1; ++p) {
          const double denom = std::pow(dists[p], nm_gamma_decay);
          const int q = kpos[p];                        // this is the partner's position in base_idx
          // neighbor weights: anchor is (l+1), partner is (q+1)
          Mj[row][p] = gj[row] * wj[l+1] * wj[q+1] / denom;
        }
        Mj[row][nG - 1] = 0.0;  // padding column (no self-interaction)
      }

      // At this point, Mj is the nG x nG matrix for central atom i, per Sec. 2.2.3 (Eqs. (2)–(3)).
      // --- Flatten Mj into a 1D vector (row-major order)
      for (int l = 0; l < nG; ++l) {
        for (int k = 0; k < nG; ++k) {
          nm_flattened_image[ii][l * nG + k] = Mj[l][k];
        }
      }
 
// ------- (Optional) Flatten row-major -------
// std::vector<double> Mj_flat(nG * nG);
// for (int r = 0; r < nG; ++r) for (int c = 0; c < nG; ++c) Mj_flat[r*nG + c] = Mj[r][c];

      ///// FINNNNN
      
      // // Reorder distsq and nearest so nG firsts are nG nearest atoms to central atom
      // select2(nG, n, distsq, nearest);
        
      // // convenience: cache neighbor indices for the nG closest neighbors of j
      // // assumes `nearest[k]` gives the atom index of the k-th closest neighbor of j (k = 0..nG-1)
      // std::vector<int> neigh_idx(nG);
      // for (int k = 0; k < nG; ++k) neigh_idx[k] = nearest[k];

      // // precompute r_{j:0k} for k = 1..nG  (note: in code, neighbors are 0-based; in the paper k starts at 1)
      // std::vector<double> rj0k(nG, 0.0);

      // for (int k = 0; k < nG; ++k) {
      //   const int kk = neigh_idx[k];
      //   const double dx = x[kk][0] - xtmp;
      //   const double dy = x[kk][1] - ytmp;
      //   const double dz = x[kk][2] - ztmp;
      //   double d = std::sqrt(dx*dx + dy*dy + dz*dz); 
      //   rj0k[k] = (d < eps) ? eps : d; // guard
      // }

      // // --- Build per-l sorted neighbor lists (exclude self) and distances r_{j:lk}
      // std::vector<std::vector<int>>    sorted_k_idx_by_l(nG);  // indices of neighbors for each l, sorted by distance from l
      // std::vector<std::vector<double>> sorted_r_by_l(nG);      // corresponding distances

      // for (int l = 0; l < nG; ++l) {
      //   const int ll = neigh_idx[l];
      //   const double xl = x[ll][0];
      //   const double yl = x[ll][1];
      //   const double zl = x[ll][2];

      //   // build arrays excluding self
      //   std::vector<double> dists; dists.reserve(nG - 1);
      //   std::vector<int>    kidx;  kidx.reserve(nG - 1);

      //   for (int k = 0; k < nG; ++k) {
      //     if (k == l) continue; // exclude self -> no zero distance
      //     const int kk = neigh_idx[k];
      //     const double dx = xl - x[kk][0];
      //     const double dy = yl - x[kk][1];
      //     const double dz = zl - x[kk][2];
      //     double d = std::sqrt(dx*dx + dy*dy + dz*dz);
      //     if (d < eps) d = eps; // guard in case of duplicate coords
      //     dists.push_back(d);
      //     kidx.push_back(k);     // store original k within nearest[]
      //   }

      //   // sort ascending by distance using custom selection sort
      //   selection_sort_dist_idx(dists, kidx);

      //   // store results
      //   sorted_r_by_l[l]     = std::move(dists);
      //   sorted_k_idx_by_l[l] = std::move(kidx);
      // }
      
      // // line weights g_{j(l+1)}: g_{j1}=1 for first row; then use attention-like weight based on central->l distance
      // std::vector<double> gj(nG, 1.0);     // gj[row_index], with row_index = l' = 1..nG; store 0-based here
      // gj[0] = 1.0;                         // corresponds to line for l = 0 (first row in the paper)
      // for (int l = 1; l < nG; ++l) {
      //   // paper suggests g_{j(l+1)} = 1 / r_{j:0l}^{nm_beta_att}; adjust if you want something else
      //   gj[l] = 1.0 / std::pow(rj0k[l], nm_beta_att);
      // }
      

      // // allocate M_j
      // std::vector<std::vector<double>> Mj(nG, std::vector<double>(nG, 0.0));

      // // Row 0 (central -> k), as before
      // for (int k = 0; k < nG; ++k) {
      //   const double denom = std::pow(rj0k[k], nm_gamma_decay);
      //   Mj[0][k] = gj[0] * wj[0] * wj[k+1] / denom;
      // }

      // // Rows l+1: sorted by distance from neighbor l, exclude self, pad last column
      // for (int l = 0; l < nG; ++l) {
      //   const int row = l + 1;
      //   if (row >= nG) break; // total nG rows

      //   for (int p = 0; p < nG; ++p) {
      //     if (p < nG - 1) {
      //       const int k_idx_in_neigh = sorted_k_idx_by_l[l][p];
      //       const double d = sorted_r_by_l[l][p];
      //       const double denom = std::pow(d, nm_gamma_decay);
      //       Mj[row][p] = 0.0;//gj[row] * wj[l+1] * wj[k_idx_in_neigh+1] / denom;
      //     } else {
      //       Mj[row][p] = 0.0; // padding column (no self)
      //     }
      //   }
      // }

      // // At this point, Mj is the nG x nG matrix for central atom i, per Sec. 2.2.3 (Eqs. (2)–(3)).
      // // --- Flatten Mj into a 1D vector (row-major order)
      // for (int l = 0; l < nG; ++l) {
      //   for (int k = 0; k < nG; ++k) {
      //     nm_flattened_image[ii][l * nG + k] = Mj[l][k];
      //   }
      // }
      
    }
  }
  
}

void ComputeNeighborsMapAtom::selection_sort_dist_idx(std::vector<double>& dist, std::vector<int>& idx) {
  const int n = (int)dist.size();
  for (int a = 0; a < n - 1; ++a) {
    int imin = a;
    double dmin = dist[a];
    for (int b = a + 1; b < n; ++b) {
      if (dist[b] < dmin) { dmin = dist[b]; imin = b; }
    }
    if (imin != a) {
      // manual swap (avoid std::swap)
      double td = dist[a]; dist[a] = dist[imin]; dist[imin] = td;
      int ti = idx[a];     idx[a]  = idx[imin];  idx[imin]  = ti;
    }
  }
}
