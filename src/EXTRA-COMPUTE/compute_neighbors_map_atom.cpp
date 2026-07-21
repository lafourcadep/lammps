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
   Contributing authors: Paul Lafourcade (CEA-DAM-DIF, Arpajon, France),
   Eliott (fast neighbor search, type filter/weight), Raynol (envelope,
   decay and sortrows options)
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
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;

static const char cite_compute_neighbors_map_atom_c[] =
    "compute neighborsmap/atom command: doi:10.1016/j.commatsci.2023.112535\n\n"
    "@Article{Allera2024,\n"
    " author = {A. Allera and A. M. Goryaeva and P. Lafourcade and J.-B. Maillet and M.-C. Marinica},\n"
    " title = {Neighbors Map: An efficient atomic descriptor for structural analysis},\n"
    " journal = {Computational Materials Science},\n"
    " year = 2024,\n"
    " volume = 231,\n"
    " pages = 112535\n"
    "}\n\n";

namespace {
int next_keyword_index(int pos, int argc, const std::unordered_map<std::string, int> &keypos)
{
  int next = argc;
  for (auto &p : keypos) {
    int other = p.second;
    if (other > pos && other < next) next = other;
  }
  return next;
}
}    // namespace

/* ---------------------------------------------------------------------- */

ComputeNeighborsMapAtom::ComputeNeighborsMapAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), list(nullptr), type_bit(nullptr), type_weight(nullptr), flattened_image(nullptr)
{
  if (lmp->citeme) lmp->citeme->add(cite_compute_neighbors_map_atom_c);

  // compute ID GROUP neighborsmap/atom rcut target_img_size gamma beta ...
  if (narg < 7) utils::missing_cmd_args(FLERR, "compute neighborsmap/atom", error);

  rcut = utils::numeric(FLERR, arg[3], false, lmp);
  target_img_size = utils::inumeric(FLERR, arg[4], false, lmp);
  gamma = utils::numeric(FLERR, arg[5], false, lmp);
  beta = utils::numeric(FLERR, arg[6], false, lmp);

  // defaults
  envelope = 1;    // 1 = DimeNet envelope, 0 = standard (paper) envelope
  decay = 0.0;     // no rank-based decay
  sortrows = 0;    // rows not sorted by decreasing sum

  // locate all optional keywords first, so list-valued keywords (type/weight)
  // know where their value list ends
  std::unordered_map<std::string, int> keypos;
  std::vector<std::string> keywords = {"type", "weight", "envelope", "decay", "sortrows"};
  for (int i = 7; i < narg; i++) {
    for (std::string &kw : keywords) {
      if (strcmp(arg[i], kw.c_str()) == 0) keypos[kw] = i;
    }
  }

  if (keypos.count("envelope")) {
    int pos = keypos["envelope"];
    if (pos + 1 >= narg) utils::missing_cmd_args(FLERR, "compute neighborsmap/atom envelope", error);
    envelope = utils::inumeric(FLERR, arg[pos + 1], false, lmp);
  }
  if (keypos.count("decay")) {
    int pos = keypos["decay"];
    if (pos + 1 >= narg) utils::missing_cmd_args(FLERR, "compute neighborsmap/atom decay", error);
    decay = utils::numeric(FLERR, arg[pos + 1], false, lmp);
  }
  if (keypos.count("sortrows")) {
    int pos = keypos["sortrows"];
    if (pos + 1 >= narg) utils::missing_cmd_args(FLERR, "compute neighborsmap/atom sortrows", error);
    sortrows = utils::inumeric(FLERR, arg[pos + 1], false, lmp);
  }

  if (envelope != 0 && envelope != 1)
    error->all(FLERR, "Illegal compute {} command: envelope must be 0 (standard) or 1 (dimenet)", style);
  if (decay < 0.0) error->all(FLERR, "Illegal compute {} command: decay must be >= 0", style);
  if (sortrows != 0 && sortrows != 1)
    error->all(FLERR, "Illegal compute {} command: sortrows must be 0 (false) or 1 (true)", style);

  // parse the type keyword, build a bitmask filter
  std::vector<int> type_list;
  type_mask = 0u;

  if (keypos.count("type")) {
    int start = keypos["type"] + 1;
    int end = next_keyword_index(start, narg, keypos);

    for (int i = start; i < end; i++) {
      int t = utils::inumeric(FLERR, arg[i], false, lmp);
      if (t < 1 || t > atom->ntypes)
        error->all(FLERR, i, "Invalid type: {} (type > 0 && type <= {})", t, atom->ntypes);

      uint32_t bit = 1u << (t - 1);
      if ((type_mask & bit) == 0) type_list.push_back(t);
      type_mask |= bit;
    }
  } else {
    for (int i = 1; i <= atom->ntypes; i++) type_list.push_back(i);
    type_mask = (atom->ntypes >= 32) ? ~0u : ((1u << atom->ntypes) - 1);
  }

  // parse the weight keyword (one weight per listed type, default 1.0)
  std::vector<double> weights;
  if (keypos.count("weight")) {
    int start = keypos["weight"] + 1;
    int end = next_keyword_index(start, narg, keypos);
    for (int i = start; i < end; i++) weights.push_back(utils::numeric(FLERR, arg[i], false, lmp));
  } else {
    weights.resize(type_list.size(), 1.0);
  }

  if (weights.size() != type_list.size())
    error->all(FLERR, "Number of weights {} do not match number of types {}", weights.size(),
               type_list.size());

  memory->create(type_bit, atom->ntypes + 1, "neighborsmap/atom:type_bit");
  memory->create(type_weight, atom->ntypes + 1, "neighborsmap/atom:type_weight");
  for (int t = 0; t <= atom->ntypes; t++) {
    type_bit[t] = 0;
    type_weight[t] = 0.0;
  }
  for (int t = 1; t <= atom->ntypes; t++)
    if (type_mask & (1u << (t - 1))) type_bit[t] = 1;
  for (int i = 0; i < (int) type_list.size(); i++) type_weight[type_list[i]] = weights[i];

  nmax = 0;
  peratom_flag = 1;
  size_peratom_cols = target_img_size * target_img_size;
}

/* ---------------------------------------------------------------------- */

ComputeNeighborsMapAtom::~ComputeNeighborsMapAtom()
{
  memory->destroy(flattened_image);
  memory->destroy(type_bit);
  memory->destroy(type_weight);
}

/* ---------------------------------------------------------------------- */

void ComputeNeighborsMapAtom::init()
{
  if (force->pair == nullptr)
    error->all(FLERR, "Compute neighborsmap/atom requires a pair style be defined");

  // request our own short neighbor list at rcut, instead of relying on the
  // pair style cutoff being large enough
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL)->set_cutoff(rcut);

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
  invoked_peratom = update->ntimestep;

  // grow per-atom array if necessary
  if (atom->nmax > nmax) {
    memory->destroy(flattened_image);
    nmax = atom->nmax;
    memory->create(flattened_image, nmax, size_peratom_cols, "neighborsmap/atom:flattened_image");
    array_atom = flattened_image;
  }

  // invoke full neighbor list (will copy or build if necessary)
  neighbor->build_one(list);

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  int *mask = atom->mask;
  int *type = atom->type;
  double **x = atom->x;
  const double rcutsq = rcut * rcut;
  const int N = target_img_size;

  // size reusable buffers to the largest neighbor list among atoms in the group
  int nbh_max = 0;
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    if (mask[i] & groupbit) nbh_max = std::max(nbh_max, numneigh[i]);
  }

  std::vector<Node> nbh_buf(nbh_max);
  std::vector<Node> row_buf(nbh_max + 1);

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;

    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];

    // gather type-filtered candidate neighbors within rcut
    int nbh_count = 0;
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      int type_j = type[j];
      double delx = x[j][0] - xtmp;
      double dely = x[j][1] - ytmp;
      double delz = x[j][2] - ztmp;
      double rsq = delx * delx + dely * dely + delz * delz;
      int keep = (rsq < rcutsq) & type_bit[type_j];
      nbh_buf[nbh_count] = {rsq, j, type_j};
      nbh_count += keep;
    }

    // row 0 (central atom) uses up to N nearest neighbors as columns; the
    // anchor ROWS are capped one short of that (N-1), since row 0 itself
    // already occupies row index 0 of the N x N image
    int row0_cols = std::min(nbh_count, N);
    int used_nbh = std::min(nbh_count, N - 1);
    std::partial_sort(nbh_buf.begin(), nbh_buf.begin() + row0_cols, nbh_buf.begin() + nbh_count, NodeLess{});

    std::vector<std::vector<double>> M_j(N, std::vector<double>(N, 0.0));
    std::vector<std::vector<int>> col_type(N, std::vector<int>(N, 0));
    std::vector<int> row_type(N, 0);
    std::vector<int> used_cols(N, 0);

    // row 0: central atom -> its row0_cols nearest type-filtered neighbors
    row_type[0] = type[i];
    used_cols[0] = row0_cols;
    for (int k = 0; k < row0_cols; k++) {
      M_j[0][k] = std::sqrt(nbh_buf[k].r2);
      col_type[0][k] = nbh_buf[k].type;
    }

    // rows 1..used_nbh: anchor neighbor r -> its nearest neighbors among
    // {central atom + full type-filtered candidate pool}, excluding itself
    for (int r = 0; r < used_nbh; r++) {
      int j = nbh_buf[r].idx;
      const double xj = x[j][0];
      const double yj = x[j][1];
      const double zj = x[j][2];

      int local_size = 0;
      double dxc = xtmp - xj, dyc = ytmp - yj, dzc = ztmp - zj;
      row_buf[local_size++] = {dxc * dxc + dyc * dyc + dzc * dzc, i, type[i]};
      for (int c = 0; c < nbh_count; c++) {
        if (c == r) continue;    // exclude the anchor itself
        int k = nbh_buf[c].idx;
        double dx = x[k][0] - xj, dy = x[k][1] - yj, dz = x[k][2] - zj;
        row_buf[local_size++] = {dx * dx + dy * dy + dz * dz, k, nbh_buf[c].type};
      }

      int ncols = std::min(local_size, N);
      std::partial_sort(row_buf.begin(), row_buf.begin() + ncols, row_buf.begin() + local_size, NodeLess{});

      row_type[r + 1] = nbh_buf[r].type;
      used_cols[r + 1] = ncols;
      for (int k = 0; k < ncols; k++) {
        M_j[r + 1][k] = std::sqrt(row_buf[k].r2);
        col_type[r + 1][k] = row_buf[k].type;
      }
    }

    const int n_used_rows = used_nbh + 1;

    // apply the geometric envelope to the raw distance matrix
    if (envelope == 0) apply_standard_envelope(M_j, beta, gamma);
    else apply_dimenet_envelope(M_j, rcut, beta, gamma);

    // re-establish exact zero padding for rows/cols this atom didn't actually
    // have neighbors for (the envelope functions turn a raw 0 into a
    // nonzero value, e.g. dimenet_envelope(0) == 1)
    for (int l = n_used_rows; l < N; l++) std::fill(M_j[l].begin(), M_j[l].end(), 0.0);
    for (int l = 0; l < n_used_rows; l++)
      for (int k = used_cols[l]; k < N; k++) M_j[l][k] = 0.0;

    // apply per-type weights
    apply_type_weights(M_j, row_type, col_type);

    // optionally sort rows by decreasing sum
    if (sortrows == 1) sort_rows_by_decreasing_sum(M_j, N);

    // optionally apply the rank-based decay envelope
    if (decay > 0.0) apply_decay_envelope(M_j, n_used_rows, decay);

    // flatten the image to output; indexed by local atom index i (not the
    // ilist position ii) so results stay aligned with the rest of LAMMPS
    for (int l = 0; l < N; l++)
      for (int k = 0; k < N; k++) flattened_image[i][l * N + k] = M_j[l][k];
  }
}

/* ---------------------------------------------------------------------- */

// Sort in-place the squared distance and neighbor index vectors

/* ---------------------------------------------------------------------- */

void ComputeNeighborsMapAtom::inplace_selection_sort(std::vector<double> &dist, std::vector<int> &idx, int nvalid)
{
  for (int a = 0; a < nvalid - 1; a++) {
    int imin = a;
    double dmin = dist[a];

    for (int b = a + 1; b < nvalid; b++) {
      if (dist[b] < dmin) {
        dmin = dist[b];
        imin = b;
      }
    }
    if (imin != a) {
      double td = dist[a];
      dist[a] = dist[imin];
      dist[imin] = td;
      int ti = idx[a];
      idx[a] = idx[imin];
      idx[imin] = ti;
    }
  }
}

/* ---------------------------------------------------------------------- */

// Standard envelope function from https://doi.org/10.1016/j.commatsci.2023.112535 Eqs. (2) and (3)

/* ---------------------------------------------------------------------- */

void ComputeNeighborsMapAtom::apply_standard_envelope(std::vector<std::vector<double>> &matrix, double beta,
                                                        double gamma)
{
  const int nrows = matrix.size();
  const int ncols = matrix[0].size();
  std::vector<double> scale_factors(nrows, 1.0);

  // row l (l >= 1) is anchored on the (l-1)-th nearest neighbor of the
  // central atom; its scale factor comes from that same neighbor's own
  // central distance, stored at row 0, column (l-1). A raw distance of 0
  // means padding (not enough neighbors to fill the image) -> scale/value
  // must stay 0 instead of blowing up through 1/0.
  // r^(-1/beta), matching the nm_repo reference implementation (verified
  // numerically against neighbors_map.py) -- not r^(-beta)
  for (int l = 1; l < nrows; l++) {
    double d = matrix[0][l - 1];
    scale_factors[l] = (d > 0.0) ? std::pow(d, -1.0 / beta) : 0.0;
  }

  for (int l = 0; l < nrows; l++)
    for (int k = 0; k < ncols; k++) {
      double d = matrix[l][k];
      matrix[l][k] = (d > 0.0) ? scale_factors[l] / std::pow(d, gamma) : 0.0;
    }
}

/* ---------------------------------------------------------------------- */

// DimeNet envelope function, from https://doi.org/10.48550/arXiv.2003.03123 Eq. (8)

/* ---------------------------------------------------------------------- */

void ComputeNeighborsMapAtom::dimenet_envelope(std::vector<double> &x, double p)
{
  const double c1 = (p + 1.0) * (p + 2.0) * 0.5;
  const double c2 = p * (p + 2.0);
  const double c3 = p * (p + 1.0) * 0.5;
  const int n = x.size();

  for (int i = 0; i < n; i++) {
    double xi = x[i];
    if (xi < 0.0) xi = 0.0;
    else if (xi > 1.0) xi = 1.0;

    double x_p = std::pow(xi, p);
    double x_p1 = xi * x_p;
    double x_p2 = xi * x_p1;
    x[i] = 1.0 - c1 * x_p + c2 * x_p1 - c3 * x_p2;
  }
}

/* ---------------------------------------------------------------------- */

// Apply the DimeNet envelope function to a matrix

/* ---------------------------------------------------------------------- */

void ComputeNeighborsMapAtom::apply_dimenet_envelope(std::vector<std::vector<double>> &matrix, double rcut,
                                                       double beta, double gamma)
{
  const int nrows = matrix.size();
  const int ncols = matrix[0].size();
  std::vector<double> scale_factors(nrows, 0.0);

  // see apply_standard_envelope: row l (l >= 1) takes its scale factor from
  // its own anchor neighbor's central distance, stored at row 0, column (l-1)
  for (int l = 1; l < nrows; l++) scale_factors[l] = matrix[0][l - 1] / rcut;
  dimenet_envelope(scale_factors, beta);
  scale_factors[0] = 1.0;

  for (int l = 0; l < nrows; l++) {
    for (int k = 0; k < ncols; k++) matrix[l][k] /= rcut;
    dimenet_envelope(matrix[l], gamma);
    for (int k = 0; k < ncols; k++) matrix[l][k] *= scale_factors[l];
  }
}

/* ---------------------------------------------------------------------- */

// Apply a (DimeNet) rank-based decay envelope to a matrix

/* ---------------------------------------------------------------------- */

void ComputeNeighborsMapAtom::apply_decay_envelope(std::vector<std::vector<double>> &matrix, int nvalid,
                                                     double decay)
{
  if (nvalid < 2) return;

  std::vector<double> sq_decay_factors(nvalid);
  for (int l = 0; l < nvalid; l++) sq_decay_factors[l] = static_cast<double>(l) / (nvalid - 1);
  dimenet_envelope(sq_decay_factors, decay);

  for (int l = 0; l < nvalid; l++)
    for (int k = 0; k < nvalid; k++) matrix[l][k] *= std::sqrt(sq_decay_factors[l]) * std::sqrt(sq_decay_factors[k]);
}

/* ---------------------------------------------------------------------- */

// Sort the rows of a matrix by their decreasing sum

/* ---------------------------------------------------------------------- */

void ComputeNeighborsMapAtom::sort_rows_by_decreasing_sum(std::vector<std::vector<double>> &matrix, int nvalid)
{
  std::vector<double> row_summed(nvalid, 0.0);
  std::vector<int> row_index(nvalid);
  for (int l = 0; l < nvalid; l++) {
    row_index[l] = l;
    for (int k = 0; k < nvalid; k++) row_summed[l] += matrix[l][k];
  }

  inplace_selection_sort(row_summed, row_index, nvalid);

  std::vector<std::vector<double>> matrix_sorted_dec(nvalid);
  for (int l = 0; l < nvalid; l++) matrix_sorted_dec[l] = matrix[row_index[nvalid - 1 - l]];

  matrix = std::move(matrix_sorted_dec);
}

/* ---------------------------------------------------------------------- */

// Multiply each matrix entry by the per-type weight of its row atom and column atom

/* ---------------------------------------------------------------------- */

void ComputeNeighborsMapAtom::apply_type_weights(std::vector<std::vector<double>> &matrix,
                                                   const std::vector<int> &row_type,
                                                   const std::vector<std::vector<int>> &col_type)
{
  const int nrows = matrix.size();
  const int ncols = matrix[0].size();
  for (int l = 0; l < nrows; l++) {
    double wl = type_weight[row_type[l]];
    for (int k = 0; k < ncols; k++) matrix[l][k] *= wl * type_weight[col_type[l][k]];
  }
}

/* ---------------------------------------------------------------------- */

double ComputeNeighborsMapAtom::memory_usage()
{
  double bytes = (double) nmax * (double) size_peratom_cols * sizeof(double);
  return bytes;
}
