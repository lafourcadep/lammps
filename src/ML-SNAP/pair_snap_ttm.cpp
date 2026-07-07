// clang-format off
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

#include "pair_snap_ttm.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "info.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "sna.h"
#include "tokenizer.h"
#include "arg_info.h"
#include "fix.h"
#include "fix_ttm.h"
#include "fix_ttm_grid.h"
#include "fix_ttm_mod.h"
#include "modify.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include <vector>
#include <algorithm>

using namespace LAMMPS_NS;

static constexpr int MAXLINE = 1024;

/* ---------------------------------------------------------------------- */

PairSNAPTTM::PairSNAPTTM(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;

  radelem = nullptr;
  wjelem = nullptr;
  coeffelem = nullptr;
  sinnerelem = nullptr;
  dinnerelem = nullptr;

  beta_max = 0;
  beta = nullptr;
  bispectrum = nullptr;
  snaptr = nullptr;
}

/* ---------------------------------------------------------------------- */

PairSNAPTTM::~PairSNAPTTM()
{
  if (copymode) return;

  memory->destroy(radelem);
  memory->destroy(wjelem);
  memory->destroy(coeffelem);
  memory->destroy(sinnerelem);
  memory->destroy(dinnerelem);

  memory->destroy(beta);
  memory->destroy(bispectrum);

  delete snaptr;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }

}

/* ----------------------------------------------------------------------
   This version is a straightforward implementation
   ---------------------------------------------------------------------- */

void PairSNAPTTM::compute(int eflag, int vflag)
{
  int i,j,jnum,ninside;
  double delx,dely,delz,evdwl,rsq;
  double fij[3];
  int *jlist,*numneigh,**firstneigh;

  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  if (beta_max < list->inum) {
    memory->grow(beta,list->inum,ncoeff,"PairSNAPTTM:beta");
    memory->grow(bispectrum,list->inum,ncoeff,"PairSNAPTTM:bispectrum");
    beta_max = list->inum;
  }

  // compute dE_i/dB_i = beta_i for all i in list

  if (quadraticflag || eflag)
    compute_bispectrum();
  compute_beta();

  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  Fix *fix = modify->get_fix_by_id(idref);
  double *fix_vector = fix->vector_atom;
  double conv_K_to_eV = 8.61732814974056e-5;
  
  for (int ii = 0; ii < list->inum; ii++) {
    i = list->ilist[ii];

    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];
    const int itype = type[i];
    const int ielem = map[itype];
    const double radi = radelem[ielem];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    // ensure rij, inside, wj, and rcutij are of size jnum

    snaptr->grow_rij(jnum);

    // rij[][3] = displacements between atom I and those neighbors
    // inside = indices of neighbors of I within cutoff
    // wj = weights for neighbors of I within cutoff
    // rcutij = cutoffs for neighbors of I within cutoff
    // note Rij sign convention => dU/dRij = dU/dRj = -dU/dRi

    ninside = 0;
    for (int jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx*delx + dely*dely + delz*delz;
      int jtype = type[j];
      int jelem = map[jtype];

      if (rsq < cutsq[itype][jtype]&&rsq>1e-20) {
        snaptr->rij[ninside][0] = delx;
        snaptr->rij[ninside][1] = dely;
        snaptr->rij[ninside][2] = delz;
        snaptr->inside[ninside] = j;
        snaptr->wj[ninside] = wjelem[jelem];
        snaptr->rcutij[ninside] = (radi + radelem[jelem])*rcutfac;
        if (switchinnerflag) {
          snaptr->sinnerij[ninside] = 0.5*(sinnerelem[ielem]+sinnerelem[jelem]);
          snaptr->dinnerij[ninside] = 0.5*(dinnerelem[ielem]+dinnerelem[jelem]);
        }
        if (chemflag) snaptr->element[ninside] = jelem;
        ninside++;
      }
    }

    // compute Ui, Yi for atom I

    if (chemflag)
      snaptr->compute_ui(ninside, ielem);
    else
      snaptr->compute_ui(ninside, 0);

    // for neighbors of I within cutoff:
    // compute Fij = dEi/dRj = -dEi/dRi
    // add to Fi, subtract from Fj
    // scaling is that for type I

    snaptr->compute_yi(beta[ii]);

    for (int jj = 0; jj < ninside; jj++) {
      int j = snaptr->inside[jj];
      snaptr->compute_duidrj(jj);

      snaptr->compute_deidrj(fij);

      f[i][0] += fij[0]*scale[itype][itype];
      f[i][1] += fij[1]*scale[itype][itype];
      f[i][2] += fij[2]*scale[itype][itype];
      f[j][0] -= fij[0]*scale[itype][itype];
      f[j][1] -= fij[1]*scale[itype][itype];
      f[j][2] -= fij[2]*scale[itype][itype];

      // tally per-atom virial contribution

      if (vflag)
        ev_tally_xyz(i,j,nlocal,newton_pair,0.0,0.0,
                     fij[0],fij[1],fij[2],
                     -snaptr->rij[jj][0],-snaptr->rij[jj][1],
                     -snaptr->rij[jj][2]);
    }

    // tally energy contribution

    if (eflag) {
      double Te_loc = 0.;
      
      // evdwl = energy of atom I, sum over coeffs_k * Bi_k
      double* coeffi = beta[ii];
      //      double* coeffi = coeffelem[ielem];
      int i = list->ilist[ii];
      Te_loc = fix_vector[i] * conv_K_to_eV;
      evdwl = compute_electronic_temperature_dependent_betazero(Te_loc);
      
      // snaptr->copy_bi2bvec();

      // E = beta.B + 0.5*B^t.alpha.B

      // linear contributions

      for (int icoeff = 0; icoeff < ncoeff; icoeff++)
        evdwl += coeffi[icoeff]*bispectrum[ii][icoeff];
      //        evdwl += coeffi[icoeff+1]*bispectrum[ii][icoeff];      

      // quadratic contributions

      if (quadraticflag) {
        int k = ncoeff+1;
        for (int icoeff = 0; icoeff < ncoeff; icoeff++) {
          double bveci = bispectrum[ii][icoeff];
          evdwl += 0.5*coeffi[k++]*bveci*bveci;
          for (int jcoeff = icoeff+1; jcoeff < ncoeff; jcoeff++) {
            double bvecj = bispectrum[ii][jcoeff];
            evdwl += coeffi[k++]*bveci*bvecj;
          }
        }
      }
      evdwl *= scale[itype][itype];
      ev_tally_full(i,2.0*evdwl,0.0,0.0,0.0,0.0,0.0);
    }

  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   compute beta
------------------------------------------------------------------------- */

double PairSNAPTTM::compute_electronic_temperature_dependent_betazero(double Te_input)
{
  double val = 0.0;
  for (int i = 0; i <= bzero_poly_order; i++) {
    val = val * Te_input + bzero_poly_coeffs[i];
  }
  return val;
}

void PairSNAPTTM::evaluate_electronic_temperature_dependent_betazero(const std::string& csv_path)
{
  double min_x = 0.;
  double max_x = 6.;
  int M = 10000;
  
  // Open CSV and write header + samples
  std::ofstream ofs(csv_path);
  if (!ofs) {
    throw std::runtime_error("Failed to open CSV file: " + csv_path);
  }
  ofs << "# Te b0(Te)\n";
  ofs << std::fixed << std::setprecision(17);
  
  // Evenly spaced points, inclusive of both endpoints when M >= 2
  for (int i = 0; i < M; ++i) {
    double xi;
    xi = min_x + (max_x - min_x) * static_cast<double>(i) / static_cast<double>(M - 1);
    
    // Horner's method at xi
    double yi = 0.0;
    for (int k = 0; k <= bzero_poly_order; ++k) {
      yi = yi * xi + bzero_poly_coeffs[k];
    }
    
    ofs << xi << " " << yi << "\n";
  }
}

void PairSNAPTTM::check_read_betas(const std::string& csv_path)
{

  // Open CSV and write header + samples
  std::ofstream ofs(csv_path);
  if (!ofs) {
    throw std::runtime_error("Failed to open CSV file: " + csv_path);
  }
  ofs << "# Te";
  for (int i = 0; i<ncoeffall; i++)
    ofs << " b" << i;
  ofs << "\n";  
  
  ofs << std::fixed << std::setprecision(17);

  for (int i = 0; i<ntelec; i++) {
    ofs << telec[i] << " ";
    for (int j = 0; j<ncoeffall; j++) {
      ofs << betas[j][i] << " ";
    }
    ofs << "\n";
  }
  
}

/* ----------------------------------------------------------------------
   compute beta
------------------------------------------------------------------------- */

void PairSNAPTTM::compute_beta()
{
  int i;
  int *type = atom->type;

  std::vector<double> coeffivec;
  Fix *fix = modify->get_fix_by_id(idref);
  double *fix_vector = fix->vector_atom;
  double conv_K_to_eV = 8.61732814974056e-5;  
  for (int ii = 0; ii < list->inum; ii++) {
    i = list->ilist[ii];
    double Te_loc = fix_vector[i] * conv_K_to_eV;
    const int itype = type[i];
    const int ielem = map[itype];
    double* coeffi = coeffelem[ielem];
    //    double* coeffi = compute_electronic_temperature_dependent_betas(Te_loc);
    coeffivec = beta_splines_eval_vec(Te_loc);
    for (int icoeff = 0; icoeff < ncoeff; icoeff++)
      beta[ii][icoeff] = coeffi[icoeff+1];

    // if (quadraticflag) {
    //   int k = ncoeff+1;
    //   for (int icoeff = 0; icoeff < ncoeff; icoeff++) {
    //     double bveci = bispectrum[ii][icoeff];
    //     beta[ii][icoeff] += coeffi[k]*bveci;
    //     k++;
    //     for (int jcoeff = icoeff+1; jcoeff < ncoeff; jcoeff++) {
    //       double bvecj = bispectrum[ii][jcoeff];
    //       beta[ii][icoeff] += coeffi[k]*bvecj;
    //       beta[ii][jcoeff] += coeffi[k]*bveci;
    //       k++;
    //     }
    //   }
    // }
  }
}

/* ----------------------------------------------------------------------
   compute bispectrum
------------------------------------------------------------------------- */

void PairSNAPTTM::compute_bispectrum()
{
  int i,j,jnum,ninside;
  double delx,dely,delz,rsq;
  int *jlist;

  double **x = atom->x;
  int *type = atom->type;

  for (int ii = 0; ii < list->inum; ii++) {
    i = list->ilist[ii];

    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];
    const int itype = type[i];
    const int ielem = map[itype];
    const double radi = radelem[ielem];

    jlist = list->firstneigh[i];
    jnum = list->numneigh[i];

    // ensure rij, inside, wj, and rcutij are of size jnum

    snaptr->grow_rij(jnum);

    // rij[][3] = displacements between atom I and those neighbors
    // inside = indices of neighbors of I within cutoff
    // wj = weights for neighbors of I within cutoff
    // rcutij = cutoffs for neighbors of I within cutoff
    // note Rij sign convention => dU/dRij = dU/dRj = -dU/dRi

    ninside = 0;
    for (int jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx*delx + dely*dely + delz*delz;
      int jtype = type[j];
      int jelem = map[jtype];

      if (rsq < cutsq[itype][jtype]&&rsq>1e-20) {
        snaptr->rij[ninside][0] = delx;
        snaptr->rij[ninside][1] = dely;
        snaptr->rij[ninside][2] = delz;
        snaptr->inside[ninside] = j;
        snaptr->wj[ninside] = wjelem[jelem];
        snaptr->rcutij[ninside] = (radi + radelem[jelem])*rcutfac;
        if (switchinnerflag) {
          snaptr->sinnerij[ninside] = 0.5*(sinnerelem[ielem]+sinnerelem[jelem]);
          snaptr->dinnerij[ninside] = 0.5*(dinnerelem[ielem]+dinnerelem[jelem]);
        }
        if (chemflag) snaptr->element[ninside] = jelem;
        ninside++;
      }
    }

    if (chemflag)
      snaptr->compute_ui(ninside, ielem);
    else
      snaptr->compute_ui(ninside, 0);
    snaptr->compute_zi();
    if (chemflag)
      snaptr->compute_bi(ielem);
    else
      snaptr->compute_bi(0);

    for (int icoeff = 0; icoeff < ncoeff; icoeff++) {
      bispectrum[ii][icoeff] = snaptr->blist[icoeff];
    }
  }

}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSNAPTTM::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(scale,n+1,n+1,"pair:scale");
  map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSNAPTTM::settings(int narg, char ** /* arg */)
{
  if (narg > 0)
    error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSNAPTTM::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  //  if (narg != 4 + atom->ntypes) error->all(FLERR,"Incorrect args for pair coefficients");
  if (narg != 6 + atom->ntypes)
    {
      std::cout << "nargs = " << narg << std::endl;
      error->all(FLERR,"Incorrect args for pair coefficients");

    }

  //  map_element2type(narg-4,arg+4);
  map_element2type(narg-6,arg+5);

  // Get input homogeneous electronic temperature
  //  Te_input = utils::numeric(FLERR,arg[7],false,lmp);
  int iarg = 6;
  ArgInfo argi(arg[iarg]);
  
  whichref = argi.get_type();
  idref = argi.copy_name();

  std::cout << "Pair style kiss with input fix " << idref << std::endl;
  
  if ((whichref == ArgInfo::FIX)) {
    auto ifix = modify->get_fix_by_id(idref);
    if (!ifix)
      error->all(FLERR,"Fix ID {} for fetching per-atom electronic temperature does not exist", idref);
  }

  // read snapcoeff and snapparam files

  read_files(arg[2],arg[3]);
  read_betas_files(arg[4]);

  //  std::abort();
  
  // Check if b0 is correctly evaluated using the polynomial function
  std::string csv_path = "dir.inputs/eval_betazero.csv";
  evaluate_electronic_temperature_dependent_betazero(csv_path);

  // Check if b1->bN are correctly read
  csv_path = "dir.inputs/eval_betas.csv";
  check_read_betas(csv_path);

  // Construct the spline evaluations of b1 to b55
  beta_splines_build();
  
  if (!quadraticflag)
    ncoeff = ncoeffall - 1;
  else {

    // ncoeffall should be (ncoeff+2)*(ncoeff+1)/2
    // so, ncoeff = floor(sqrt(2*ncoeffall))-1

    ncoeff = sqrt(2.0*ncoeffall)-1;
    ncoeffq = (ncoeff*(ncoeff+1))/2;
    int ntmp = 1+ncoeff+ncoeffq;
    if (ntmp != ncoeffall) {
      error->all(FLERR,"Incorrect SNAPTTM coeff file");
    }
  }

  snaptr = new SNA(lmp, rfac0, twojmax,
                   rmin0, switchflag, bzeroflag,
                   chemflag, bnormflag, wselfallflag,
                   nelements, switchinnerflag);

  if (ncoeff != snaptr->ncoeff) {
    if (comm->me == 0)
      printf("ncoeff = %d snancoeff = %d \n",ncoeff,snaptr->ncoeff);
    error->all(FLERR,"Incorrect SNAPTTM parameter file");
  }

  // Calculate maximum cutoff for all elements
  rcutmax = 0.0;
  for (int ielem = 0; ielem < nelements; ielem++)
    rcutmax = MAX(2.0*radelem[ielem]*rcutfac,rcutmax);

  // set default scaling
  int n = atom->ntypes;
  for (int ii = 0; ii < n+1; ii++)
    for (int jj = 0; jj < n+1; jj++)
      scale[ii][jj] = 1.0;

}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSNAPTTM::init_style()
{
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style SNAPTTM requires newton pair on");

  // need a full neighbor list

  neighbor->add_request(this, NeighConst::REQ_FULL);

  snaptr->init();

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSNAPTTM::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");
  scale[j][i] = scale[i][j];
  return (radelem[map[i]] +
          radelem[map[j]])*rcutfac;
}

/* ---------------------------------------------------------------------- */
void PairSNAPTTM::read_betas_files(char *alphasfilename)
{
  std::cout << "Reading kernel files..." << std::endl;

  // open SNAP kernel alphas coefficients file
  
  FILE *fpalphas;
  if (comm->me == 0) {
    fpalphas = utils::open_potential(alphasfilename,lmp,nullptr);
    if (fpalphas == nullptr)
      error->one(FLERR,"Cannot open SNAP kernel alphas file {}: ",
                                   alphasfilename, utils::getsyserror());
  }

  char line[MAXLINE],*ptr;
  int eof = 0;
  int nwords = 0;
  while (nwords == 0) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fpalphas);
      if (ptr == nullptr) {
        eof = 1;
        fclose(fpalphas);
      }
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);
    // strip comment, skip line if blank
    std::cout << "line = " << line << std::endl;
    nwords = utils::count_words(utils::trim_comment(line));
  }
  
  // // initialize checklist for all required nelements
  // int *elementflags = new int[nelements];
  // for (int jelem = 0; jelem < nelements; jelem++)
  //   elementflags[jelem] = 0;

  std::vector<std::string> words;
  try {
    words = Tokenizer(utils::trim_comment(line),"\"' \t\n\r\f").as_vector();
  } catch (TokenizerException &) {
    // ignore
  }
  if (words.size() != 2)
    error->all(FLERR,"AAAIncorrect format in SNAP kernel alphas file");
  
  ntelec = utils::numeric(FLERR,words[1],false,lmp);
  std::cout <<  "ntelec = " << ntelec << std::endl;
  
  std::cout <<  "words[0] = " << words[0] << std::endl;
  std::cout <<  "ntelec = " << ntelec << std::endl;  
  std::cout << "Creating betas tabs" << std::endl;
  std::cout << "betas size      = " << ncoeffall << " x " << ntelec << std::endl;
  std::cout << "telec           = " << ntelec << " x " << 1 << std::endl;
  
  memory->destroy(telec);  
  memory->create(telec,ntelec,"pair:telec");  

  if (comm->me == 0) {
    ptr = fgets(line,MAXLINE,fpalphas);
    if (ptr == nullptr) {
      eof = 1;
      fclose(fpalphas);
    }
  }
  MPI_Bcast(&eof,1,MPI_INT,0,world);
  MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

  nwords = utils::count_words(utils::trim_comment(line));

  if (nwords != ntelec)
    error->all(FLERR,"Incorrect format in SNAP kernel alphas file");
  
  int nelemtmp = 0;
  try {
    ValueTokenizer words(utils::trim_comment(line),"\"' \t\n\r\f");
    for (int l=0; l<ntelec; l++) {
      telec[l] = words.next_double();
    }
  } catch (TokenizerException &e) {
    error->all(FLERR,"Incorrect format in SNAP kernel alphas file: {}", e.what());
  }

  std::cout <<  "telec =";
  for (int l=0; l<ntelec; l++) {
    std::cout << " " << telec[l];
  }  
  std::cout << std::endl;

  memory->destroy(betas);
  memory->create(betas,ncoeffall,ntelec,"pair:alphas");
  
  // End of telec section

  // Start of poly order section
  if (comm->me == 0) {
    ptr = fgets(line,MAXLINE,fpalphas);
    if (ptr == nullptr) {
      eof = 1;
      fclose(fpalphas);
    }
  }
  MPI_Bcast(&eof,1,MPI_INT,0,world);
  MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

  try {
    words = Tokenizer(utils::trim_comment(line),"\"' \t\n\r\f").as_vector();
  } catch (TokenizerException &) {
    // ignore
  }

  bzero_poly_order = utils::numeric(FLERR,words[1],false,lmp);
  
  nwords = utils::count_words(utils::trim_comment(line));

  if (nwords != 2)
    error->all(FLERR,"Incorrect format in SNAP kernel alphas file");

  memory->destroy(bzero_poly_coeffs);
  memory->create(bzero_poly_coeffs,bzero_poly_order+1,"pair:bzero_poly_coeffs");
  
  if (comm->me == 0) {
    ptr = fgets(line,MAXLINE,fpalphas);
    if (ptr == nullptr) {
      eof = 1;
      fclose(fpalphas);
    }
  }
  MPI_Bcast(&eof,1,MPI_INT,0,world);

  MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

  try {
    words = Tokenizer(utils::trim_comment(line),"\"' \t\n\r\f").as_vector();
  } catch (TokenizerException &) {
    // ignore
  }
  
  nwords = utils::count_words(utils::trim_comment(line));
  std::cout << "nwords = " << nwords << std::endl;
  std::cout << "line   = " << line << std::endl;
  std::cout << "order  = " << bzero_poly_order << std::endl;
  
  if (nwords != bzero_poly_order+1)
    error->all(FLERR,"Incorrect format in SNAP kernel alphas file");
  
  try {
    ValueTokenizer words(utils::trim_comment(line),"\"' \t\n\r\f");
    for (int l=0; l<bzero_poly_order+1; l++) {
      bzero_poly_coeffs[l] = words.next_double();
    }
  } catch (TokenizerException &e) {
    error->all(FLERR,"Incorrect format in SNAP kernel alphas file: {}", e.what());
  }

  for (int l=0; l<bzero_poly_order+1; l++) {
    std::cout << "bzero_poly_coeffs[" << l << "] = " << bzero_poly_coeffs[l] << std::endl;  
  }
  // End of poly order section

  for (int icoeff = 0; icoeff < ncoeffall; icoeff++) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fpalphas);
      if (ptr == nullptr) {
	eof = 1;
	fclose(fpalphas);
      }
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof)
      error->all(FLERR,"AAAIncorrect format in SNAP kernel alphas file");
    MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);
    
    try {
      words = Tokenizer(utils::trim_comment(line),"\"' \t\n\r\f").as_vector();      
      if (words.size() != ntelec)
	error->all(FLERR,"BBBIncorrect format in SNAP coefficient file");
      for (int l=0; l<ntelec; l++){
        betas[icoeff][l] = utils::numeric(FLERR,words[l],false,lmp);
      }
      
    } catch (TokenizerException &e) {
      error->all(FLERR,"Incorrect format in SNAP coefficient file: {}", e.what());
    }
    
  }
  if (comm->me == 0) fclose(fpalphas);

  std::cout << "At this point, the alphas coefficients for the kernel are read!" << std::endl;
}

void PairSNAPTTM::read_files(char *coefffilename, char *paramfilename)
{

  // open SNAPTTM coefficient file on proc 0

  FILE *fpcoeff;
  if (comm->me == 0) {
    fpcoeff = utils::open_potential(coefffilename,lmp,nullptr);
    if (fpcoeff == nullptr)
      error->one(FLERR,"Cannot open SNAPTTM coefficient file {}: ",
                                   coefffilename, utils::getsyserror());
  }

  char line[MAXLINE] = {'\0'};
  char *ptr;
  int eof = 0;
  int nwords = 0;
  while (nwords == 0) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fpcoeff);
      if (ptr == nullptr) {
        eof = 1;
        fclose(fpcoeff);
      }
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

    // strip comment, skip line if blank

    nwords = utils::count_words(utils::trim_comment(line));
  }
  if (nwords != 2)
    error->all(FLERR,"Incorrect format in SNAPTTM coefficient file");

  // strip single and double quotes from words

  int nelemtmp = 0;
  try {
    ValueTokenizer words(utils::trim_comment(line),"\"' \t\n\r\f");
    nelemtmp = words.next_int();
    ncoeffall = words.next_int();
  } catch (TokenizerException &e) {
    error->all(FLERR,"Incorrect format in SNAPTTM coefficient file: {}", e.what());
  }

  // clean out old arrays and set up element lists

  memory->destroy(radelem);
  memory->destroy(wjelem);
  memory->destroy(coeffelem);
  memory->destroy(sinnerelem);
  memory->destroy(dinnerelem);
  memory->create(radelem,nelements,"pair:radelem");
  memory->create(wjelem,nelements,"pair:wjelem");
  memory->create(coeffelem,nelements,ncoeffall,"pair:coeffelem");
  memory->create(sinnerelem,nelements,"pair:sinnerelem");
  memory->create(dinnerelem,nelements,"pair:dinnerelem");

  // initialize checklist for all required nelements

  int *elementflags = new int[nelements];
  for (int jelem = 0; jelem < nelements; jelem++)
      elementflags[jelem] = 0;

  // loop over nelemtmp blocks in the SNAPTTM coefficient file

  for (int ielem = 0; ielem < nelemtmp; ielem++) {

    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fpcoeff);
      if (ptr == nullptr) {
        eof = 1;
        fclose(fpcoeff);
      }
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof)
      error->all(FLERR,"Incorrect format in SNAPTTM coefficient file");
    MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

    std::vector<std::string> words;
    try {
      words = Tokenizer(utils::trim_comment(line),"\"' \t\n\r\f").as_vector();
    } catch (TokenizerException &) {
      // ignore
    }
    if (words.size() != 3)
      error->all(FLERR,"Incorrect format in SNAPTTM coefficient file");

    std::cout <<  "WORDS = " << words[0] << " " << words[1] << " " << words[2] << std::endl;
    
    int jelem;
    for (jelem = 0; jelem < nelements; jelem++)
      if (words[0] == elements[jelem]) break;

    // if this element not needed, skip this block

    if (jelem == nelements) {
      if (comm->me == 0) {
        for (int icoeff = 0; icoeff < ncoeffall; icoeff++) {
          ptr = fgets(line,MAXLINE,fpcoeff);
          if (ptr == nullptr) {
            eof = 1;
            fclose(fpcoeff);
          }
        }
      }
      MPI_Bcast(&eof,1,MPI_INT,0,world);
      if (eof)
        error->all(FLERR,"Incorrect format in SNAPTTM coefficient file");
      continue;
    }

    if (elementflags[jelem] == 1)
      error->all(FLERR,"Incorrect format in SNAPTTM coefficient file");
    else
      elementflags[jelem] = 1;

    radelem[jelem] = utils::numeric(FLERR,words[1],false,lmp);
    wjelem[jelem] = utils::numeric(FLERR,words[2],false,lmp);

    if (comm->me == 0)
      utils::logmesg(lmp,"SNAPTTM Element = {}, Radius {}, Weight {}\n",
                     elements[jelem], radelem[jelem], wjelem[jelem]);

    for (int icoeff = 0; icoeff < ncoeffall; icoeff++) {
      if (comm->me == 0) {
        ptr = fgets(line,MAXLINE,fpcoeff);
        if (ptr == nullptr) {
          eof = 1;
          fclose(fpcoeff);
        }
      }

      MPI_Bcast(&eof,1,MPI_INT,0,world);
      if (eof)
        error->all(FLERR,"Incorrect format in SNAPTTM coefficient file");
      MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

      try {
        ValueTokenizer coeff(utils::trim_comment(line));
        if (coeff.count() != 1)
          error->all(FLERR,"Incorrect format in SNAPTTM coefficient file");

        coeffelem[jelem][icoeff] = coeff.next_double();
      } catch (TokenizerException &e) {
        error->all(FLERR,"Incorrect format in SNAPTTM coefficient file: {}", e.what());
      }
    }
  }

  if (comm->me == 0) fclose(fpcoeff);

  for (int jelem = 0; jelem < nelements; jelem++) {
    if (elementflags[jelem] == 0)
      error->all(FLERR,"Element {} not found in SNAPTTM coefficient file", elements[jelem]);
  }
  delete[] elementflags;

  // set flags for required keywords

  rcutfacflag = 0;
  twojmaxflag = 0;

  // Set defaults for optional keywords

  rfac0 = 0.99363;
  rmin0 = 0.0;
  switchflag = 1;
  bzeroflag = 1;
  quadraticflag = 0;
  chemflag = 0;
  bnormflag = 0;
  wselfallflag = 0;
  switchinnerflag = 0;
  chunksize = 32768;
  parallel_thresh = 8192;

  // set local input checks

  int sinnerflag = 0;
  int dinnerflag = 0;

  // open SNAPTTM parameter file on proc 0

  FILE *fpparam;
  if (comm->me == 0) {
    fpparam = utils::open_potential(paramfilename,lmp,nullptr);
    if (fpparam == nullptr)
      error->one(FLERR,"Cannot open SNAPTTM parameter file {}: {}",
                                   paramfilename, utils::getsyserror());
  }

  eof = 0;
  while (true) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fpparam);
      if (ptr == nullptr) {
        eof = 1;
        fclose(fpparam);
      }
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

    // words = ptrs to all words in line
    // strip single and double quotes from words

    std::vector<std::string> words;
    try {
      words = Tokenizer(utils::trim_comment(line),"\"' \t\n\r\f").as_vector();
    } catch (TokenizerException &) {
      // ignore
    }

    if (words.size() == 0) continue;

    if (words.size() < 2)
      error->all(FLERR,"Incorrect format in SNAPTTM parameter file");

    auto keywd = words[0];
    auto keyval = words[1];

    // check for keywords with more than one value per element

    if (keywd == "sinner" || keywd == "dinner") {

      if ((int)words.size() != nelements+1)
        error->all(FLERR,"Incorrect SNAPTTM parameter file");

      // innerlogstr collects all values of sinner or dinner for log output below

      std::string innerlogstr;

      int iword = 1;

      if (keywd == "sinner") {
        for (int ielem = 0; ielem < nelements; ielem++) {
          keyval = words[iword];
          sinnerelem[ielem] = utils::numeric(FLERR,keyval,false,lmp);
          iword++;
          innerlogstr += keyval + " ";
        }
        sinnerflag = 1;
      } else if (keywd == "dinner") {
        for (int ielem = 0; ielem < nelements; ielem++) {
          keyval = words[iword];
          dinnerelem[ielem] = utils::numeric(FLERR,keyval,false,lmp);
          iword++;
          innerlogstr += keyval + " ";
        }
        dinnerflag = 1;
      }

      if (comm->me == 0)
        utils::logmesg(lmp,"SNAPTTM keyword {} {} ... \n", keywd, innerlogstr);

    } else {

      // all other keywords take one value

      if (nwords != 2)
        error->all(FLERR,"Incorrect SNAPTTM parameter file");

      if (comm->me == 0)
        utils::logmesg(lmp,"SNAPTTM keyword {} {}\n",keywd,keyval);

      if (keywd == "rcutfac") {
        rcutfac = utils::numeric(FLERR,keyval,false,lmp);
        rcutfacflag = 1;
      } else if (keywd == "twojmax") {
        twojmax = utils::inumeric(FLERR,keyval,false,lmp);
        twojmaxflag = 1;
      } else if (keywd == "rfac0")
        rfac0 = utils::numeric(FLERR,keyval,false,lmp);
      else if (keywd == "rmin0")
        rmin0 = utils::numeric(FLERR,keyval,false,lmp);
      else if (keywd == "switchflag")
        switchflag = utils::inumeric(FLERR,keyval,false,lmp);
      else if (keywd == "bzeroflag")
        bzeroflag = utils::inumeric(FLERR,keyval,false,lmp);
      else if (keywd == "quadraticflag")
        quadraticflag = utils::inumeric(FLERR,keyval,false,lmp);
      else if (keywd == "chemflag")
        chemflag = utils::inumeric(FLERR,keyval,false,lmp);
      else if (keywd == "bnormflag")
        bnormflag = utils::inumeric(FLERR,keyval,false,lmp);
      else if (keywd == "wselfallflag")
        wselfallflag = utils::inumeric(FLERR,keyval,false,lmp);
      else if (keywd == "switchinnerflag")
        switchinnerflag = utils::inumeric(FLERR,keyval,false,lmp);
      else if (keywd == "chunksize")
        chunksize = utils::inumeric(FLERR,keyval,false,lmp);
      else if (keywd == "parallelthresh")
        parallel_thresh = utils::inumeric(FLERR,keyval,false,lmp);
      else
        error->all(FLERR,"Unknown parameter '{}' in SNAPTTM parameter file", keywd);
    }
  }

  if (rcutfacflag == 0 || twojmaxflag == 0)
    error->all(FLERR,"Incorrect SNAPTTM parameter file");

  if (chemflag && nelemtmp != nelements)
    error->all(FLERR,"Incorrect SNAPTTM parameter file");

  if (switchinnerflag && !(sinnerflag && dinnerflag))
    error->all(FLERR,"Incorrect SNAPTTM parameter file");

  if (!switchinnerflag && (sinnerflag || dinnerflag))
    error->all(FLERR,"Incorrect SNAPTTM parameter file");
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double PairSNAPTTM::memory_usage()
{
  double bytes = Pair::memory_usage();

  int n = atom->ntypes+1;
  bytes += (double)n*n*sizeof(int);      // setflag
  bytes += (double)n*n*sizeof(double);   // cutsq
  bytes += (double)n*n*sizeof(double);   // scale
  bytes += (double)n*sizeof(int);        // map
  bytes += (double)beta_max*ncoeff*sizeof(double); // bispectrum
  bytes += (double)beta_max*ncoeff*sizeof(double); // beta

  bytes += snaptr->memory_usage(); // SNA object

  return bytes;
}

/* ---------------------------------------------------------------------- */

void *PairSNAPTTM::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"scale") == 0) return (void *) scale;
  return nullptr;
}

double* PairSNAPTTM::compute_electronic_temperature_dependent_betas(double Te_input)
{
  
  double result[ncoeffall];
  for (int i = 0; i < ncoeffall; ++i) result[i] = 0.0;
  return result;
}


void PairSNAPTTM::evaluate_electronic_temperature_dependent_betas(const std::string& csv_path)
{
  double min_x = 0.;
  double max_x = 6.;
  int M = 10000;
  
  // Open CSV and write header + samples
  std::ofstream ofs(csv_path);
  if (!ofs) {
    throw std::runtime_error("Failed to open CSV file: " + csv_path);
  }
  ofs << "# Te b0(Te)\n";
  // ofs << "# Te b0(Te)\n";  
  // ofs << std::fixed << std::setprecision(17);
  
  // // Evenly spaced points, inclusive of both endpoints when M >= 2
  // for (int i = 0; i < M; ++i) {
  //   double xi;
  //   xi = min_x + (max_x - min_x) * static_cast<double>(i) / static_cast<double>(M - 1);
    
  //   // Horner's method at xi
  //   double yi = 0.0;
  //   for (int k = 0; k <= bzero_poly_order; ++k) {
  //     yi = yi * xi + bzero_poly_coeffs[k];
  //   }
    
  //   ofs << xi << " " << yi << "\n";
  // }
}


void PairSNAPTTM::factor_tridiagonal_natural(const std::vector<double>& h,
                                       std::vector<double>& cprime,
                                       std::vector<double>& denom)
{
    const int N = (int)denom.size();
    // Build the tridiagonal coefficients a, b, c (natural BCs)
    std::vector<double> a(N, 0.0), b(N, 0.0), c(N, 0.0);
    b[0] = 1.0;
    b[N-1] = 1.0;
    for (int i = 1; i <= N-2; ++i) {
        a[i] = h[i-1];
        b[i] = 2.0 * (h[i-1] + h[i]);
        c[i] = h[i];
    }

    // Thomas factorization (store modified diagonal in denom and modified superdiag in cprime)
    cprime[0] = (N > 1) ? c[0] / b[0] : 0.0; // c[0]=0; harmless
    denom[0]  = b[0];
    for (int i = 1; i < N; ++i) {
        denom[i] = b[i] - a[i] * cprime[i-1];
        cprime[i] = (i == N-1) ? 0.0 : c[i] / denom[i];
    }
}

// Solve A*cvec = rhs where A is the natural spline tridiagonal factorized by factor_tridiagonal_natural.
// Inputs: h (for forward subst sub-diagonal = a[i]=h[i-1]), cprime, denom; rhs is in/out (solution returned there).
void PairSNAPTTM::solve_tridiagonal_natural(const std::vector<double>& h,
                               const std::vector<double>& cprime,
                               const std::vector<double>& denom,
                               std::vector<double>& rhs)
{
    const int N = (int)rhs.size();

    // Forward substitution (using a[i]=h[i-1])
    std::vector<double> y(N);
    y[0] = rhs[0] / denom[0];
    for (int i = 1; i < N; ++i) {
        y[i] = (rhs[i] - h[i-1] * y[i-1]) / denom[i];
    }

    // Back substitution
    rhs[N-1] = y[N-1];
    for (int i = N - 2; i >= 0; --i) {
        rhs[i] = y[i] - cprime[i] * rhs[i+1];
    }
}

// Compute per-row spline coefficients (a,b,c,d) for all intervals and store into S.coeffs.
void PairSNAPTTM::build_row_coeffs(const double* y, const std::vector<double>& h,
                                   const std::vector<double>& cprime,
                                   const std::vector<double>& denom,
                                   int N, int row, BetaSplines& S)
{
    // RHS for natural spline system
    std::vector<double> rhs(N, 0.0);
    rhs[0] = 0.0;
    rhs[N-1] = 0.0;
    for (int i = 1; i <= N-2; ++i) {
        const double slope_next = (y[i+1] - y[i]) / h[i];
        const double slope_prev = (y[i]   - y[i-1]) / h[i-1];
        rhs[i] = 3.0 * (slope_next - slope_prev);
    }

    // Solve for cvec (second-derivative coefficients at knots)
    solve_tridiagonal_natural(h, cprime, denom, rhs);
    const std::vector<double>& cvec = rhs;

    // For each interval k: S_k(t) = a + b t + c t^2 + d t^3, t = x - x_k
    for (int k = 0; k < N - 1; ++k) {
        const double ak = y[k];
        const double ck = cvec[k];
        const double dk = (cvec[k+1] - cvec[k]) / (3.0 * h[k]);
        const double bk = (y[k+1] - y[k]) / h[k] - (2.0*ck + cvec[k+1]) * h[k] / 3.0;

        const size_t base = (size_t)row * (size_t)(N - 1) * 4 + (size_t)k * 4;
        S.coeffs[base + 0] = ak; // a
        S.coeffs[base + 1] = bk; // b
        S.coeffs[base + 2] = ck; // c
        S.coeffs[base + 3] = dk; // d
    }
}

void PairSNAPTTM::beta_splines_build()
{
  int N = ntelec;
  int M = ncoeffall;
  if (N < 2) throw std::invalid_argument("beta_splines_build: Need at least 2 knots");
  if (M < 1) throw std::invalid_argument("beta_splines_build: Need at least 1 row");
  
  // Copy knots and validate strict increase
  BetaSpl.N = N;
  BetaSpl.M = M;
  BetaSpl.x.assign(telec, telec + N);
  BetaSpl.h.resize(N - 1);
  for (int i = 1; i < N; ++i) {
    if (!(BetaSpl.x[i] > BetaSpl.x[i-1])) {
      throw std::invalid_argument("beta_splines_build: telec must be strictly increasing");
    }
    BetaSpl.h[i-1] = BetaSpl.x[i] - BetaSpl.x[i-1];
  }

  // Prepare factorization buffers
  BetaSpl.cprime.assign(N, 0.0);
  BetaSpl.denom.assign(N, 0.0);
  factor_tridiagonal_natural(BetaSpl.h, BetaSpl.cprime, BetaSpl.denom);

  // Allocate coeff storage: M rows × (N-1 intervals) × 4 coefficients
  BetaSpl.coeffs.assign((size_t)M * (size_t)(N - 1) * 4, 0.0);

  // Compute coefficients row by row
  for (int row = 0; row < M; ++row) {
    if (!betas[row]) throw std::invalid_argument("beta_splines_build: betas[row] is null");
    build_row_coeffs(betas[row], BetaSpl.h, BetaSpl.cprime, BetaSpl.denom, N, row, BetaSpl);
  }
  std::cout << "Beta Splines builds is DONE." << std::endl << std::flush;

}

int PairSNAPTTM::beta_splines_find_interval(double x)
{
  
  if (BetaSpl.N < 2) return 0;
  if (x <= BetaSpl.x.front()) return 0;
  if (x >= BetaSpl.x.back())  return BetaSpl.N - 2;
  auto it = std::upper_bound(BetaSpl.x.begin(), BetaSpl.x.end(), x);
  return int((it - BetaSpl.x.begin()) - 1); // k s.t. x[k] <= x < x[k+1]
}

void PairSNAPTTM::beta_splines_eval(double x, double* out)
{
    if (!out) throw std::invalid_argument("beta_splines_eval: out is null");
    if (BetaSpl.N < 2 || BetaSpl.M < 1) throw std::invalid_argument("beta_splines_eval: empty cache");

    const int k = beta_splines_find_interval(x);
    const double t = x - BetaSpl.x[k];

    // Evaluate each row on interval k: (((d*t)+c)*t + b)*t + a
    const size_t row_stride = (size_t)(BetaSpl.N - 1) * 4;
    const size_t base_k = (size_t)k * 4;
    for (int row = 0; row < BetaSpl.M; ++row) {
        const size_t base = (size_t)row * row_stride + base_k;
        const double a = BetaSpl.coeffs[base + 0];
        const double b = BetaSpl.coeffs[base + 1];
        const double c = BetaSpl.coeffs[base + 2];
        const double d = BetaSpl.coeffs[base + 3];
        out[row] = ((d * t + c) * t + b) * t + a;
    }
}

std::vector<double> PairSNAPTTM::beta_splines_eval_vec(double x)
{
    std::vector<double> out(BetaSpl.M);
    beta_splines_eval(x, out.data());
    return out;
}

// void PairSNAPTTM::beta_splines_eval_many(const BetaSplines& S, const double* xs, int K, double* out)
// {
//     if (!xs || !out) throw std::invalid_argument("beta_splines_eval_many: null pointer");
//     if (K < 0) throw std::invalid_argument("beta_splines_eval_many: negative K");
//     if (S.N < 2 || S.M < 1) throw std::invalid_argument("beta_splines_eval_many: empty cache");

//     const size_t row_stride = (size_t)(S.N - 1) * 4;

//     for (int q = 0; q < K; ++q) {
//         const double x = xs[q];
//         const int k = beta_splines_find_interval(S, x);
//         const double t = x - S.x[k];
//         const size_t base_k = (size_t)k * 4;

//         for (int row = 0; row < S.M; ++row) {
//             const size_t base = (size_t)row * row_stride + base_k;
//             const double a = S.coeffs[base + 0];
//             const double b = S.coeffs[base + 1];
//             const double c = S.coeffs[base + 2];
//             const double d = S.coeffs[base + 3];
//             out[(size_t)q * (size_t)S.M + (size_t)row] = ((d * t + c) * t + b) * t + a;
//         }
//     }
// }
