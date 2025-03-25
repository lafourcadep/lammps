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
    memory->grow(betazero,list->inum,"PairSNAPTTM:betazero");    
    memory->grow(bispectrum,list->inum,ncoeff,"PairSNAPTTM:bispectrum");
    beta_max = list->inum;
  }

  // compute dE_i/dB_i = beta_i for all i in list

  if (quadraticflag || eflag)
    compute_bispectrum();
  compute_beta();

  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

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

      // evdwl = energy of atom I, sum over coeffs_k * Bi_k
      double* coeffi = beta[ii];
      //      double* coeffi = coeffelem[ielem];
      Fix *fix = modify->fix[ref2index];
      double *fix_vector = modify->fix[ref2index]->vector_atom;
      double conv_K_to_eV = 8.61732814974056e-5;
      int i = list->ilist[ii];
      double Te_loc = fix_vector[i] * conv_K_to_eV;
      evdwl = compute_electronic_temperature_dependent_betazero(Te_loc);

      //      std::cout << "evdwl = " << Te_loc << " " << evdwl << std::endl;
      
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

double* PairSNAPTTM::compute_electronic_temperature_dependent_coeff(double Te_input)
{

  //  std::cout << "Computing new beta coefficients for telec = " << Te_input << " eV" << std::endl;

  // For now, consider valid only for 1 element SNAP potentials
  const int ielem=0;
  double gamma=0.25;

  // Loop over kernel alpha vectors
  for (int icoeff=0; icoeff<ncoeffall; icoeff++)
    {
      double beta_coeff;
      double kernel_sum=0.;
      for (int jte=0; jte<ntelec; jte++)
        {
          double expterm = exp(-gamma * (Te_input - telec[jte]) * (Te_input - telec[jte]));
          kernel_sum += (betas[icoeff][jte] * expterm);
        }
      
      beta_coeff = kernel_sum;
      coeffelem[ielem][icoeff] = beta_coeff;
    }

  // std::cout << Te_input << " ";
  // for (int icoeff=0; icoeff<ncoeffall; icoeff++)
  //   {
  //     std::cout << coeffelem[ielem][icoeff] << " ";
  //   }
  // std::cout << std::endl;
  
  return coeffelem[ielem];
}

double PairSNAPTTM::compute_electronic_temperature_dependent_betazero(double Te_input)
{

  // For now, consider valid only for 1 element SNAP potentials
  const int ielem=0;
  double gamma=0.25;

  double beta_coeff;
  double kernel_sum=0.;
  for (int jte=0; jte<ntelec; jte++)
    {
      double expterm = exp(-gamma * (Te_input - telec[jte]) * (Te_input - telec[jte]));
      kernel_sum += (betas[0][jte] * expterm);
    }      
  beta_coeff = kernel_sum;  
  return beta_coeff;
}

/* ----------------------------------------------------------------------
   compute beta
------------------------------------------------------------------------- */

void PairSNAPTTM::compute_beta()
{
  int i;
  int *type = atom->type;

  Fix *fix = modify->fix[ref2index];
  double *fix_vector = modify->fix[ref2index]->vector_atom;
  double conv_K_to_eV = 8.61732814974056e-5;  
  for (int ii = 0; ii < list->inum; ii++) {
    i = list->ilist[ii];
    double Te_loc = fix_vector[i] * conv_K_to_eV;    
    const int itype = type[i];
    const int ielem = map[itype];
    //    double* coeffi = coeffelem[ielem];
    double* coeffi = compute_electronic_temperature_dependent_coeff(Te_loc);

    betazero[ii] = coeffi[0];    
    for (int icoeff = 0; icoeff < ncoeff; icoeff++)
      beta[ii][icoeff] = coeffi[icoeff+1];

    if (quadraticflag) {
      int k = ncoeff+1;
      for (int icoeff = 0; icoeff < ncoeff; icoeff++) {
        double bveci = bispectrum[ii][icoeff];
        beta[ii][icoeff] += coeffi[k]*bveci;
        k++;
        for (int jcoeff = icoeff+1; jcoeff < ncoeff; jcoeff++) {
          double bvecj = bispectrum[ii][jcoeff];
          beta[ii][icoeff] += coeffi[k]*bvecj;
          beta[ii][jcoeff] += coeffi[k]*bveci;
          k++;
        }
      }
    }
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
    
    nwords = utils::count_words(utils::trim_comment(line));
  }
  if (nwords != 2)
    error->all(FLERR,"Incorrect format in SNAP kernel alphas file");

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
  std::cout <<  "words[0] = " << words[0] << std::endl;
  std::cout <<  "ntelec = " << ntelec << std::endl;  
  std::cout << "Creating betas tabs" << std::endl;
  std::cout << "betas size      = " << ncoeffall << " x " << ntelec << std::endl;
  std::cout << "telec           = " << ntelec << " x " << 1 << std::endl;
  
  // memory->destroy(betas);
  // memory->destroy(telec);  
  // memory->create(betas,ncoeffall,ntelec,"pair:betas");
  // memory->create(telec,ntelec,"pair:telec");  

  // if (comm->me == 0) {
  //   ptr = fgets(line,MAXLINE,fpalphas);
  //   if (ptr == nullptr) {
  //     eof = 1;
  //     fclose(fpalphas);
  //   }
  // }
  // MPI_Bcast(&eof,1,MPI_INT,0,world);

  // MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

  // nwords = utils::count_words(utils::trim_comment(line));

  // if (nwords != ntelec)
  //   error->all(FLERR,"Incorrect format in SNAP kernel alphas file");

  // int nelemtmp = 0;
  // try {
  //   ValueTokenizer words(utils::trim_comment(line),"\"' \t\n\r\f");
  //   for (int l=0; l<ntelec; l++) {
  //     telec[l] = words.next_double();
  //   }
  // } catch (TokenizerException &e) {
  //   error->all(FLERR,"Incorrect format in SNAP kernel alphas file: {}", e.what());
  // }

  // std::cout <<  "telec =";
  // for (int l=0; l<ntelec; l++) {
  //   std::cout << " " << telec[l];
  // }  
  // std::cout << std::endl;

  // //////////////////////////////////////
  // if (comm->me == 0) {
  //   ptr = fgets(line,MAXLINE,fpalphas);
  //   if (ptr == nullptr) {
  //     eof = 1;
  //     fclose(fpalphas);
  //   }
  // }
  // MPI_Bcast(&eof,1,MPI_INT,0,world);

  // MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

  // nwords = utils::count_words(utils::trim_comment(line));
  // bzero_poly_order = utils::numeric(FLERR,words[1],false,lmp);
  // std::cout << "words[0] = " << words[0] << std::endl;
  // std::cout << "words[1] = " << words[1] << std::endl;    
  // std::cout << "bzero_poly_order = " << bzero_poly_order << std::endl;  

  // std::abort();
  
  // memory->destroy(bzero_poly_coeffs);
  // memory->create(bzero_poly_coeffs,bzero_poly_order,"pair:bzero_poly_coeffs");

  // if (comm->me == 0) {
  //   ptr = fgets(line,MAXLINE,fpalphas);
  //   if (ptr == nullptr) {
  //     eof = 1;
  //     fclose(fpalphas);
  //   }
  // }
  // MPI_Bcast(&eof,1,MPI_INT,0,world);

  // MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

  // nwords = utils::count_words(utils::trim_comment(line));

  // if (nwords != bzero_poly_order)
  //   error->all(FLERR,"Incorrect format in SNAP kernel alphas file");
  
  // try {
  //   ValueTokenizer words(utils::trim_comment(line),"\"' \t\n\r\f");
  //   for (int l=0; l<bzero_poly_order; l++) {
  //     bzero_poly_coeffs[l] = words.next_double();
  //   }
  // } catch (TokenizerException &e) {
  //   error->all(FLERR,"Incorrect format in SNAP kernel alphas file: {}", e.what());
  // }
  
  // for (int l=0; l<bzero_poly_order; l++) {
  //   std::cout << "bzero_poly_coeffs[" << l << "] = " << bzero_poly_coeffs[l] << std::endl;  
  // }
  // /////////////////////////////////////
  // std::abort();
  
  // for (int icoeff = 0; icoeff < ncoeffall; icoeff++) {
  //   if (comm->me == 0) {
  //     ptr = fgets(line,MAXLINE,fpalphas);
  //     if (ptr == nullptr) {
	// eof = 1;
	// fclose(fpalphas);
  //     }
  //   }
  //   MPI_Bcast(&eof,1,MPI_INT,0,world);
  //   if (eof)
  //     error->all(FLERR,"AAAIncorrect format in SNAP kernel alphas file");
  //   MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);
    
  //   try {
  //     words = Tokenizer(utils::trim_comment(line),"\"' \t\n\r\f").as_vector();      
  //     if (words.size() != ntelec)
	// error->all(FLERR,"BBBIncorrect format in SNAP coefficient file");
  //     for (int l=0; l<ntelec; l++){
  //       betas[icoeff][l] = utils::numeric(FLERR,words[l],false,lmp);
  //     }
      
  //   } catch (TokenizerException &e) {
  //     error->all(FLERR,"Incorrect format in SNAP coefficient file: {}", e.what());
  //   }
    
  // }
  // if (comm->me == 0) fclose(fpalphas);

  std::abort();
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

void PairSNAPTTM::interpolate(int n, double delta, double *f, double **spline)
{
  for (int m = 1; m <= n; m++) spline[m][6] = f[m];

  spline[1][5] = spline[2][6] - spline[1][6];
  spline[2][5] = 0.5 * (spline[3][6]-spline[1][6]);
  spline[n-1][5] = 0.5 * (spline[n][6]-spline[n-2][6]);
  spline[n][5] = spline[n][6] - spline[n-1][6];

  for (int m = 3; m <= n-2; m++)
    spline[m][5] = ((spline[m-2][6]-spline[m+2][6]) +
                    8.0*(spline[m+1][6]-spline[m-1][6])) / 12.0;

  for (int m = 1; m <= n-1; m++) {
    spline[m][4] = 3.0*(spline[m+1][6]-spline[m][6]) -
      2.0*spline[m][5] - spline[m+1][5];
    spline[m][3] = spline[m][5] + spline[m+1][5] -
      2.0*(spline[m+1][6]-spline[m][6]);
  }

  spline[n][4] = 0.0;
  spline[n][3] = 0.0;

  for (int m = 1; m <= n; m++) {
    spline[m][2] = spline[m][5]/delta;
    spline[m][1] = 2.0*spline[m][4]/delta;
    spline[m][0] = 3.0*spline[m][3]/delta;
  }
}
