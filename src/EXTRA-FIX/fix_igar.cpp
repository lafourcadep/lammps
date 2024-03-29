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

/* ----------------------------------------------------------------------
   Contributing authors: Paul Crozier (SNL)
                         Carolyn Phillips (University of Michigan)
------------------------------------------------------------------------- */

#include "fix_igar.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "random_mars.h"
//#include "respa.h"
#include "potential_file_reader.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;

// OFFSET avoids outside-of-box atoms being rounded to grid pts incorrectly
// SHIFT = 0.0 assigns atoms to lower-left grid pt
// SHIFT = 0.5 assigns atoms to nearest grid pt
// use SHIFT = 0.0 for now since it allows fix ave/chunk
//   to spatially average consistent with the IGAR grid

static constexpr int OFFSET = 16384;
static constexpr double SHIFT = 0.0;

/* ---------------------------------------------------------------------- */

FixIGAR::FixIGAR(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  figar(nullptr), eigar(nullptr),
  U_igar(nullptr), /*U_igar_old(nullptr),*/
  igar_energy_transfer(nullptr), igar_energy_transfer_all(nullptr)
{
  if (narg != 9) error->all(FLERR,"Illegal fix igar command");

  peratom_flag = 1;
  size_vector = 2;
  nevery = 1;
  
  thermo_energy = 1;
  // These 4 are needed to ensure energy conservation
  ecouple_flag = 1;
  scalar_flag = 1;
  energy_global_flag = 1;
  global_freq = 1;  
  extscalar = 1;  

  kIm = utils::inumeric(FLERR,arg[3],false,lmp);
  std::cout << "kIm = " << kIm << std::endl;
  
  nxgrid = utils::inumeric(FLERR,arg[4],false,lmp);
  nygrid = utils::inumeric(FLERR,arg[5],false,lmp);
  nzgrid = utils::inumeric(FLERR,arg[6],false,lmp);
  std::cout << "nx = " << nxgrid << std::endl;
  std::cout << "ny = " << nygrid << std::endl;
  std::cout << "nz = " << nzgrid << std::endl;  
  infile = nullptr;
  if (strcmp(arg[7],"infile") == 0) {
    infile = utils::strdup(arg[8]);
  }

  // outfile = nullptr;
  // if (strcmp(arg[9],"outfile") == 0) {
  //   infile = utils::strdup(arg[10]);
  // }
  
  if (nxgrid <= 0 || nygrid <= 0 || nzgrid <= 0)
    error->all(FLERR,"Fix hrtem grid sizes must be > 0");
  
  // grid OFFSET to perform
  // SHIFT to map atom to nearest or lower-left grid point
  
  shift = OFFSET + SHIFT;
  
  // check for allowed maximum number of total grid points

  bigint totalgrid = (bigint) nxgrid * nygrid * nzgrid;
  if (totalgrid > MAXSMALLINT)
    error->all(FLERR,"Too many grid points in fix hrtem");
  ngridtotal = totalgrid;

  // allocate per-atom figar and zero it

  figar = nullptr;
  eigar = nullptr;
  FixIGAR::grow_arrays(atom->nmax);
  
  for (int i = 0; i < atom->nmax; i++) {
    figar[i][0] = 0.0;
    figar[i][1] = 0.0;
    figar[i][2] = 0.0;
    eigar[i] = 0.0;
  }
  vector_atom = eigar;

  // set 2 callbacks

  atom->add_callback(Atom::GROW);
  atom->add_callback(Atom::RESTART);

  // determines which class deallocate_grid() is called from

  deallocate_flag = 0;
}

/* ---------------------------------------------------------------------- */

FixIGAR::~FixIGAR()
{
  delete [] infile;
  //  delete [] outfile;
  
  memory->destroy(figar);
  memory->destroy(eigar);

  if (!deallocate_flag) FixIGAR::deallocate_grid();
}

/* ---------------------------------------------------------------------- */

void FixIGAR::post_constructor()
{
  // allocate global grid on each proc
  // needs to be done in post_contructor() beccause is virtual method

  allocate_grid();

  // initialize electron temperatures on grid

  int ix,iy,iz;
  for (iz = 0; iz < nzgrid; iz++)
    for (iy = 0; iy < nygrid; iy++)
      for (ix = 0; ix < nxgrid; ix++)
        U_igar[iz][iy][ix] = tinit;
  
  for (int i = 0; i < atom->nmax; i++) {
    eigar[i] = 0.0;
  }
  vector_atom = eigar;

  // zero igar_energy_transfer_all
  // in case compute_vector accesses it on timestep 0

  outflag = 0;
  memset(&igar_energy_transfer_all[0][0][0],0,ngridtotal*sizeof(double));

  // set initial electron temperatures from user input file

  if (infile) read_igar_energies(infile);
}

/* ---------------------------------------------------------------------- */

int FixIGAR::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixIGAR::init()
{
  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use fix igar with 2d simulation");
  if (domain->nonperiodic != 0)
    error->all(FLERR,"Cannot use non-periodic boundares with fix igar");
  if (domain->triclinic)
    error->all(FLERR,"Cannot use fix igar with triclinic box");
  std::cout << "INIT DONE" << std::endl;  
}

/* ---------------------------------------------------------------------- */

void FixIGAR::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style,"^verlet"))
    post_force_setup(vflag);
}

/* ---------------------------------------------------------------------- */

void FixIGAR::post_force_setup(int /*vflag*/)
{
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  
  int ix,iy,iz;
  
  double dx = domain->xprd/nxgrid;
  double dy = domain->yprd/nygrid;
  double dz = domain->zprd/nzgrid;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      f[i][0] += figar[i][0];
      f[i][1] += figar[i][1];
      f[i][2] += figar[i][2];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixIGAR::post_force(int /*vflag*/)
{
  int ix,iy,iz;
  //  double gamma1,gamma2;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double *boxlo = domain->boxlo;
  double dxinv = nxgrid/domain->xprd;
  double dyinv = nygrid/domain->yprd;
  double dzinv = nzgrid/domain->zprd;

  double dx = 1./dxinv;
  double dy = 1./dyinv;
  double dz = 1./dzinv;
  
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      ix = static_cast<int> ((x[i][0]-boxlo[0])*dxinv + shift) - OFFSET;
      iy = static_cast<int> ((x[i][1]-boxlo[1])*dyinv + shift) - OFFSET;
      iz = static_cast<int> ((x[i][2]-boxlo[2])*dzinv + shift) - OFFSET;
      if (ix < 0) ix += nxgrid;
      if (iy < 0) iy += nygrid;
      if (iz < 0) iz += nzgrid;
      if (ix >= nxgrid) ix -= nxgrid;
      if (iy >= nygrid) iy -= nygrid;
      if (iz >= nzgrid) iz -= nzgrid;

      if (U_igar[iz][iy][ix] < 0)
        error->one(FLERR,"U_igar dropped below zero");

      double pxc = x[i][0];
      double pyc = x[i][1];
      double pzc = x[i][2];

      double dl = 0.05;

      double pxcdx = pxc+dl;
      double pycdy = pyc+dl;
      double pzcdz = pzc+dl;
      
      double EEE  = kIm * interpolation_igar_grid(pxc,   pyc,   pzc  );      
      double EEEx = kIm * interpolation_igar_grid(pxcdx, pyc,   pzc  );
      double EEEy = kIm * interpolation_igar_grid(pxc,   pycdy, pzc  );
      double EEEz = kIm * interpolation_igar_grid(pxc,   pyc,   pzcdz);

      double fxigar = -1.0 * (EEEx-EEE) / dl;
      double fyigar = -1.0 * (EEEy-EEE) / dl;
      double fzigar = -1.0 * (EEEz-EEE) / dl;

      eigar[i] = EEE;
      figar[i][0] = fxigar;
      figar[i][1] = fyigar;
      figar[i][2] = fzigar;

      f[i][0] += figar[i][0];
      f[i][1] += figar[i][1];
      f[i][2] += figar[i][2];      
    }
  }
  vector_atom = eigar;
  
}

/* ---------------------------------------------------------------------- */

// void FixIGAR::post_force_respa_setup(int vflag, int ilevel, int /*iloop*/)
// {
//   if (ilevel == nlevels_respa-1) post_force_setup(vflag);
// }

// /* ---------------------------------------------------------------------- */

// void FixIGAR::post_force_respa(int vflag, int ilevel, int /*iloop*/)
// {
//   if (ilevel == nlevels_respa-1) post_force(vflag);
// }

/* ---------------------------------------------------------------------- */

double FixIGAR::interpolation_igar_grid(double pos_x, double pos_y, double pos_z)
{
  double Eval = 0.0;

  double *boxlo = domain->boxlo;
  double dxinv = nxgrid/domain->xprd;
  double dyinv = nygrid/domain->yprd;
  double dzinv = nzgrid/domain->zprd;  
  
  int ind000[3];
  int ind100[3];
  int ind010[3];
  int ind001[3];
  
  int ind110[3];
  int ind101[3];
  int ind011[3];
  
  int ind111[3];

  int ix0=0;
  int iy0=0;
  int iz0=0;
  int ix=0;
  int iy=0;
  int iz=0;
  
  ix0 = static_cast<int> ((pos_x-boxlo[0])*dxinv + shift) - OFFSET;
  iy0 = static_cast<int> ((pos_y-boxlo[1])*dyinv + shift) - OFFSET;
  iz0 = static_cast<int> ((pos_z-boxlo[2])*dzinv + shift) - OFFSET;
  
  if (ix0 < 0) ix0 += nxgrid;
  if (iy0 < 0) iy0 += nygrid;
  if (iz0 < 0) iz0 += nzgrid;
  if (ix0 >= nxgrid) ix0 -= nxgrid;
  if (iy0 >= nygrid) iy0 -= nygrid;
  if (iz0 >= nzgrid) iz0 -= nzgrid;  
  ind000[0] = ix0;
  ind000[1] = iy0;
  ind000[2] = iz0;

  ix = ix0+1;
  iy = iy0;
  iz = iz0;
  if (ix < 0) ix += nxgrid;
  if (iy < 0) iy += nygrid;
  if (iz < 0) iz += nzgrid;
  if (ix >= nxgrid) ix -= nxgrid;
  if (iy >= nygrid) iy -= nygrid;
  if (iz >= nzgrid) iz -= nzgrid;  
  ind100[0] = ix;
  ind100[1] = iy;
  ind100[2] = iz;

  ix = ix0+1;
  iy = iy0+1;
  iz = iz0;
  if (ix < 0) ix += nxgrid;
  if (iy < 0) iy += nygrid;
  if (iz < 0) iz += nzgrid;
  if (ix >= nxgrid) ix -= nxgrid;
  if (iy >= nygrid) iy -= nygrid;
  if (iz >= nzgrid) iz -= nzgrid;    
  ind110[0] = ix;
  ind110[1] = iy;
  ind110[2] = iz;

  ix = ix0;
  iy = iy0+1;
  iz = iz0;
  if (ix < 0) ix += nxgrid;
  if (iy < 0) iy += nygrid;
  if (iz < 0) iz += nzgrid;
  if (ix >= nxgrid) ix -= nxgrid;
  if (iy >= nygrid) iy -= nygrid;
  if (iz >= nzgrid) iz -= nzgrid;    
  ind010[0] = ix;
  ind010[1] = iy;
  ind010[2] = iz;
  
  ix = ix0;
  iy = iy0;
  iz = iz0+1;
  if (ix < 0) ix += nxgrid;
  if (iy < 0) iy += nygrid;
  if (iz < 0) iz += nzgrid;
  if (ix >= nxgrid) ix -= nxgrid;
  if (iy >= nygrid) iy -= nygrid;
  if (iz >= nzgrid) iz -= nzgrid;  
  ind001[0] = ix;
  ind001[1] = iy;
  ind001[2] = iz;
    
  ix = ix0+1;
  iy = iy0;
  iz = iz0+1;
  if (ix < 0) ix += nxgrid;
  if (iy < 0) iy += nygrid;
  if (iz < 0) iz += nzgrid;
  if (ix >= nxgrid) ix -= nxgrid;
  if (iy >= nygrid) iy -= nygrid;
  if (iz >= nzgrid) iz -= nzgrid;  
  ind101[0] = ix;
  ind101[1] = iy;
  ind101[2] = iz;
    
  ix = ix0+1;
  iy = iy0+1;
  iz = iz0+1;
  if (ix < 0) ix += nxgrid;
  if (iy < 0) iy += nygrid;
  if (iz < 0) iz += nzgrid;
  if (ix >= nxgrid) ix -= nxgrid;
  if (iy >= nygrid) iy -= nygrid;
  if (iz >= nzgrid) iz -= nzgrid;  
  ind111[0] = ix;
  ind111[1] = iy;
  ind111[2] = iz;
    
  ix = ix0;
  iy = iy0+1;
  iz = iz0+1;
  if (ix < 0) ix += nxgrid;
  if (iy < 0) iy += nygrid;
  if (iz < 0) iz += nzgrid;
  if (ix >= nxgrid) ix -= nxgrid;
  if (iy >= nygrid) iy -= nygrid;
  if (iz >= nzgrid) iz -= nzgrid;  
  ind011[0] = ix;
  ind011[1] = iy;
  ind011[2] = iz;

  double en000,en100,en110,en010,en001,en101,en111,en011;

  en000 = U_igar[ind000[2]][ind000[1]][ind000[0]];
  en100 = U_igar[ind100[2]][ind100[1]][ind100[0]];
  en010 = U_igar[ind010[2]][ind010[1]][ind010[0]];
  en110 = U_igar[ind110[2]][ind110[1]][ind110[0]];
  en001 = U_igar[ind001[2]][ind001[1]][ind001[0]];
  en101 = U_igar[ind101[2]][ind101[1]][ind101[0]];
  en011 = U_igar[ind011[2]][ind011[1]][ind011[0]];
  en111 = U_igar[ind111[2]][ind111[1]][ind111[0]];

  double xl,yl,zl;

  double pos_grid_x = ((pos_x-boxlo[0])*dxinv + shift) - OFFSET;
  double pos_grid_y = ((pos_y-boxlo[1])*dyinv + shift) - OFFSET;
  double pos_grid_z = ((pos_z-boxlo[2])*dzinv + shift) - OFFSET;
  
  xl = pos_grid_x - floor(pos_grid_x);
  yl = pos_grid_y - floor(pos_grid_y);
  zl = pos_grid_z - floor(pos_grid_z);

  Eval = en000*(1.0-xl)*(1.0-yl)*(1.0-zl) + en100*xl*(1.0-yl)*(1.0-zl) + en010*(1.0-xl)*yl*(1.0-zl) + en001*(1.0-xl)*(1.0-yl)*zl + en101*xl*(1.0-yl)*zl + en011*(1.0-xl)*yl*zl + en110*xl*yl*(1.0-zl) + en111*xl*yl*zl;  
  return Eval;
}

void FixIGAR::end_of_step()
{
  int ix,iy,iz;

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double *boxlo = domain->boxlo;
  double dxinv = nxgrid/domain->xprd;
  double dyinv = nygrid/domain->yprd;
  double dzinv = nzgrid/domain->zprd;

  for (iz = 0; iz < nzgrid; iz++)
    for (iy = 0; iy < nygrid; iy++)
      for (ix = 0; ix < nxgrid; ix++)
        igar_energy_transfer[iz][iy][ix] = 0.0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      ix = static_cast<int> ((x[i][0]-boxlo[0])*dxinv + shift) - OFFSET;
      iy = static_cast<int> ((x[i][1]-boxlo[1])*dyinv + shift) - OFFSET;
      iz = static_cast<int> ((x[i][2]-boxlo[2])*dzinv + shift) - OFFSET;
      if (ix < 0) ix += nxgrid;
      if (iy < 0) iy += nygrid;
      if (iz < 0) iz += nzgrid;
      if (ix >= nxgrid) ix -= nxgrid;
      if (iy >= nygrid) iy -= nygrid;
      if (iz >= nzgrid) iz -= nzgrid;

      igar_energy_transfer[iz][iy][ix] += eigar[i];
    }
  
  outflag = 0;
  MPI_Allreduce(&igar_energy_transfer[0][0][0],&igar_energy_transfer_all[0][0][0],
                ngridtotal,MPI_DOUBLE,MPI_SUM,world);

  // if (outfile && (update->ntimestep % outevery == 0))
  //   write_igar_energies(fmt::format("{}.{}",outfile,update->ntimestep));
}

/* ----------------------------------------------------------------------
   read in initial electron temperatures from a user-specified file
   only read by proc 0, grid values are Bcast to other procs
------------------------------------------------------------------------- */

void FixIGAR::read_igar_energies(const std::string &filename)
{
  if (comm->me == 0) {

    int ***U_igar_initial_set;
    memory->create(U_igar_initial_set,nzgrid,nygrid,nxgrid,"igar:U_igar_set");
    memset(&U_igar_initial_set[0][0][0],0,ngridtotal*sizeof(int));

    // read initial electron temperature values from file
    bigint nread = 0;

    try {
      PotentialFileReader reader(lmp, filename, "igar energy grid");

      while (nread < ngridtotal) {
        // reader will skip over comment-only lines
        auto values = reader.next_values(4);
        ++nread;

        int ix = values.next_int();// - 1;
        int iy = values.next_int();// - 1;
        int iz = values.next_int();// - 1;
        double U_igar_tmp  = values.next_double();

        // check correctness of input data

        if ((ix < 0) || (ix >= nxgrid) || (iy < 0) || (iy >= nygrid) || (iz < 0) || (iz >= nzgrid))
          throw TokenizerException("Fix igar invalid grid index in fix igar grid file","");

        if (U_igar_tmp < 0.0)
          throw TokenizerException("Fix igar electron temperatures must be > 0.0","");

        U_igar[iz][iy][ix] = U_igar_tmp;
        U_igar_initial_set[iz][iy][ix] = 1;
      }
    } catch (std::exception &e) {
      error->one(FLERR, e.what());
    }

    // check completeness of input data

    for (int iz = 0; iz < nzgrid; iz++)
      for (int iy = 0; iy < nygrid; iy++)
        for (int ix = 0; ix < nxgrid; ix++)
          if (U_igar_initial_set[iz][iy][ix] == 0)
            error->all(FLERR,"Fix igar infile did not set all temperatures");

    memory->destroy(U_igar_initial_set);
  }

  MPI_Bcast(&U_igar[0][0][0],ngridtotal,MPI_DOUBLE,0,world);
}

/* ----------------------------------------------------------------------
   write out current electron temperatures to user-specified file
   only written by proc 0
------------------------------------------------------------------------- */

// void FixIGAR::write_igar_energies(const std::string &filename)
// {
//   if (comm->me) return;

//   FILE *fp = fopen(filename.c_str(),"w");
//   if (!fp) error->one(FLERR,"Fix igar could not open output file {}: {}",
//                       filename,utils::getsyserror());
//   fmt::print(fp,"# DATE: {} UNITS: {} COMMENT: Electron temperature on "
//              "{}x{}x{} grid at step {} - created by fix {}\n", utils::current_date(),
//              update->unit_style, nxgrid, nygrid, nzgrid, update->ntimestep, style);

//   int ix,iy,iz;

//   for (iz = 0; iz < nzgrid; iz++)
//     for (iy = 0; iy < nygrid; iy++)
//       for (ix = 0; ix < nxgrid; ix++)
//         fprintf(fp,"%d %d %d %20.16g\n",ix+1,iy+1,iz+1,U_igar[iz][iy][ix]);

//   fclose(fp);
// }

// /* ---------------------------------------------------------------------- */

void FixIGAR::grow_arrays(int ngrow)
{
  memory->grow(figar,ngrow,3,"igar:figar");
  memory->grow(eigar,ngrow,"igar:peratom_et");  
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixIGAR::write_restart(FILE *fp)
{
  double *rlist;
  memory->create(rlist,nxgrid*nygrid*nzgrid+4,"igar:rlist");

  int n = 0;
  rlist[n++] = nxgrid;
  rlist[n++] = nygrid;
  rlist[n++] = nzgrid;
  rlist[n++] = seed;

  // store global grid values

  for (int iz = 0; iz < nzgrid; iz++)
    for (int iy = 0; iy < nygrid; iy++)
      for (int ix = 0; ix < nxgrid; ix++)
        rlist[n++] =  U_igar[iz][iy][ix];

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(rlist,sizeof(double),n,fp);
  }

  memory->destroy(rlist);
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixIGAR::restart(char *buf)
{
  int n = 0;
  auto rlist = (double *) buf;

  // check that restart grid size is same as current grid size

  int nxgrid_old = static_cast<int> (rlist[n++]);
  int nygrid_old = static_cast<int> (rlist[n++]);
  int nzgrid_old = static_cast<int> (rlist[n++]);

  if (nxgrid_old != nxgrid || nygrid_old != nygrid || nzgrid_old != nzgrid)
    error->all(FLERR,"Must restart fix igar with same grid size");

  // change RN seed from initial seed, to avoid same Igar factors
  // just increment by 1, since for RanMars that is a new RN stream

  seed = static_cast<int> (rlist[n++]) + 1;
  delete random;
  random = new RanMars(lmp,seed+comm->me);

  // restore global grid values

  for (int iz = 0; iz < nzgrid; iz++)
    for (int iy = 0; iy < nygrid; iy++)
      for (int ix = 0; ix < nxgrid; ix++)
        U_igar[iz][iy][ix] = rlist[n++];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixIGAR::pack_restart(int i, double *buf)
{
  // pack buf[0] this way because other fixes unpack it

  buf[0] = 4;
  buf[1] = figar[i][0];
  buf[2] = figar[i][1];
  buf[3] = figar[i][2];
  return 4;
}

/* ----------------------------------------------------------------------
   unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixIGAR::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to Nth set of extra values
  // unpack the Nth first values this way because other fixes pack them

  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  figar[nlocal][0] = extra[nlocal][m++];
  figar[nlocal][1] = extra[nlocal][m++];
  figar[nlocal][2] = extra[nlocal][m++];
}

/* ----------------------------------------------------------------------
   size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixIGAR::size_restart(int /*nlocal*/)
{
  return 4;
}

/* ----------------------------------------------------------------------
   maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixIGAR::maxsize_restart()
{
  return 4;
}

/* ----------------------------------------------------------------------
   return the energy of the electronic subsystem or the igar_energy transfer
   between the subsystems
------------------------------------------------------------------------- */

double FixIGAR::compute_scalar()
{
  igar_energy = 0.0;
  
  int ix,iy,iz;
  
  double dx = domain->xprd/nxgrid;
  double dy = domain->yprd/nygrid;
  double dz = domain->zprd/nzgrid;
  
  for (iz = 0; iz < nzgrid; iz++) {
    for (iy = 0; iy < nygrid; iy++) {
      for (ix = 0; ix < nxgrid; ix++) {
        igar_energy -= igar_energy_transfer_all[iz][iy][ix];
      }
    }
  }
  return igar_energy;
}

/* ----------------------------------------------------------------------
   memory usage for figar and 3d grids
------------------------------------------------------------------------- */

double FixIGAR::memory_usage()
{
  double bytes = 0.0;
  bytes += (double) atom->nmax * 3 * sizeof(double);
  bytes += (double) atom->nmax * 1 * sizeof(double);
  //  bytes += (double) 4*ngridtotal * sizeof(int);
  bytes += (double) 3*ngridtotal * sizeof(int);  
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate 3d grid quantities
------------------------------------------------------------------------- */

void FixIGAR::allocate_grid()
{
  //  memory->create(U_igar_old,nzgrid,nygrid,nxgrid,"igar:U_igar_old");
  memory->create(U_igar,nzgrid,nygrid,nxgrid,"igar:U_igar");  
  memory->create(igar_energy_transfer,nzgrid,nygrid,nxgrid,"igar:igar_energy_transfer");
  memory->create(igar_energy_transfer_all,nzgrid,nygrid,nxgrid,"igar:igar_energy_transfer_all");
}

/* ----------------------------------------------------------------------
   deallocate 3d grid quantities
------------------------------------------------------------------------- */

void FixIGAR::deallocate_grid()
{
  //  memory->destroy(U_igar_old);
  memory->destroy(U_igar);  
  memory->destroy(igar_energy_transfer);
  memory->destroy(igar_energy_transfer_all);
}
