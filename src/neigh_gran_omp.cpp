/* ----------------------------------------------------------------------
   LIGGGHTS - LAMMPS Improved for General Granular and Granular Heat
   Transfer Simulations

   LIGGGHTS is part of the CFDEMproject
   www.liggghts.com | www.cfdem.com

   This file was modified with respect to the release in LAMMPS
   Modifications are Copyright 2009-2012 JKU Linz
                     Copyright 2012-2014 DCS Computing GmbH, Linz
                     Copyright 2013-     JKU Linz

   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors:
   Christoph Kloss (JKU Linz, DCS Computing GmbH, Linz)
   Richard Berger (JKU Linz)
------------------------------------------------------------------------- */

#include "neighbor.h"
#include "neigh_list.h"
#include "atom.h"
#include "comm.h"
#include "group.h"
#include "fix_contact_history.h"
#include "error.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   granular particles
   N^2 / 2 search for neighbor pairs with partial Newton's 3rd law
   shear history must be accounted for when a neighbor pair is added
   pair added to list if atoms i and j are both owned and i < j
   pair added if j is ghost (also stored by proc owning j)
------------------------------------------------------------------------- */

void Neighbor::granular_nsq_no_newton_omp(NeighList *list)
{
  const int nlocal = (includegroup) ? atom->nfirst : atom->nlocal;
  const int bitmask = (includegroup) ? group->bitmask[includegroup] : 0;

  FixContactHistory * const fix_history = list->fix_history;
  NeighList * listgranhistory = list->listgranhistory;

  const int nthreads = comm->nthreads;

#if defined(_OPENMP)
#pragma omp parallel default(none) shared(list,listgranhistory)
#endif
{
#if defined(_OPENMP)
  const int tid = omp_get_thread_num();
  const int idelta = 1 + nlocal/nthreads;
  const int ifrom = tid*idelta;
  const int imax  = ifrom + idelta;
  const int ito = (imax > nlocal) ? nlocal : imax;
#else
  const int tid = 0;
  const int ifrom = 0;
  const int ito = nlocal;
#endif

  int i,j,m,n,nn,d;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  double radi,radsum,cutsq;
  int *neighptr,*touchptr;
  double *shearptr;

  int *npartner,**partner;
  double **contacthistory;
  int **firsttouch;
  double **firstshear;
  MyPage<int> *ipage_touch;
  MyPage<double> *dpage_shear;
  int dnum;

  double **x = atom->x;
  double *radius = atom->radius;
  int *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int nall = atom->nlocal + atom->nghost;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // each thread has its own page allocator
  MyPage<int> &ipage = list->ipage[tid];
  ipage.reset();

  if (fix_history) {
    npartner = fix_history->npartner_;
    partner = fix_history->partner_;
    contacthistory = fix_history->contacthistory_;
    listgranhistory = list->listgranhistory;
    firsttouch = listgranhistory->firstneigh;
    firstshear = listgranhistory->firstdouble;
    ipage_touch = listgranhistory->ipage+tid;
    dpage_shear = listgranhistory->dpage+tid;
    ipage_touch->reset();
    dpage_shear->reset();
    dnum = listgranhistory->dnum;
  }

  for (i = ifrom; i < ito; i++) {
    n = 0;
    neighptr = ipage.vget();
    if (fix_history) {
      nn = 0;
      touchptr = ipage_touch->vget();
      shearptr = dpage_shear->vget();
    }

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    radi = radius[i];

    // loop over remaining atoms, owned and ghost

    for (j = 0; j < nall; j++) {
      if(j <= i && atom->same_thread(i, j)) continue; // use partial newton if on same thread
      if(i == j) continue;
      if (includegroup && !(mask[j] & bitmask)) continue;
      if (exclude && exclusion(i,j,type[i],type[j],mask,molecule)) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      radsum = (radi + radius[j]) * contactDistanceFactor;
      cutsq = (radsum+skin) * (radsum+skin);

      if (rsq <= cutsq) {
        neighptr[n] = j;

        if (fix_history) {
          if (rsq < radsum*radsum)
          {
            for (m = 0; m < npartner[i]; m++)
              if (partner[i][m] == tag[j]) break;
            if (m < npartner[i]) {
              touchptr[n] = 1;
              for (d = 0; d < dnum; d++) {
                shearptr[nn++] = contacthistory[i][m*dnum+d];
              }
            } else {
              touchptr[n] = 0;
              for (d = 0; d < dnum; d++) {
                shearptr[nn++] = 0.0;
              }
            }
          } else {
            touchptr[n] = 0;
            for (d = 0; d < dnum; d++) {
              shearptr[nn++] = 0.0;
            }
          }
        }

        n++;
      }
    }

    ilist[i] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage.vgot(n);
    if (ipage.status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");

    if (fix_history) {
      firsttouch[i] = touchptr;
      firstshear[i] = shearptr;
      ipage_touch->vgot(n);
      dpage_shear->vgot(nn);
    }
  }
}

  list->inum = nlocal;
}

/* ----------------------------------------------------------------------
   granular particles
   N^2 / 2 search for neighbor pairs with full Newton's 3rd law
   no shear history is allowed for this option
   pair added to list if atoms i and j are both owned and i < j
   if j is ghost only me or other proc adds pair
   decision based on itag,jtag tests
------------------------------------------------------------------------- */

void Neighbor::granular_nsq_newton_omp(NeighList *list)
{
  const int nlocal = (includegroup) ? atom->nfirst : atom->nlocal;
  const int bitmask = (includegroup) ? group->bitmask[includegroup] : 0;

  const int nthreads = comm->nthreads;

#if defined(_OPENMP)
#pragma omp parallel default(none) shared(list)
#endif
{
#if defined(_OPENMP)
  const int tid = omp_get_thread_num();
  const int idelta = 1 + nlocal/nthreads;
  const int ifrom = tid*idelta;
  const int ito   = ((ifrom + idelta) > nlocal) ? nlocal : (ifrom+idelta);
#else
  const int tid = 0;
  const int ifrom = 0;
  const int ito = nlocal;
#endif

  int i,j,n,itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  double radi,radsum,cutsq;
  int *neighptr;

  double **x = atom->x;
  double *radius = atom->radius;
  int *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int nall = atom->nlocal + atom->nghost;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // each thread has its own page allocator
  MyPage<int> &ipage = list->ipage[tid];
  ipage.reset();

  for (i = ifrom; i < ito; i++) {

    n = 0;
    neighptr = ipage.vget();

    itag = tag[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    radi = radius[i];

    // loop over remaining atoms, owned and ghost

    for (j = i+1; j < nall; j++) {
      if (includegroup && !(mask[j] & bitmask)) continue;

      if (j >= nlocal) {
        jtag = tag[j];
        if (itag > jtag) {
          if ((itag+jtag) % 2 == 0) continue;
        } else if (itag < jtag) {
          if ((itag+jtag) % 2 == 1) continue;
        } else {
          if (x[j][2] < ztmp) continue;
          if (x[j][2] == ztmp) {
            if (x[j][1] < ytmp) continue;
            if (x[j][1] == ytmp && x[j][0] < xtmp) continue;
          }
        }
      }

      if (exclude && exclusion(i,j,type[i],type[j],mask,molecule)) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      radsum = (radi + radius[j]) * contactDistanceFactor;
      cutsq = (radsum+skin) * (radsum+skin);

      if (rsq <= cutsq) neighptr[n++] = j;
    }

    ilist[i] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage.vgot(n);
    if (ipage.status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
  }
}
  list->inum = nlocal;
}

/* ----------------------------------------------------------------------
   granular particles
   binned neighbor list construction with partial Newton's 3rd law
   shear history must be accounted for when a neighbor pair is added
   each owned atom i checks own bin and surrounding bins in non-Newton stencil
   pair stored once if i,j are both owned and i < j
   pair stored by me if j is ghost (also stored by proc owning j)
------------------------------------------------------------------------- */

void Neighbor::granular_bin_no_newton_omp(NeighList *list)
{
  // bin local & ghost atoms

  bin_atoms();

  const int nlocal = (includegroup) ? atom->nfirst : atom->nlocal;

  FixContactHistory * const fix_history = list->fix_history;
  NeighList * listgranhistory = list->listgranhistory;

  const int nthreads = comm->nthreads;

#if defined(_OPENMP)
#pragma omp parallel default(none) shared(list,listgranhistory)
#endif
{
#if defined(_OPENMP)
  const int tid = omp_get_thread_num();
  const int idelta = 1 + nlocal/nthreads;
  const int ifrom = tid*idelta;
  const int ito   = ((ifrom + idelta) > nlocal) ? nlocal : (ifrom+idelta);
#else
  const int tid = 0;
  const int ifrom = 0;
  const int ito = nlocal;
#endif

  int i,j,k,m,n,nn,ibin,d;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  double radi,radsum,cutsq;
  int *neighptr,*touchptr;
  double *shearptr;
  MyPage<int> *ipage_touch;
  MyPage<double> *dpage_shear;

  int *npartner,**partner;
  double **contacthistory;
  int **firsttouch;
  double **firstshear;
  int dnum;

  // loop over each atom, storing neighbors

  double **x = atom->x;
  double *radius = atom->radius;
  int *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  int *molecule = atom->molecule;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int nstencil = list->nstencil;
  int *stencil = list->stencil;

  // each thread has its own page allocator
  MyPage<int> &ipage = list->ipage[tid];
  ipage.reset();

  if (fix_history) {
    npartner = fix_history->npartner_;
    partner = fix_history->partner_;
    contacthistory = fix_history->contacthistory_;
    firsttouch = listgranhistory->firstneigh;
    firstshear = listgranhistory->firstdouble;
    ipage_touch = listgranhistory->ipage+tid;
    dpage_shear = listgranhistory->dpage+tid;
    ipage_touch->reset();
    dpage_shear->reset();
    dnum = listgranhistory->dnum;
  }

  for (i = ifrom; i < ito; i++) {

    n = 0;
    neighptr = ipage.vget();
    if (fix_history) {
      nn = 0;
      touchptr = ipage_touch->vget();
      shearptr = dpage_shear->vget();
    }

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    radi = radius[i];
    ibin = coord2bin(x[i]);

    // loop over all atoms in surrounding bins in stencil including self
    // only store pair if i < j
    // stores own/own pairs only once
    // stores own/ghost pairs on both procs

    for (k = 0; k < nstencil; k++) {
      for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
        if(j <= i && atom->same_thread(i, j)) continue; // use partial newton if on same thread
        if(i == j) continue;
        if (exclude && exclusion(i,j,type[i],type[j],mask,molecule)) continue;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        radsum = (radi + radius[j]) * contactDistanceFactor;
        cutsq = (radsum+skin) * (radsum+skin);

        if (rsq <= cutsq) {
          neighptr[n] = j;
          if (fix_history) {
            if (rsq < radsum*radsum)
                {
              for (m = 0; m < npartner[i]; m++)
                if (partner[i][m] == tag[j]) break;
              if (m < npartner[i]) {
                touchptr[n] = 1;
                for (d = 0; d < dnum; d++) {
                  shearptr[nn++] = contacthistory[i][m*dnum+d];
                }
              } else {
                 touchptr[n] = 0;
                 for (d = 0; d < dnum; d++) {
                   shearptr[nn++] = 0.0;
                 }
              }
            } else {
              touchptr[n] = 0;
              for (d = 0; d < dnum; d++) {
                shearptr[nn++] = 0.0;
              }
            }
          }

          n++;
        }
      }
    }

    ilist[i] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage.vgot(n);
    if (ipage.status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");

    if (fix_history) {
      firsttouch[i] = touchptr;
      firstshear[i] = shearptr;
      ipage_touch->vgot(n);
      dpage_shear->vgot(nn);
    }
  }
}
  list->inum = nlocal;
}

/* ----------------------------------------------------------------------
   granular particles
   binned neighbor list construction with full Newton's 3rd law
   no shear history is allowed for this option
   each owned atom i checks its own bin and other bins in Newton stencil
   every pair stored exactly once by some processor
------------------------------------------------------------------------- */

void Neighbor::granular_bin_newton_omp(NeighList *list)
{
  // bin local & ghost atoms

  bin_atoms();

  const int nlocal = (includegroup) ? atom->nfirst : atom->nlocal;

  const int nthreads = comm->nthreads;

#if defined(_OPENMP)
#pragma omp parallel default(none) shared(list)
#endif
{
#if defined(_OPENMP)
  const int tid = omp_get_thread_num();
  const int idelta = 1 + nlocal/nthreads;
  const int ifrom = tid*idelta;
  const int ito   = ((ifrom + idelta) > nlocal) ? nlocal : (ifrom+idelta);
#else
  const int tid = 0;
  const int ifrom = 0;
  const int ito = nlocal;
#endif

  int i,j,k,n,ibin;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  double radi,radsum,cutsq;
  int *neighptr;

  // loop over each atom, storing neighbors

  double **x = atom->x;
  double *radius = atom->radius;
  int *type = atom->type;
  int *mask = atom->mask;
  int *molecule = atom->molecule;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int nstencil = list->nstencil;
  int *stencil = list->stencil;

  // each thread has its own page allocator
  MyPage<int> &ipage = list->ipage[tid];
  ipage.reset();

  for (i = ifrom; i < ito; i++) {
    n = 0;
    neighptr = ipage.vget();

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    radi = radius[i];

    // loop over rest of atoms in i's bin, ghosts are at end of linked list
    // if j is owned atom, store it, since j is beyond i in linked list
    // if j is ghost, only store if j coords are "above and to the right" of i

    for (j = bins[i]; j >= 0; j = bins[j]) {
      if (j >= nlocal) {
        if (x[j][2] < ztmp) continue;
        if (x[j][2] == ztmp) {
          if (x[j][1] < ytmp) continue;
          if (x[j][1] == ytmp && x[j][0] < xtmp) continue;
        }
      }

      if (exclude && exclusion(i,j,type[i],type[j],mask,molecule)) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      radsum = radi + radius[j];
      cutsq = (radsum+skin) * (radsum+skin);

      if (rsq <= cutsq) neighptr[n++] = j;
    }

    // loop over all atoms in other bins in stencil, store every pair

    ibin = coord2bin(x[i]);
    for (k = 0; k < nstencil; k++) {
      for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
        if (exclude && exclusion(i,j,type[i],type[j],mask,molecule)) continue;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        radsum = radi + radius[j];
        cutsq = (radsum+skin) * (radsum+skin);

        if (rsq <= cutsq) neighptr[n++] = j;
      }
    }

    ilist[i] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage.vgot(n);
    if (ipage.status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
  }
}
  list->inum = nlocal;
}

/* ----------------------------------------------------------------------
   granular particles
   binned neighbor list construction with Newton's 3rd law for triclinic
   no shear history is allowed for this option
   each owned atom i checks its own bin and other bins in triclinic stencil
   every pair stored exactly once by some processor
------------------------------------------------------------------------- */

void Neighbor::granular_bin_newton_tri_omp(NeighList *list)
{
  // bin local & ghost atoms

  bin_atoms();

  const int nlocal = (includegroup) ? atom->nfirst : atom->nlocal;
  const int nthreads = comm->nthreads;

#if defined(_OPENMP)
#pragma omp parallel default(none) shared(list)
#endif
{
#if defined(_OPENMP)
  const int tid = omp_get_thread_num();
  const int idelta = 1 + nlocal/nthreads;
  const int ifrom = tid*idelta;
  const int ito   = ((ifrom + idelta) > nlocal) ? nlocal : (ifrom+idelta);
#else
  const int tid = 0;
  const int ifrom = 0;
  const int ito = nlocal;
#endif

  int i,j,k,n,ibin;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  double radi,radsum,cutsq;
  int *neighptr;

  // loop over each atom, storing neighbors

  double **x = atom->x;
  double *radius = atom->radius;
  int *type = atom->type;
  int *mask = atom->mask;
  int *molecule = atom->molecule;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int nstencil = list->nstencil;
  int *stencil = list->stencil;

  // each thread has its own page allocator
  MyPage<int> &ipage = list->ipage[tid];
  ipage.reset();

  for (i = ifrom; i < ito; i++) {

    n = 0;
    neighptr = ipage.vget();

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    radi = radius[i];

    // loop over all atoms in bins in stencil
    // pairs for atoms j "below" i are excluded
    // below = lower z or (equal z and lower y) or (equal zy and lower x)
    //         (equal zyx and j <= i)
    // latter excludes self-self interaction but allows superposed atoms

    ibin = coord2bin(x[i]);
    for (k = 0; k < nstencil; k++) {
      for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
        if (x[j][2] < ztmp) continue;
        if (x[j][2] == ztmp) {
          if (x[j][1] < ytmp) continue;
          if (x[j][1] == ytmp) {
            if (x[j][0] < xtmp) continue;
            if (x[j][0] == xtmp && j <= i) continue;
          }
        }

        if (exclude && exclusion(i,j,type[i],type[j],mask,molecule)) continue;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        radsum = (radi + radius[j]) * contactDistanceFactor;
        cutsq = (radsum+skin) * (radsum+skin);

        if (rsq <= cutsq) neighptr[n++] = j;
      }
    }

    ilist[i] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage.vgot(n);
    if (ipage.status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
  }
}
  list->inum = nlocal;
}
