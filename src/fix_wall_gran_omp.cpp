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
   Philippe Seil (JKU Linz)
   Richard Berger (JKU Linz)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fix_wall_gran_omp.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "force.h"
#include "pair_gran.h"
#include "fix_rigid.h"
#include "fix_mesh.h"
#include "fix_contact_history.h"
#include "modify.h"
#include "respa.h"
#include "memory.h"
#include "comm.h"
#include "error.h"
#include "fix_property_atom.h"
#include "fix_contact_property_atom_wall.h"
#include "math_extra.h"
#include "math_extra_liggghts.h"
#include "compute_pair_gran_local.h"
#include "fix_neighlist_mesh_omp.h"
#include "fix_mesh_surface_stress.h"
#include "tri_mesh.h"
#include "primitive_wall.h"
#include "primitive_wall_definitions.h"
#include "mpi_liggghts.h"
#include "neighbor.h"
#include "thr_omp.h"
#include "contact_interface.h"
#include "fix_property_global.h"
#include "mpi.h"
#include <vector>
#include "granular_wall.h"
#define NDEBUG
#include <assert.h>

#include <omp.h>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace LAMMPS_NS::PRIMITIVE_WALL_DEFINITIONS;
using namespace LIGGGHTS::Walls;
using namespace LIGGGHTS::ContactModels;

const double SMALL = 1e-12;

/* ---------------------------------------------------------------------- */

FixWallGranOMP::FixWallGranOMP(LAMMPS *lmp, int narg, char **arg) : FixWallGran(lmp, narg, arg)
{
}

/* ---------------------------------------------------------------------- */

FixWallGranOMP::~FixWallGranOMP()
{
}

/* ---------------------------------------------------------------------- */

void FixWallGranOMP::setup(int vflag)
{
    if (strstr(update->integrate_style,"verlet"))
    {
      pre_neighbor();
      pre_force(vflag);
      post_force(vflag);
    }
    else
    {
      ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa_-1);
      post_force_respa(vflag,nlevels_respa_-1,0);
      ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa_-1);
    }

    init_heattransfer();
}

/* ----------------------------------------------------------------------
   neighbor list via fix wall/gran is only relevant for primitive walls
------------------------------------------------------------------------- */

void FixWallGranOMP::pre_neighbor()
{
    rebuildPrimitiveNeighlist_ = (primitiveWall_ != 0);
}

void FixWallGranOMP::pre_force(int vflag)
{
    const double halfskin = neighbor->skin*0.5;
    const int nlocal = atom->nlocal;

    x_ = atom->x;
    radius_ = atom->radius;
    cutneighmax_ = neighbor->cutneighmax;

    // build neighlist for primitive walls
    if(rebuildPrimitiveNeighlist_)
      primitiveWall_->buildNeighList(radius_ ? halfskin:(r0_+halfskin),x_,radius_,nlocal);

    rebuildPrimitiveNeighlist_ = false;
}

/* ----------------------------------------------------------------------
   force on each atom calculated via post_force
   called via verlet
------------------------------------------------------------------------- */

void FixWallGranOMP::post_force(int vflag)
{
    computeflag_ = 1;
    shearupdate_ = 1;
    if (update->setupflag) shearupdate_ = 0;
    addflag_ = 0;
    post_force_wall(vflag);
}

/* ----------------------------------------------------------------------
   post_force
------------------------------------------------------------------------- */

void FixWallGranOMP::post_force_wall(int vflag)
{
  // set pointers and values appropriately
  nlocal_ = atom->nlocal;
  x_ = atom->x;
  f_ = atom->f;
  radius_ = atom->radius;
  rmass_ = atom->rmass;

  if(fix_rigid_) {
      body_ = fix_rigid_->body;
      masstotal_ = fix_rigid_->masstotal;
  }

  if(fix_wallforce_) {
    wallforce_ = fix_wallforce_->array_atom;
  }

  cutneighmax_ = neighbor->cutneighmax;

  if(nlocal_ && !radius_ && r0_ == 0.) {
    error->fix_error(FLERR,this,"need either per-atom radius or r0_ being set");
  }

  if(store_force_) {
    for(int i = 0; i < nlocal_; i++) {
        vectorZeroize3D(wallforce_[i]);
    }
  }

  #pragma omp parallel
  {
    if(meshwall_ == 1) {
      post_force_mesh(vflag);
    } else {
      post_force_primitive(vflag);
    }
  }

  if(store_force_contact_) {
    if(meshwall_ == 0) {
      fix_wallforce_contact_->do_forward_comm();
    } else {
      for(int imesh = 0; imesh < n_FixMesh_; imesh++) {
        FixMesh_list_[imesh]->meshforceContact()->do_forward_comm();
      }
    }
  }
}

/* ----------------------------------------------------------------------
   post_force for mesh wall
------------------------------------------------------------------------- */

void FixWallGranOMP::post_force_mesh(int vflag)
{
    const int tid = omp_get_thread_num();

    // contact properties
    double v_wall[3],bary[3];
    double delta[3],deltan;

    const int nlocal = atom->nlocal;

    SurfacesIntersectData sidata;
    sidata.is_wall = true;
    sidata.computeflag = computeflag_;
    sidata.shearupdate = shearupdate_;

    for(int iMesh = 0; iMesh < n_FixMesh_; iMesh++)
    {
      FixContactHistoryMesh *fix_contact = FixMesh_list_[iMesh]->contactHistory();
      // mark all contacts for deletion at this point
      if(fix_contact) {
        FixNeighlistMeshOMP * meshNeighlist = static_cast<FixNeighlistMeshOMP*>(FixMesh_list_[iMesh]->meshNeighlist());
        fix_contact->resetDeletionPage(tid);

        int* b = meshNeighlist->partition_begin(tid);
        int* e = meshNeighlist->partition_end(tid);

	// mark all contacts for delettion at this point
        for(int* it = b; it != e; ++it) {
          assert(meshNeighlist->in_thread_partition(tid, *it));
          fix_contact->markForDeletion(tid, *it);
        }
      }
      // TODO if(store_force_contact_)
      // TODO  fix_wallforce_contact_ = FixMesh_list_[iMesh]->meshforceContact();
    }

    for(int iMesh = 0; iMesh < n_FixMesh_; iMesh++)
    {
      // need to enforce synchronization after each mesh, because thread partitions of different meshes
      // may overlap. without these barriers write-conflicts are possible.
      // load balancing keeps the negative effect of these barriers low
      #pragma omp barrier

      TriMesh *mesh = FixMesh_list_[iMesh]->triMesh();
      const int nTriAll = mesh->sizeLocal() + mesh->sizeGhost();
      FixContactHistoryMesh * const fix_contact = FixMesh_list_[iMesh]->contactHistory();

      // get neighborList and numNeigh
      FixNeighlistMeshOMP * meshNeighlist = static_cast<FixNeighlistMeshOMP*>(FixMesh_list_[iMesh]->meshNeighlist());
      int* b = meshNeighlist->partition_begin(tid);
      int* e = meshNeighlist->partition_end(tid);
      if(b == e) continue; // nothing to do

      vectorZeroize3D(v_wall);
      MultiVectorContainer<double,3,3> *vMeshC = mesh->prop().getElementProperty<MultiVectorContainer<double,3,3> >("v");

      sidata.jtype = FixMesh_list_[iMesh]->atomTypeWall();

      // moving mesh
      if(vMeshC)
      {
        double ***vMesh = vMeshC->begin();

        // loop owned and ghost triangles
        for(int iTri = 0; iTri < nTriAll; iTri++)
        {
          const std::vector<int> & neighborList = meshNeighlist->get_contact_list(iTri);
          const int numneigh = neighborList.size();
          for(int iCont = 0; iCont < numneigh; iCont++)
          {
            const int iPart = neighborList[iCont];

            // do not need to handle ghost particles
            if(iPart >= nlocal) continue;

            // do not handle particles not in thread region
            if(!meshNeighlist->in_thread_partition(tid, iPart)) continue;

            int idTri = mesh->id(iTri);

            deltan = mesh->resolveTriSphereContactBary(iPart,iTri,radius_ ? radius_[iPart]:r0_ ,x_[iPart],delta,bary);

            if(deltan > skinDistance_) //allow force calculation away from the wall
            {
            }
            else
            {
              if(fix_contact && ! fix_contact->handleContact(iPart,idTri,sidata.contact_history)) continue;

              for(int i = 0; i < 3; i++)
                v_wall[i] = (bary[0]*vMesh[iTri][0][i] + bary[1]*vMesh[iTri][1][i] + bary[2]*vMesh[iTri][2][i]);

              sidata.i = iPart;
              sidata.deltan = -deltan;
              sidata.delta[0] = -delta[0];
              sidata.delta[1] = -delta[1];
              sidata.delta[2] = -delta[2];
              post_force_eval_contact(sidata, v_wall,iMesh,FixMesh_list_[iMesh],mesh,iTri);
            }
          }
        }
      }
      // non-moving mesh - do not calculate v_wall, use standard distance function
      else
      {
        // loop owned and ghost particles
        for(int iTri = 0; iTri < nTriAll; iTri++)
        {
          const std::vector<int> & neighborList = meshNeighlist->get_contact_list(iTri);
          const int numneigh = neighborList.size();
          for(int iCont = 0; iCont < numneigh; iCont++)
          {
            const int iPart = neighborList[iCont];

            // do not need to handle ghost particles
            if(iPart >= nlocal) continue;

            // do not handle particles not in thread region
            // this ensures that contact history of a particle is only manipulated by a single thread
            if(!meshNeighlist->in_thread_partition(tid, iPart)) continue;

            int idTri = mesh->id(iTri);
            deltan = mesh->resolveTriSphereContact(iPart,iTri,radius_ ? radius_[iPart]:r0_,x_[iPart],delta);

            if(deltan > skinDistance_) //allow force calculation away from the wall
            {
            }
            else
            {
              if(fix_contact && ! fix_contact->handleContact(iPart,idTri,sidata.contact_history)) continue;
              sidata.i = iPart;
              sidata.deltan = -deltan;
              sidata.delta[0] = -delta[0];
              sidata.delta[1] = -delta[1];
              sidata.delta[2] = -delta[2];
              post_force_eval_contact(sidata, v_wall,iMesh,FixMesh_list_[iMesh],mesh,iTri);
            }
          }
        }
      }
    }

  // clean-up contacts
  for(int iMesh = 0; iMesh < n_FixMesh_; iMesh++)
  {
    FixContactHistoryMesh *fix_contact = FixMesh_list_[iMesh]->contactHistory();
      // clean-up contacts
      if(fix_contact) {
        FixNeighlistMeshOMP * meshNeighlist = static_cast<FixNeighlistMeshOMP*>(FixMesh_list_[iMesh]->meshNeighlist());
        int * b = meshNeighlist->partition_begin(tid);
        int * e = meshNeighlist->partition_end(tid);

        for(int * it = b; it != e; ++it) {
          fix_contact->cleanUpContact(*it);
        }
      }
  }
}

/* ----------------------------------------------------------------------
   post_force for primitive wall
------------------------------------------------------------------------- */

void FixWallGranOMP::post_force_primitive(int vflag)
{
  int *mask = atom->mask;

  SurfacesIntersectData sidata;
  sidata.is_wall = true;

  // contact properties
  double delta[3]={},deltan,rdist[3];
  double v_wall[] = {0.,0.,0.};
  double **c_history = 0;

  if(dnum() > 0)
    c_history = fix_history_primitive_->array_atom;

  // if shear, set velocity accordingly
  if (shear_) v_wall[shearDim_] = vshear_;

  // loop neighbor list
  int *neighborList;
  int nNeigh = primitiveWall_->getNeighbors(neighborList);

  for (int iCont = 0; iCont < nNeigh ; iCont++, neighborList++)
  {
    int iPart = *neighborList;

    if(!(mask[iPart] & groupbit)) continue;

    deltan = primitiveWall_->resolveContact(x_[iPart],radius_?radius_[iPart]:r0_,delta);

    if(deltan > skinDistance_) //allow force calculation away from the wall
    {
      if(c_history) vectorZeroizeN(c_history[iPart],dnum_);
    }
    else
    {
      bool particle_wall_intersection = true; //always true for spheres
      #ifdef SUPERQUADRIC_ACTIVE_FLAG
          if(atom->superquadric_flag) {
            double sphere_contact_point[3];
            vectorAdd3D(x_[iPart], delta, sphere_contact_point);
            sidata.pos_i = x_[iPart];
            sidata.quat_i = quat_[iPart];
            sidata.shape_i = shape_[iPart];
            sidata.roundness_i = roundness_[iPart];
            double closestPoint[3], closestPointProjection[3], point_of_lowest_potential[3];
            Superquadric particle(sidata.pos_i, sidata.quat_i, sidata.shape_i, sidata.roundness_i);
            particle_wall_intersection = particle.plane_intersection(delta, sphere_contact_point, closestPoint, point_of_lowest_potential);
            deltan = -MathExtraLiggghtsSuperquadric::point_wall_projection(delta, sphere_contact_point, closestPoint, closestPointProjection);
            vectorCopy3D(closestPoint, sidata.contact_point);
            sidata.is_non_spherical = true; //by default it is false
          }
      #endif
      if(particle_wall_intersection) { //always true for spheres
          if(shear_ && shearAxis_ >= 0)
          {
              primitiveWall_->calcRadialDistance(x_[iPart],rdist);
              vectorCross3D(shearAxisVec_,rdist,v_wall);
              
          }
          sidata.i = iPart;
          sidata.contact_history = c_history ? c_history[iPart] : NULL;
          sidata.deltan = -deltan;
          sidata.delta[0] = -delta[0];
          sidata.delta[1] = -delta[1];
          sidata.delta[2] = -delta[2];
          post_force_eval_contact(sidata,v_wall);
      } else {
        if(c_history) vectorZeroizeN(c_history[iPart],dnum_);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   actually calculate force, called for both mesh and primitive
------------------------------------------------------------------------- */

inline void FixWallGranOMP::post_force_eval_contact(SurfacesIntersectData & sidata, double * v_wall, int iMesh, FixMeshSurface *fix_mesh, TriMesh *mesh, int iTri)
{
  const int iPart = sidata.i;

  // deltan > 0 in compute_force
  // but negative in distance algorithm
  sidata.r = (radius_ ? radius_[iPart] : r0_) - sidata.deltan; // sign of corrected, because negative value is passed
  sidata.rsq = sidata.r*sidata.r;
  sidata.meff = rmass_ ? rmass_[iPart] : atom->mass[atom->type[iPart]];
  sidata.area_ratio = 1.;


  double force_old[3], f_pw[3];

  // if force should be stored - remember old force
  if(store_force_ || stress_flag_)
    vectorCopy3D(f_[iPart],force_old);

  // add to cwl
  if(cwl_ && addflag_)
  {
      double contactPoint[3];
      vectorAdd3D(x_[iPart],sidata.delta,contactPoint);
      #pragma omp critical
      cwl_->add_wall_1(iMesh,mesh->id(iTri),iPart,contactPoint, v_wall);
  }

  if(impl)
    impl->compute_force(this, sidata, v_wall);
  else
    compute_force(sidata, v_wall); // LEGACY CODE (SPH)

  // if force should be stored or evaluated
  if(store_force_ || stress_flag_)
  {
    vectorSubtract3D(f_[iPart],force_old,f_pw);

    if(store_force_)
        vectorAdd3D (wallforce_[iPart], f_pw, wallforce_[iPart]);

    if(stress_flag_ && fix_mesh->trackStress())
    {
        double delta[3];
        delta[0] = -sidata.delta[0];
        delta[1] = -sidata.delta[1];
        delta[2] = -sidata.delta[2];
        #pragma omp critical
        static_cast<FixMeshSurfaceStress*>(fix_mesh)->add_particle_contribution
        (
           iPart,f_pw,delta,iTri,v_wall
        );
    }
  }

  // add heat flux
  if(heattransfer_flag_) {
    #pragma omp critical
    addHeatFlux(mesh,iPart,sidata.jtype,sidata.deltan,1.);
  }
}

/* ---------------------------------------------------------------------- */

void FixWallGranOMP::addHeatFlux(TriMesh *mesh, int ip, int wall_type, double delta_n, double area_ratio)
{
    //r is the distance between the sphere center and wall
    double tcop, tcowall, hc, Acont, r;
    double reff_wall = atom->radius[ip];
    int itype = atom->type[ip];
    double ri = atom->radius[ip];

    if(mesh)
        Temp_wall = (*mesh->prop().getGlobalProperty< ScalarContainer<double> >("Temp"))(0);

    double *Temp_p = fppa_T->vector_atom;
    double *heatflux = fppa_hf->vector_atom;


    if(deltan_ratio)
       delta_n *= deltan_ratio[itype-1][wall_type-1];

    r = ri - delta_n;

    Acont = (reff_wall*reff_wall-r*r)*M_PI*area_ratio; //contact area sphere-wall
    tcop = th_cond[itype-1]; //types start at 1, array at 0
    tcowall = th_cond[wall_type-1];

    if ((fabs(tcop) < SMALL) || (fabs(tcowall) < SMALL)) hc = 0.;
    else hc = 4.*tcop*tcowall/(tcop+tcowall)*sqrt(Acont);

    if(computeflag_)
    {
        heatflux[ip] += (Temp_wall-Temp_p[ip]) * hc;
        Q_add += (Temp_wall-Temp_p[ip]) * hc * update->dt;
    }
    if(cwl_ && addflag_)
        cwl_->add_heat_wall(ip,(Temp_wall-Temp_p[ip]) * hc);
}

/* ---------------------------------------------------------------------- */

void FixWallGranOMP::add_contactforce_wall(int ip, const LCM::ForceData & i_forces)
{
  // add to fix wallforce contact
  // always add 0 as ID
  #pragma omp critical
  if(!fix_wallforce_contact_->has_partner(ip,0))
  {
    double forces_torques_i[6];
    vectorCopy3D(i_forces.delta_F,&(forces_torques_i[0]));
    vectorCopy3D(i_forces.delta_torque,&(forces_torques_i[3]));
    fix_wallforce_contact_->add_partner(ip,0,forces_torques_i);
  }
}

/* ---------------------------------------------------------------------- */

void FixWallGranOMP::cwl_add_wall_2(const LCM::SurfacesIntersectData & sidata, const LCM::ForceData & i_forces)
{
  const double fx = i_forces.delta_F[0];
  const double fy = i_forces.delta_F[1];
  const double fz = i_forces.delta_F[2];
  const double tor1 = i_forces.delta_torque[0]*sidata.area_ratio;
  const double tor2 = i_forces.delta_torque[1]*sidata.area_ratio;
  const double tor3 = i_forces.delta_torque[2]*sidata.area_ratio;
  #pragma omp critical
  cwl_->add_wall_2(sidata.i,fx,fy,fz,tor1,tor2,tor3,sidata.contact_history,sidata.rsq);
}
