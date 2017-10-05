#ifndef LMP_MULTISPHERE_PARALLEL_H
#define LMP_MULTISPHERE_PARALLEL_H

#include "multisphere.h"
#include "comm.h"

namespace LAMMPS_NS {

  class MultisphereParallel : public Multisphere {

    friend class FixMultisphere;

    public:

      MultisphereParallel(LAMMPS *lmp);
      ~MultisphereParallel();

      int pack_exchange_rigid(int i, double *buf);
      int unpack_exchange_rigid(double *buf);

      void writeRestart(FILE *fp);
      void restart(double *list);

    private:

      void exchange();
      void grow_send(int, int);
      void grow_recv(int);

      // current size of send/recv buffer
      // send buffer and recv buffer for all comm
      int maxsend_,maxrecv_;
      double *buf_send_;
      double *buf_recv_;
  };

  // *************************************
  #include "multisphere_parallel_I.h"
  // *************************************
}

#endif
