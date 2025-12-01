#pragma once

#include "gpu_core.hpp"

void calculate_residual_launch_kernel(int dtypenum,
                                      int nG,
                                      int nn,
                                      void* residual_ng,
                                      void* eps_n,
                                      void* wf_nG,
                                      gpuStream_t stream);

void pwlfc_expand_gpu_launch_kernel(int dtypenum,
                                    void* f_Gs,
                                    void* Gk_Gv,
                                    void* pos_av,
                                    void* eikR_a,
                                    void* Y_GL,
                                    int* l_s,
                                    int* a_J,
                                    int* s_J,
                                    void* f_GI,
                                    int* I_J,
                                    int nG,
                                    int nJ,
                                    int nL,
                                    int nI,
                                    int natoms,
                                    int nsplines,
                                    bool cc,
                                    gpuStream_t stream);

void pw_insert_gpu_launch_kernel(
                             int dtypenum,
                             int nb,
                             int nG,
                             int nQ,
                             void* c_nG,
                             int32_t* Q_G,
                             double scale,
                             void* tmp_nQ,
                             int rx, int ry, int rz,
                             gpuStream_t stream);

void pw_norm_gpu_launch_kernel(int dtypenum,
                               int nx, int nG,
                               void* result_x,
                               void* C_xG,
                               gpuStream_t stream);

void pw_norm_kinetic_gpu_launch_kernel(int dtypenum,
                                       int nx, int nG,
                                       void* result_x,
                                       void* C_xG,
                                       void* kin_G,
                                       gpuStream_t stream);

void pw_amend_insert_realwf_gpu_launch_kernel(int dtypenum,
                                              int nb,
                                              int nx,
                                              int ny,
                                              int nz,
                                              int n,
                                              int m,
                                              void* array_nQ,
                                              gpuStream_t stream);

void add_to_density_gpu_launch_kernel(int nb,
                                      int nR,
                                      double* f_n,
                                      void* psit_nR,
                                      double* rho_R,
                                      int dtypenum,
                                      gpuStream_t stream);


void dH_aii_times_P_ani_launch_kernel(int dtypenum,
                                      int nA, int nn,
                                      int nI, int32_t* ni_a,
                                      void* dH_aii_dev,
                                      void* P_ani_dev,
                                      void* outP_ani_dev,
                                      gpuStream_t stream);

void evaluate_pbe_launch_kernel(int nspin, int ng,
                                double* n,
                                double* v,
                                double* e,
                                double* sigma,
                                double* dedsigma,
                                gpuStream_t stream);

void evaluate_lda_launch_kernel(int nspin, int ng,
                                double* n,
                                double* v,
                                double* e,
                                gpuStream_t stream);

