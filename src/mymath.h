#ifndef MYMATH_H
#define MYMATH_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <R_ext/BLAS.h>
#include <R_ext/Linpack.h>

double sign(double x);

double max(double x,double y);

double max_abs_vec(double * x, int n);

double max_vec(double * x, int n);

void max_selc(double *x, double vmax, double *x_s, int n, int *n_s, double z);

double min(double x,double y);

double l1norm(double * x, int n);

void euc_proj(double * v, double z, int n);

double fun1(double lambda, double * v, double z, int n);

double mod_bisec(double * v, double z, int n);

void fabs_vc(double *v_in, double *v_out, int n);

void max_fabs_vc(double *v_in, double *v_out, double *vmax, int *n1, int n, double z);

void sort_up_bubble(double *v, int n);

void get_residual(double *r, double *y, double *A, double *x, int *xa_idx, int *nn, int *mm);

void get_dual(double *u, double *r, double *mmu, int *nn);

void get_dual1(double *u, double *r, double *mmu, int *nn);

void get_dual2(double *u, double *r, double *mmu, int *nn);

void get_grad(double *g, double *A, double *u, int *dd, int *nn);

void get_base(double *base, double *u, double *r, double *mmu, int *nn);

// r = y - A* x
void get_residual_mat(double *r, double *y, double *A, double *x, int *idx, int *size, int *nn, int *mm, int *dd);

// u = proj(r)
void get_dual_mat(double *u, double *r, double *mmu, int *nn, int *mm);

// g = -A * u
void get_grad_mat(double *g, double *A, double *u, int *dd, int *nn, int *mm);

// base = u * r - mu * ||u||_F^2/2
void get_base_mat(double *base, double *fro, double *u, double *r, double *mmu, int *nn, int *mm);

void dif_mat(double *x0, double *x1, double *x2, int *nn, int *mm);

void dif_mat2(double *x0, double *x1, double *x2, double *c2, int *nn, int *mm);

// tr(x1'*x2)
double tr_norm(double *x1, double *x2, int *nn, int *mm);

// ||x||_F^2
double fro_norm(double *x, int *nn, int *mm);

// ||x||_12;
double lnorm_12(double *x, int *nn, int *mm);

void trunc_svd(double *U, double *Vt, double *S, double *x, double *eps, int *nn, int *mm, int *min_nnmm);

void equ_mat(double *x0, double * x1, int *nn, int *mm);

void proj_mat_sparse(double *u, int *idx, int *size_u, double *lambda, int *nn, int *mm);

#endif
