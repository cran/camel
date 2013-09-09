#include "mymath.h"

double sign(double x){
    return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

double max(double x,double y){
    return (x > y) ? x : y;
}

double max_abs_vec(double * x, int n){
    int i;
    double tmp = fabs(x[0]);
    
    for(i=1; i<n ; i++){
        tmp = max(tmp, fabs(x[i]));
    }
    return tmp;
}

double max_vec(double * x, int n){
    int i;
    double tmp = x[0];
    
    for(i=1; i<n ; i++){
        tmp = max(tmp, x[i]);
    }
    return tmp;
}

void max_selc(double *x, double vmax, double *x_s, int n, int *n_s, double z){
    int i,tmp;
    double thresh = vmax-z;

    tmp = 0;
    for(i=0; i<n ; i++){
        if(x[i]>thresh){
            x_s[tmp] = x[i];
            tmp++;
        }
    }
    *n_s = tmp;
}

double min(double x,double y){
    return (x < y) ? x : y;
}

double l1norm(double * x, int n){
    int i;
    double tmp=0;

    for(i=0; i<n; i++) 
        tmp += fabs(x[i]);
    return tmp;
}

void fabs_vc(double *v_in, double *v_out, int n){
    int i;

    for(i=0; i<n; i++)
        v_out[i] = fabs(v_in[i]);
}

void max_fabs_vc(double *v_in, double *v_out, double *vmax, int *n1, int n, double z){
    int i,cnt;
    double tmp, v_abs;

    tmp = 0;
    cnt = 0;
    for(i=0; i<n; i++){
        v_abs = fabs(v_in[i]);
        v_out[i] = v_abs;
        tmp = max(tmp, v_abs);
    }
    *vmax = tmp;
    *n1 = n;
}

void sort_up_bubble(double *v, int n){
    int i,j;
    double tmp;
    int ischanged;

    for(i=n-1; i>=0; i--){ 
        ischanged = 0;
        for(j=0; j<i; j++){
            if(v[j]>v[j+1]){
                tmp = v[j];
                v[j] = v[j+1];
                v[j+1] = tmp;
                ischanged = 1;
            }
        }
        if(ischanged==0) 
            break;
    }
}


void get_residual(double *r, double *y, double *A, double *x, int *xa_idx, int *nn, int *mm)
{
    int i,j,b_idx;
    int n,m;
    double tmp;
    n = *nn;
    m = *mm;
    
    for(i=0;i<n;i++){
        tmp=0;
        for(j=0;j<m;j++){
            b_idx = xa_idx[j];
            tmp+=A[b_idx*n+i]*x[b_idx];
        }
        r[i] = y[i]-tmp;
    }
}

void get_dual(double *u, double *r, double *mmu, int *nn)
{
    int i,n;
    double mu, zv;
    mu = *mmu;
    n = *nn;
    zv = 1;
    for(i=0;i<n;i++){
        u[i] = r[i]/mu;
    }
    euc_proj(u, zv, n); //euclidean projection
}

void get_dual1(double *u, double *r, double *mmu, int *nn)
{
    int i,n;
    double mu, zv;
    mu = *mmu;
    n = *nn;
    zv = 1;
    for(i=0;i<n;i++){
        u[i] = r[i]/mu;
        if(u[i]>zv)
            u[i] = zv;
        if(u[i]<-zv)
            u[i] = -zv;
    }
}

void get_dual2(double *u, double *r, double *mmu, int *nn)
{
    int i,n;
    double mu, zv, tmp_sum;
    mu = *mmu;
    n = *nn;
    zv = 1;
    tmp_sum = 0;
    for(i=0;i<n;i++){
        u[i] = r[i]/mu;
        tmp_sum += u[i]*u[i];
    }
    tmp_sum = sqrt(tmp_sum);
    if(tmp_sum>=zv){
        for(i=0;i<n;i++){
            u[i] = u[i]/tmp_sum;
        }
    }
}

void get_grad(double *g, double *A, double *u, int *dd, int *nn)
{
    int i,j;
    int d,n;
    
    d = *dd;
    n = *nn;
    
    for(i=0;i<d;i++){
        g[i]=0;
        for(j=0;j<n;j++){
            g[i] -= A[i*n+j]*u[j];
        }
    }
}

void get_base(double *base, double *u, double *r, double *mmu, int *nn)
{
    int i,n;
    double mu,tmp;
    mu = *mmu;
    n = *nn; 
    tmp = 0;
    for(i=0;i<n;i++){
        tmp += u[i]*u[i];
    }

    *base = 0;
    for(i=0;i<n;i++){
        *base += u[i]*r[i];
    }
    *base -= mu*tmp/2;
}
// r = y - A * x, r is n by m, A is n by d, x d by m
void get_residual_mat(double *r, double *y, double *A, double *x, int *idx_x, int *size_x, int *nn, int *mm, int *dd)
{
    int i,j,k,id;
    int n,m,d,size;
    double tmp;
    n = *nn;
    m = *mm;
    d = *dd;
    size = *size_x;
    
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            tmp = 0;
            for(k=0;k<size;k++){
                id = idx_x[k];
                tmp += A[id*n+j]*x[i*d+id];
            }
            r[i*n+j] = y[i*n+j]-tmp;
        }
    }
}

// u = proj(r)
void get_dual_mat(double *u, double *r, double *mmu, int *nn, int *mm)
{
    int i,j,n,m;
    double mu, zv, tmp_sum;
    mu = *mmu;
    n = *nn;
    m = *mm;
    zv = 1;

    for(i=0;i<m;i++){
        tmp_sum = 0;
        for(j=0;j<n;j++){
            u[i*n+j] = r[i*n+j]/mu;
            tmp_sum += u[i*n+j]*u[i*n+j];
        }
        tmp_sum = sqrt(tmp_sum);
        if (tmp_sum >= zv) {
            for(j=0;j<n;j++){
                u[i*n+j] = u[i*n+j]/tmp_sum;
            }
        }
    }
}

void proj_mat_sparse(double *u, int *idx, int *size_u, double *lambda, int *nn, int *mm)
{
    int i,j,n,m,size,flag;
    double mu, zero, tmp_sum;
    n = *nn;
    m = *mm;
    zero = 0;
    size = 0;

    for(i=0;i<n;i++){
        tmp_sum = 0;
        for(j=0;j<m;j++){
            tmp_sum += u[j*n+i]*u[j*n+i];
        }
        tmp_sum = sqrt(tmp_sum);
        flag = 0;
        for(j=0;j<m;j++){
            u[j*n+i] = u[j*n+i]*max(1-*lambda/tmp_sum, zero);
            if(flag == 0){
                if(u[j*n+i] != 0){
                    flag = 1;
                }
            }
        }
        if(flag == 1){
            idx[size] = i;
            size++;
        }
    }
    *size_u = size;
}

// g = -A' * u, g is d by m, A is n by d, u is n by m
void get_grad_mat(double *g, double *A, double *u, int *dd, int *nn, int *mm)
{
    int i,j,k;
    int d,n,m;
    double tmp;

    d = *dd;
    n = *nn;
    m = *mm;
    
    for(i=0;i<m;i++){
        for(j=0;j<d;j++){
            tmp = 0;
            for(k=0;k<n;k++){
                tmp += A[j*n+k]*u[i*n+k];
            }
            g[i*d+j] = -tmp;
        }
    }
}

// base = trace(u' * r) + mu * ||u||_F^2/2, u is n by m, r is n by m
void get_base_mat(double *base, double *fro, double *u, double *r, double *mmu, int *nn, int *mm)
{
    int i,j,n,m;
    double mu,tmp1, tmp2;
    mu = *mmu;
    n = *nn; 
    m = *mm;
    tmp1 = 0;
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            tmp1 += u[i*n+j]*r[i*n+j];
        }
    }
    tmp2 = 0;
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            tmp2 += u[i*n+j]*u[i*n+j];
        }
    }
    *fro = tmp2;
    *base = tmp1 + mu*tmp2/2;
}

void dif_mat(double *x0, double *x1, double *x2, int *nn, int *mm)
{
    int i,j,n,m;
    
    n = *nn;
    m = *mm;
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            x0[i*n+j] = x1[i*n+j] - x2[i*n+j];
        }
    }
}

void dif_mat2(double *x0, double *x1, double *x2, double *cc2, int *nn, int *mm)
{
    int i,j,n,m;
    double c2;
    
    n = *nn;
    m = *mm;
    c2 = *cc2;
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            x0[i*n+j] = x1[i*n+j] - c2*x2[i*n+j];
        }
    }
}

// tr(x1'*x2), x1 is n by m, x2, is n by m
double tr_norm(double *x1, double *x2, int *nn, int *mm)
{
    int i,j,k,n,m;
    double trace;
    
    n = *nn;
    m = *mm;
    trace = 0;
    for(i=0; i<m; i++){
        for(j=0; j<m; j++){
            for(k=0; k<n; k++){
                trace += x1[i*n+k]*x2[j*n+k];
            }
        }
    }
    return trace;
}

// ||x||_F^2, x is n by m
double fro_norm(double *x, int *nn, int *mm)
{
    int i,j,n,m;
    double fro;
    
    n = *nn;
    m = *mm;
    fro = 0;
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            fro += x[i*n+j]*x[i*n+j];
        }
    }
    return fro;
}

double lnorm_12(double *x, int *nn, int *mm)
{
    int i,j,n,m;
    double lnorm,tmp;
    
    n = *nn;
    m = *mm;
    lnorm = 0;
    for(i=0; i<n; i++){
        tmp = 0;
        for(j=0; j<m; j++){
            tmp += x[j*n+i]*x[j*n+i];
        }
        lnorm += sqrt(tmp);
    }
    return lnorm;
}

void trunc_svd(double *U, double *Vt, double *S, double *x, double *eps, int *nn, int *mm, int *min_nnmm)
{
    int i,j,k,n,m,min_nm;
    double zero, tmp;
    
    n = *nn;
    m = *mm;
    zero = 0;
    min_nm = *min_nnmm;
    for(i=0;i<min_nm;i++){
        S[i] = max(S[i]-*eps, zero);
    }
    for(i=0;i<m;i++){ // z1 = U*S*Vt, U is n by min_dm, S is min_dm, Vt is min_dm by m
        for(j=0;j<n;j++){
            tmp = 0;
            for(k=0;k<min_nm;k++){
                tmp += U[k*n+j]*S[k]*Vt[i*min_nm+k];
            }
            x[i*n+j] = tmp;
        }
    }
}

// x0 <- x1
void equ_mat(double *x0, double *x1, int *nn, int *mm)
{
    int i,j,n,m;
    
    n = *nn;
    m = *mm;
    
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            x0[i*n+j] = x1[i*n+j];
        }
    }
}
