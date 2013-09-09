#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "R.h"
#include "mymath.h"

void cmr_sparse_mfista(double *b, double *A, double *beta, int *nn, int *mm, int *dd, double *mu, int *ite_cnt_init, int *ite_cnt_ex, int *ite_cnt_in, double *lambda, int * nnlambda, int *max_ite, double *prec, double *L)
{
    int i,j,k,m,dim,ndata,mdata,nlambda,max_ite1,ite0,ite1,ite2,ite21,gap_track;
    int size,size_x0, size_y1, size_z1, w_idx, min_dm, max_dm, dm,flag,n_ls,cnt_ls;
    double T,Tinv,T0,imu,cpuTime,opt_dif,zero,fro;
    double x1_norm1,y1_norm1,z1_norm1,y1_norm1_pre,obj_base,ilambda,ilambda0,Q,Fx,Fz;
    double norm_dif,z_dif,x_dif,y1_dif,obj_tmp,eps1,t1,t2,ratio,epsT,tmp;
    clock_t start, end;

    dim = *dd;
    ndata = *nn;
    mdata = *mm;
    dm = dim*mdata;
    imu = *mu;
    max_ite1 = *max_ite;
    nlambda = *nnlambda;
    eps1 = *prec;
    ratio = 0.8;
    epsT = 1e-2;
    y1_norm1 = 0;
    min_dm = min(dim, mdata);
    max_dm = max(dim, mdata);
    zero = 0;
    n_ls = 100;
    double *F_ls = (double*) malloc(n_ls*sizeof(double));
    double *x1 = (double*) malloc(dim*mdata*sizeof(double));
    double *x0 = (double*) malloc(dim*mdata*sizeof(double));
    double *y1 = (double*) malloc(dim*mdata*sizeof(double));
    double *y2 = (double*) malloc(dim*mdata*sizeof(double));
    double *z1 = (double*) malloc(dim*mdata*sizeof(double));
    double *z0 = (double*) malloc(dim*mdata*sizeof(double));
    double *z1y1_dif = (double*) malloc(dim*mdata*sizeof(double));
    double *u_y = (double*) malloc(ndata*mdata*sizeof(double));
    double *u_z = (double*) malloc(ndata*mdata*sizeof(double));
    double *u_x = (double*) malloc(ndata*mdata*sizeof(double));
    double *bAx0 = (double*) malloc(ndata*mdata*sizeof(double));
    double *bAy1 = (double*) malloc(ndata*mdata*sizeof(double));
    double *bAz1 = (double*) malloc(ndata*mdata*sizeof(double));
    double *g = (double*) malloc(dim*mdata*sizeof(double));
    int *idx_x0 = (int*) malloc(dim*sizeof(int));
    int *idx_y1 = (int*) malloc(dim*sizeof(int));
    int *idx_z1 = (int*) malloc(dim*sizeof(int));
    
    for(i=0;i<mdata;i++){
        for(j=0;j<dim;j++){
            x1[i*dim+j]=0;
            x0[i*dim+j]=0;
            y1[i*dim+j]=0;
            z1[i*dim+j]=0;
        }
    }
    for(i=0;i<dim;i++){
        idx_x0[i]=0;
        idx_y1[i]=0;
        idx_z1[i]=0;
    }
    size_x0 = 0;
    size_y1 = 0;
    size_z1 = 0;
    for(m=0;m<nlambda;m++){
        cnt_ls = 0;
        T = (*L)/imu;
        T0 = T;

        t1 = 1;
        ilambda = lambda[m];
        get_residual_mat(bAy1, b, A, y1, idx_y1, &size_y1, &ndata, &mdata, &dim); // bAy1=b-A*y1
        get_dual_mat(u_y, bAy1, &imu, &ndata, &mdata); // u_y=proj(bAy1)
        get_grad_mat(g, A, u_y, &dim, &ndata, &mdata); // g=-A'*u_y
        get_base_mat(&obj_base, &fro, u_y, bAy1, &imu, &ndata, &mdata); // obj_base=tr(u_y'*bAy1) + imu*||u_y||_F^2/2

        gap_track = 1;
        ite0 = 0;
        while(gap_track == 1 && T>epsT){
            ilambda0 = ilambda/T;
            Tinv = 1/T;
            dif_mat2(z1,y1,g,&Tinv,&dim,&mdata); // z1 = y1 - g/T;
            proj_mat_sparse(z1, idx_z1, &size_z1, &ilambda0, &dim, &mdata);
            dif_mat(z1y1_dif, z1, y1, &dim, &mdata);
            get_base_mat(&Q, &fro, z1y1_dif, g, &T, &dim, &mdata);
            Q += obj_base;
            
            get_residual_mat(bAz1, b, A, z1, idx_z1, &size_z1, &ndata, &mdata, &dim); // bAz1=b-A*z1
            get_dual_mat(u_z, bAz1, &imu, &ndata, &mdata); //u_z=proj(bAz1)
            get_base_mat(&Fz, &fro, u_z, bAz1, &imu, &ndata, &mdata); //obj_base=u_z*bAz1+imu*||u_z||_2^2/2
            
            ite0++;
            if(Fz<Q) T = T*ratio;
            else {
                T = T/ratio;
                if(ite0>1)
                    gap_track = 0;
            }
        }
        ilambda0 = ilambda/T;
        Tinv = 1/T;
        dif_mat2(z1,y1,g,&Tinv,&dim,&mdata);
        proj_mat_sparse(z1, idx_z1, &size_z1, &ilambda0, &dim, &mdata);
        get_residual_mat(bAz1, b, A, z1, idx_z1, &size_z1, &ndata, &mdata, &dim); // bAz1=b-A*z1
        get_dual_mat(u_z, bAz1, &imu, &ndata, &mdata); //u_z=proj(bAz1)
        get_base_mat(&Fz, &fro, u_z, bAz1, &imu, &ndata, &mdata); //obj_base=u_z*bAz1+imu*||u_z||_2^2/2
        Fz += ilambda*lnorm_12(z1, &dim, &mdata);
        
        proj_mat_sparse(x0, idx_x0, &size_x0, &ilambda0, &dim, &mdata);
        get_residual_mat(bAx0, b, A, x0, idx_x0, &size_x0, &ndata, &mdata, &dim); // bAx0=b-A*x0
        get_dual_mat(u_x, bAx0, &imu, &ndata, &mdata); //u_x=proj(bAx0)
        get_base_mat(&Fx, &fro, u_x, bAx0, &imu, &ndata, &mdata); //obj_base=u_x*bAx0-imu*||u_x||_2^2/2
        Fx += ilambda*lnorm_12(x0, &dim, &mdata);
        
        if(Fx>Fz){
            equ_mat(x1, z1, &dim, &mdata);
        }
        else{
            equ_mat(x1, x0, &dim, &mdata);
        }
        
        t2 = (1+sqrt(1+4*t1*t1))/2;
        size_y1 = 0;
        for(i=0;i<dim;i++){
            flag = 0;
            for(j=0;j<mdata;j++){
                y2[j*dim+i] = x1[j*dim+i]+(x1[j*dim+i]-x0[j*dim+i])*(t1-1)/t2+(z1[j*dim+i]-x1[j*dim+i])*t1/t2;
                if(flag == 0){
                    if(y2[j*dim+i] != 0){
                        flag = 1;
                    }
                }
            }
            if(flag == 1){
                idx_y1[size_y1] = i;
                size_y1++;
            }
        }
        equ_mat(z0, z1, &dim, &mdata);
        equ_mat(x0, x1, &dim, &mdata);
        equ_mat(y1, y2, &dim, &mdata);
        t1 = t2;
//if(m==0)
//printf("Q=%f,F=%f,T=%f,T0=%f \n",Q,F,T,T0);
        ite1=0;
        ite21=0;
        y1_dif = 1;
        opt_dif = 1;

        get_residual_mat(bAy1, b, A, y1, idx_y1, &size_y1, &ndata, &mdata, &dim); // bAy1=b-A*y1
        get_dual_mat(u_y, bAy1, &imu, &ndata, &mdata); //u_y=proj(bAy1)
        get_grad_mat(g, A, u_y, &dim, &ndata, &mdata); //g=-A*u_y
        while(y1_dif>eps1 && ite1<max_ite1){
        //while(opt_dif>eps1 && ite1<max_ite1){
            ite2=0;
            if(T<T0){
                get_base_mat(&obj_base, &fro, u_y, bAy1, &imu, &ndata, &mdata); //obj_base=u_y*bAy1-imu*||u_y||_2^2/2
                gap_track = 1;
                while(gap_track == 1){
                    ilambda0 = ilambda/T;
                    Tinv = 1/T;
                    dif_mat2(z1,y1,g,&Tinv,&dim,&mdata); // z1 = y1 - g/T;
                    proj_mat_sparse(z1, idx_z1, &size_z1, &ilambda0, &dim, &mdata);
                    dif_mat(z1y1_dif, z1, y1, &dim, &mdata);
                    get_base_mat(&Q, &fro, z1y1_dif, g, &T, &dim, &mdata);
                    Q += obj_base;
                    
                    get_residual_mat(bAz1, b, A, z1, idx_z1, &size_z1, &ndata, &mdata, &dim); // bAz1=b-A*z1
                    get_dual_mat(u_z, bAz1, &imu, &ndata, &mdata); //u_z=proj(bAz1)
                    get_base_mat(&Fz, &fro, u_z, bAz1, &imu, &ndata, &mdata); //obj_base=u_z*bAz1+imu*||u_z||_2^2/2
                    
                    if(Fz>Q) T = T/ratio;
                    else gap_track = 0;
                    ite2++;
                }
            }
            else {
                ilambda0 = ilambda/T0;
                Tinv = 1/T0;
                dif_mat2(z1,y1,g,&Tinv,&dim,&mdata);
                proj_mat_sparse(z1, idx_z1, &size_z1, &ilambda0, &dim, &mdata);
            }

            get_residual_mat(bAz1, b, A, z1, idx_z1, &size_z1, &ndata, &mdata, &dim); // bAz1=b-A*z1
            get_dual_mat(u_z, bAz1, &imu, &ndata, &mdata); //u_z=proj(bAz1)
            get_base_mat(&Fz, &fro, u_z, bAz1, &imu, &ndata, &mdata); //obj_base=u_z*bAz1+imu*||u_z||_2^2/2
            Fz += ilambda*lnorm_12(z1, &dim, &mdata);
     
            if(Fx>Fz){
                equ_mat(x1, z1, &dim, &mdata);
                Fx = Fz;
            }
            else
                equ_mat(x1, x0, &dim, &mdata);
            
            t2 = (1+sqrt(1+4*t1*t1))/2;
            y1_norm1 = 0;
            x1_norm1 = 0;
            size_y1 = 0;
            for(i=0;i<dim;i++){
                flag = 0;
                for(j=0;j<mdata;j++){
                    y2[j*dim+i] = x1[j*dim+i]+(x1[j*dim+i]-x0[j*dim+i])*(t1-1)/t2+(z1[j*dim+i]-x1[j*dim+i])*t1/t2;
                    if(flag == 0){
                        if(y2[j*dim+i] != 0){
                            flag = 1;
                        }
                    }
                }
                if(flag == 1){
                    idx_y1[size_y1] = i;
                    size_y1++;
                }
            } 
//if(ite1==10){
//printf("y2 \n");
//for(i=0;i<6;i++){
//for(j=0;j<6;j++){
//printf("[%d,%d]=%.9f, ",i,j,y2[j*dim+i]);
//}
//printf("\n");
//}
//printf("Fz=%f,Fx=%f \n",Fz,Fx);
//printf("\n");
//break;}
            equ_mat(z0, z1, &dim, &mdata);
            equ_mat(x0, x1, &dim, &mdata);
            equ_mat(y1, y2, &dim, &mdata);
            t1 = t2;
            ite1++;
            ite21 += ite2;
            get_residual_mat(bAy1, b, A, y1, idx_y1, &size_y1, &ndata, &mdata, &dim); // bAy1=b-A*y1
            get_dual_mat(u_y, bAy1, &imu, &ndata, &mdata); //u_y=proj(bAy1)
            get_grad_mat(g, A, u_y, &dim, &ndata, &mdata); //g=-A*u_y
            y1_dif = fro_norm(bAy1, &ndata, &mdata)/1e6;
            
            if(cnt_ls<n_ls){
                F_ls[cnt_ls] = Fx;
                cnt_ls++;
            }
            else{
                for(i=0;i<n_ls-1;i++){
                    F_ls[i] = F_ls[i+1];
                }
                F_ls[n_ls-1] = Fx;
                if(F_ls[n_ls-1] = F_ls[0])
                    y1_dif = 0;
            }
        }
        for(i=0;i<mdata;i++){
            for(j=0;j<dim;j++){
                beta[m*dm+i*dim+j] = x0[i*dim+j];
            }
        }
        ite_cnt_init[m] = ite0;
        ite_cnt_ex[m] = ite1;
        ite_cnt_in[m] = ite21;
    }
    //while (getchar() != '\n');
    free(F_ls);
    free(x0);
    free(x1);
    free(y1);
    free(y2);
    free(z0);
    free(z1);
    free(z1y1_dif);
    free(u_x);
    free(u_y);
    free(u_z);
    free(bAx0);
    free(bAy1);
    free(bAz1);
    free(g);
    free(idx_x0);
    free(idx_y1);
    free(idx_z1);
}
