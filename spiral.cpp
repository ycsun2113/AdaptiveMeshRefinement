// icpc -o spiral -std=c++14 spiral.cpp -lfftw3
// ./spiral Ngrid TLev

#include <iostream> 
#include <stdlib.h>
#include <iomanip>
#include <math.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <time.h>
#include <queue>

using namespace std;

#define ALPHA 0.5
#define THICK 0.05

vector<vector<double>> gmesh2(int N, int Tlevel, int ishw, double thres);

double d3frce(vector<vector<double>> G, int ishw, vector<double>& FX0IJ, vector<double>& FY0IJ, vector<double>& RFORCE, vector<double>& LFX0IJ, vector<double>& LFY0IJ);
void d3frceFL1DL2(vector<int> FP, vector<int> DP, vector<vector<double>> G, vector<double> srho, vector<double> dxsrho, vector<double> dysrho, vector<double>& LFX0IJ, vector<double>& LFY0IJ);
void FFTfrceFL1DL2(vector<int> FP, vector<int> DP, vector<vector<double>> G, vector<double> FFTsrho, vector<double> FFTdxsrho, vector<double> FFTdysrho, vector<double>& LFX0IJ, vector<double>& LFY0IJ);
void FFTfrceFL2DL1(vector<int> FP, vector<int> DP, vector<vector<double>> G, vector<double> FFTsrho, vector<double> FFTdxsrho, vector<double> FFTdysrho, vector<double>& LFX0IJ, vector<double>& LFY0IJ);

vector<int> find_unique(vector<double> &arr, int n);
double xK0(double xi, double yj, double xii, double yjj, double dx, double dy);
double xKx1(double xi, double yj, double xii, double yjj, double dx, double dy);
double xKy1(double xi, double yj, double xii, double yjj, double dx, double dy);
int sign(double x);
double createSpiral(int N);

vector<vector<double>> conv2d(const vector<vector<double>>& K, const vector<vector<double>>& rho);
vector<vector<double>> extract_submatrix(const vector<vector<double>>& arr, int row_start, int row_stride, int col_start, int col_stride);

int main(int argc, char* argv[]){
    int Ngrid = atoi(argv[1]);
    int TLev = atoi(argv[2]);
    double thres = createSpiral(Ngrid);
    vector<vector<double>> G = gmesh2(Ngrid, TLev, 0, thres);

    vector<double> FX0IJ(G.size(), 0.0);
    vector<double> FY0IJ(G.size(), 0.0);
    vector<double> RFORCE(G.size(), 0.0);
    vector<double> LFX0IJ(G.size(), 0.0);
    vector<double> LFY0IJ(G.size(), 0.0);

    // test for gmesh
    int EL_count = 0;
    for(int i=0; i<G.size(); i++){
        EL_count += G[i][4];
    }
    // cout << "----------------------------------------------------------" << endl;
    cout << "Ngrid = " << Ngrid ;
    cout << ", Level = " << TLev;
    cout << ", Nk^2 = " << Ngrid*Ngrid;
    cout << ", G size = " << G.size();
    cout << ", EL = " << EL_count;
    
    double time_sum = 0.;
    double ans[10] = {0.};
    for(int i=0; i<10; i++){
        ans[i] = d3frce(G, 0, FX0IJ, FY0IJ, RFORCE, LFX0IJ, LFY0IJ);
        time_sum = time_sum + ans[i];
    }
    double time_mean = time_sum/10.;
    cout << ", runtime = ";
    cout << fixed << setprecision(8) << time_mean;
    double time_sum2 = 0.;
    for(int i=0; i<10; i++){
        time_sum2 = time_sum2 + (ans[i]-time_mean)*(ans[i]-time_mean);
    }
    cout << ", std = ";
    cout << fixed << setprecision(8) << sqrt(time_sum2/10.) << endl; 
    cout << "----------------------------------------------------------------------------" << endl;

    // test for d3frce 
    // for(int i=0; i<LFY0IJ.size(); i++){
    //     cout << fixed << setprecision(4) << LFX0IJ[i] << endl ;
    // }
    
    return 0;
}


vector<vector<double>> gmesh2(int N, int Tlevel, int ishw, double thres){
    
    if(N==0 && Tlevel==0 && ishw==0){
        N=8;
        ishw=0;
        Tlevel=1;
    }
    double xmin=-1.0, xmax=1.0, Nlevel = 1.0;
    double dx = (xmax-xmin)/(N);
    double dy=dx;
    int size = ((xmax - xmin) / dx) + 1;
    int G_size = N * N;

    vector<vector<double>> G(G_size, vector<double>(9, 0.0));

    vector<double> x(size);
    vector<double> y(size);

    for (int i = 0; i < size; ++i) {
        x[i] = xmin + i * dx;
        y[i] = x[i];
    }

    /*********************************/
    /***** The coarst grid zones *****/
    /*********************************/
    int k=0;
    for(int j=0; j<N; j++){
        for(int i=0; i<N; i++){
            G[k][0] = 0.5*(x[i]+x[i+1]);
            G[k][1] = 0.5*(y[j]+y[j+1]);
            G[k][2] = Nlevel;
            G[k][3] = 0.0;
            G[k][4] = 1.0;

            if(i==0) G[k][5]=-1.0;
            else G[k][5]=double(k);

            if(j==0) G[k][6]=-1.0;
            else G[k][6]=k+1-N;

            if(i==N-1) G[k][7]=-1.0;
            else G[k][7]=double(k+2);

            if(j==N-1) G[k][8]=-1.0;
            else G[k][8]=k+1.0+N;

            k++;
        }
    }

    /***************************/
    /***** Refine the mesh *****/
    /***************************/
    
    int IN = Nlevel;
    int Nx, kend, ii;
    bool TFflag;
    double r;
    double R = ALPHA;
    double thick = THICK;
    double s = 0., srho;

    while(IN<Tlevel){

        Nx = G.size();
        kend = Nx;
        
        for(int k=0; k<Nx; k++){
            TFflag = 0;
            r = sqrt(G[k][0]*G[k][0] + G[k][1]*G[k][1]);

            // TFflag = (abs(r-0.25)<0.05);
            // TFflag = (abs(r-R) < thick/pow(2.0,IN));
            // TFflag = (abs(r-R) < thick);

            complex<double> z(G[k][0], G[k][1]);
            s = arg(z);
            srho = 10 * (exp(-abs(r - s/3)*2) + exp(-abs(r - s/3 - M_PI/3)*2)) / (2 + r);
            // TFflag = srho > thres;
            TFflag = srho > thres + 0.5*(IN-1);

            if(TFflag==true && (G[k][4]==1)){
                G.resize(kend + 4, vector<double>(9));
                ii = kend;
                G[ii][0] = G[k][0] - dx/pow(2.0, G[k][2])/2.0;
                G[ii][1] = G[k][1] + dy/pow(2.0, G[k][2])/2.0;
                G[ii][2] = G[k][2] + 1.0;
                G[ii][3] = double(k) + 1.0;
                G[ii][4] = 1.0;
                G[ii][5] = G[k][5];
                G[ii][6] = double(ii) + 1.0 + 2.0;
                G[ii][7] = double(ii) + 1.0 + 1.0;
                G[ii][8] = G[k][8];

                ii++;
                G[ii][0] = G[k][0] - dx/pow(2.0, G[k][2])/2.0;
                G[ii][1] = G[k][1] - dy/pow(2.0, G[k][2])/2.0;
                G[ii][2] = G[k][2] + 1.0;
                G[ii][3] = double(k) + 1.0;
                G[ii][4] = 1.0;
                G[ii][5] = double(ii) + 1.0 - 1.0;
                G[ii][6] = double(ii) + 1.0 + 2.0;
                G[ii][7] = G[k][7];
                G[ii][8] = G[k][8];

                ii++;
                G[ii][0] = G[k][0] + dx/pow(2.0, G[k][2])/2.0;
                G[ii][1] = G[k][1] - dy/pow(2.0, G[k][2])/2.0;
                G[ii][2] = G[k][2] + 1.0;
                G[ii][3] = double(k) + 1.0;
                G[ii][4] = 1.0;
                G[ii][5] = G[k][5];
                G[ii][6] = G[k][6];
                G[ii][7] = double(ii) + 1.0 + 1.0;
                G[ii][8] = double(ii) + 1.0 - 2.0;

                ii++;
                G[ii][0] = G[k][0] + dx/pow(2.0, G[k][2])/2.0;
                G[ii][1] = G[k][1] + dy/pow(2.0, G[k][2])/2.0;
                G[ii][2] = G[k][2] + 1.0;
                G[ii][3] = double(k) + 1.0;
                G[ii][4] = 1.0;
                G[ii][5] = double(ii) + 1.0 - 1.0;
                G[ii][6] = G[k][6];
                G[ii][7] = G[k][7];
                G[ii][8] = double(ii) + 1.0 - 2.0;

                G[k][4] = 0.0;

                kend = ii+1;
            }
        }
        IN++;
    }
    // if(ishw==1){
    //     shwGrid(G, dx, dy);
    // }
    
    return G;
}



/***************************************************************/
/************************ Direct Method ************************/
/***************************************************************/
double d3frce(vector<vector<double>> G, int ishw, vector<double>& FX0IJ, vector<double>& FY0IJ, vector<double>& RFORCE, vector<double>& LFX0IJ, vector<double>& LFY0IJ){
    int NX = G.size();
    int NNN = G[0].size();
    int NN = 0;
    for (int i = 0; i < NX; i++) {
        if (G[i][2] == 1)
            NN++;
    }
    double NPT = sqrt(NN);
    double xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0;
    double dxx = (xmax-xmin)/NPT;
    double dyy = dxx;
    double dx0 = dxx;
    double dy0 = dyy;
    vector<double> x(NPT); // x = xmin+0.5*dxx : dxx : xmax-0.5*dxx;
    generate(x.begin(), x.end(), [&, i=0]() mutable{
        return xmin+0.5*dxx + i++*dxx;
    });
    vector<double> y(NPT); // x = xmin+0.5*dxx : dxx : xmax-0.5*dxx;
    generate(y.begin(), y.end(), [&, i=0]() mutable{
        return ymin+0.5*dyy + i++*dyy;
    });

    // [xq, yx] = meshgrid(x,y);
    vector<vector<double>> xq;
    vector<vector<double>> yq;
    int nx = x.size();
    int ny = y.size();
    xq.resize(ny);
    yq.resize(ny);
    for (int i = 0; i < ny; i++) {
        xq[i].resize(nx, x[i]);
        yq[i].resize(nx, y[i]);
    }

    int k=1;
    // double R=0.25;
    double R = ALPHA;
    double sgm0=1.0;
    double dx = (xmax-xmin)/NPT;
    double dy = dx;
    
    // zeros:
    // vector<double> FX0IJ(NX,0.0);
    // vector<double> FY0IJ(NX,0.0);
    vector<double> srho(NX,0.0);
    vector<double> dxsrho(NX,0.0);
    vector<double> dysrho(NX,0.0);

    double ss = 0.0, rr = 0.0;
    double term1, term2, term3, term4, term5, term6, term7, term8;
    double xterm, yterm;

    for(int i=0; i<NX; i++){
        xterm = G[i][0];
        yterm = G[i][1];
        complex<double> z(xterm, yterm);
        ss = arg(z);
        rr = sqrt(xterm*xterm + yterm*yterm);
        term1 = rr - ss/3;
        term2 = rr - ss/3 - M_PI/3;

        term3 = exp(-abs(term1)*2);
        term4 = exp(-abs(term2)*2);

        srho[i] = 10 * (term3 + term4) / (2 + rr);

        term5 = xterm/rr + yterm/(3*rr*rr);
        term6 = xterm/(3*rr*rr) - yterm/rr;
        term7 = abs(ss/3 - rr + M_PI/3);
        term8 = abs(ss/3 - rr);

        dxsrho[i] = 10 * (-2*term5*term2*term4/abs(term2) - 2*term5*term1*term3/abs(term1))/(2+rr) - 10 * xterm * (term3+term4) / (rr * (2+rr) * (2+rr));
        dysrho[i] = 10 * (-2*term6*term4*(-1*term2)/term7 - 2*term6*term3*(-1*term1)/abs(term1))/(2+rr) - 10 * yterm * (term3+term4) / (rr * (2+rr) * (2+rr));
    }

    // vector<double> afx(NX,0.0);
    // vector<double> afy(NX,0.0);
    // vector<double> afr(NX,0.0);
    // double r, ccyen;

    /********** srho, dxsrho, dysrho 已測試正確 **********/
    // for(int i=0; i<NX; i++){
    //     r = sqrt(G[i][0]*G[i][0]+G[i][1]*G[i][1]); // r=sqrt(x^2+y^2)
    //     if(r<=R){
    //         ccyen = sqrt(1.0-(r/R)*(r/R));
    //         srho[i] = sgm0*pow(ccyen,3.0);
    //         dxsrho[i] = -3.0*sgm0*ccyen*G[i][0]/(R*R);
    //         dysrho[i] = -3.0*sgm0*ccyen*G[i][1]/(R*R);
    //         afr[i] = -3.0*M_PI*M_PI*sgm0*r/(16.0*R*R*R)*(4.0*R*R-3.0*r*r)*G[i][4];
    //     }
    //     else{
    //         // afr(i) = -3*pi*sgm0/(8*R^3)*(r*(4*R^2-3*r^2)*asin(R/r)-R*(2*R^2-3*r^2)*sqrt(1-R^2/r^2))*G(i,5);
    //         afr[i] = -3.0*M_PI*sgm0/(8.0*pow(R,3.0))*(r*(4.0*pow(R,2.0)-3.0*pow(r,2.0))*asin(R/r)-R*(2.0*R*R-3.0*r*r)*sqrt(1.0-(R*R)/(r*r)))*G[i][4];
    //     }
    //     afx[i] = afr[i]*G[i][0]/r;
    //     afy[i] = afr[i]*G[i][1]/r;
    // }

    /*******************************************/
    /***** Direct Method in point to point *****/
    /*******************************************/
    // Direct Method 已測試正確
    /*
    double xi, yi, xii, yii;
    for(int i=0; i<NX; i++){
        if(G[i][4] == 1.0){
            // cout << "pass1" << endl;
            xi = G[i][0];
            yi = G[i][1];
            for(int ii=0; ii<NX; ii++){
                if(G[ii][4] == 1.0){
                    // cout << "pass2" << endl;
                    xii = G[ii][0];
                    yii = G[ii][1];
                    dxx = dx/pow(2.0, G[ii][2]-1);
                    dyy = dy/pow(2.0, G[ii][2]-1);

                    FX0IJ[i] = FX0IJ[i] + srho[ii]*xK0(xi, yi, xii, yii, dxx, dyy);
                    FY0IJ[i] = FY0IJ[i] + srho[ii]*xK0(yi, xi, yii, xii, dyy, dxx);

                    FX0IJ[i] = FX0IJ[i] + dxsrho[ii]*xKx1(xi, yi, xii, yii, dxx, dyy);
                    FX0IJ[i] = FX0IJ[i] + dysrho[ii]*xKy1(xi, yi, xii, yii, dxx, dyy);

                    FY0IJ[i] = FY0IJ[i] + dysrho[ii]*xKx1(yi, xi, yii, xii, dyy, dxx);
                    FY0IJ[i] = FY0IJ[i] + dxsrho[ii]*xKy1(yi, xi, yii, xii, dyy, dxx);
                }
            }
        }
    }
    */


    /**************************/
    /***** level to level *****/
    /**************************/
    // vector<double> LFX0IJ(FX0IJ.size(), 0.0);
    // vector<double> LFY0IJ(FY0IJ.size(), 0.0);
    vector<double> A(G.size());
    for(int i=0; i<G.size(); i++){
        A[i] = G[i][2];
    }
    vector<int> Level = find_unique(A, A.size()); // Level Correct

    // NN=length(find(G(:,3)==1)); // NN = number of coaser grid
    NN = count_if(G.begin(), G.end(), [](const vector<double>& row){
        return (row[2]-1.0)<0.000001; // return row[2]==1.0;
    }); //  NN Correct

    double start_time = clock();
    
    for(int iL=0; iL<Level.size(); iL++){
        vector<int> FP;
        FP.clear();
        for(int i=0; i<G.size(); i++){
            if(G[i][2]==Level[iL]){
                FP.push_back(i);
            }
        }

        unsigned long long NCD, NPTF, NPTC;
        for(int jL=0; jL<Level.size(); jL++){
            vector<int> DP;
            DP.clear();
            for(int j=0; j<G.size(); j++){
                if(G[j][2]==Level[jL]){
                    DP.push_back(j);
                }
            }

            if(iL<=jL){
                unsigned long long a = FP.size(), b = DP.size();
                // NCD = FP.size() * DP.size();
                NCD = a*b;
                NPTF = sqrt(NN)*pow(2.0, G[FP[0]][2]-1);
                NPTC = sqrt(NN)*pow(2.0, G[DP[0]][2]-1);

                // cout << FP.size() << "\t" << DP.size() << endl;
                // cout << fixed << setprecision(15) << NCD << "\t" << NPTF << "\t" << NPTC << endl;

                // force at coarser and surface density at the finner
                if(NCD > pow(((NPTC*log2(NPTC))),2) + pow(((NPTF*log2(NPTF))),2)){
                    // Correct
                    // cout << "FFT-1" << endl;
                    FFTfrceFL1DL2(FP, DP, G, srho, dxsrho, dysrho, LFX0IJ, LFY0IJ);
                }
                else{
                    // Correct
                    // cout << "Direct-1" << endl;
                    d3frceFL1DL2(FP, DP, G, srho, dxsrho, dysrho, LFX0IJ, LFY0IJ);
                }
            }
            else{
                NCD = FP.size()*DP.size();
                NPTF = sqrt(NN)*pow(2.0, G[FP[0]][2]-1);
                NPTC = sqrt(NN)*pow(2.0, G[DP[0]][2]-1);
                // force at finner & surface density at the coarser
                if(NCD > (pow(((NPTC*log2(NPTC))),2) + pow(((NPTF*log2(NPTF))),2))){
                    // Correct
                    // cout << "FFT-2" << endl;
                    FFTfrceFL2DL1(FP, DP, G, srho, dxsrho, dysrho, LFX0IJ, LFY0IJ);
                }
                else{
                    // cout << "Direct-2" << endl;
                    d3frceFL1DL2(FP, DP, G, srho, dxsrho, dysrho, LFX0IJ, LFY0IJ);
                }
            }
        }
    }
    
    double end_time = clock();

    for(int i=0; i<NX; i++){
        FX0IJ[i] = LFX0IJ[i];
        FY0IJ[i] = LFY0IJ[i];
    }

    vector<double> nfr(NX);
    double xx;
    double yy;
    for(int ii=0; ii<NX; ii++){
        xx = G[ii][0];
        yy = G[ii][1];
        nfr[ii] = (FX0IJ[ii]*xx + FY0IJ[ii]*yy)/sqrt(xx*xx + yy*yy);
    }

    return ((end_time-start_time)/CLOCKS_PER_SEC);
}



/*************************************************************************/
/*** Subroutine for Level L1 (force position) to L2 (density position) ***/
/*************************************************************************/

void d3frceFL1DL2(vector<int> FP, vector<int> DP, vector<vector<double>> G, vector<double> srho, vector<double> dxsrho, vector<double> dysrho, vector<double>& LFX0IJ, vector<double>& LFY0IJ){
    double xmin=-1.0;
    double xmax=1.0;
    double ymin=-1.0;
    double ymax=1.0;
    // NN=length(find(G(:,3)==1));
    int NN = count_if(G.begin(), G.end(), [](const vector<double>& row){
        return (row[2]-1.0)<0.000001; // return row[2]==1.0;
    });

    double dx = (xmax-xmin)/sqrt(NN);
    double dy=dx;

    double xi, yi, xii, yii, dxx, dyy;
    for(int i=0; i<FP.size(); i++){
        if(G[FP[i]][4]==1.0){
            xi = G[FP[i]][0];
            yi = G[FP[i]][1];
            for(int ii=0; ii<DP.size(); ii++){
                if(G[DP[ii]][4]==1.0){
                    xii = G[DP[ii]][0];
                    yii = G[DP[ii]][1];
                    dxx = dx/pow(2.0, G[DP[ii]][2]-1.0);
                    dyy = dy/pow(2.0, G[DP[ii]][2]-1.0);

                    LFX0IJ[FP[i]] = LFX0IJ[FP[i]] + srho[DP[ii]]*xK0(xi, yi, xii, yii, dxx, dyy);
                    LFY0IJ[FP[i]] = LFY0IJ[FP[i]] + srho[DP[ii]]*xK0(yi, xi, yii, xii, dyy, dxx);

                    LFX0IJ[FP[i]] = LFX0IJ[FP[i]] + dxsrho[DP[ii]]*xKx1(xi, yi, xii, yii, dxx, dyy);
                    LFX0IJ[FP[i]] = LFX0IJ[FP[i]] + dysrho[DP[ii]]*xKy1(xi, yi, xii, yii, dxx, dyy);

                    LFY0IJ[FP[i]] = LFY0IJ[FP[i]] + dysrho[DP[ii]]*xKx1(yi, xi, yii, xii, dyy, dxx);
                    LFY0IJ[FP[i]] = LFY0IJ[FP[i]] + dxsrho[DP[ii]]*xKy1(yi, xi, yii, xii, dyy, dxx);
                }
            }
        }
    }

}




/***************************************************************/
/*** Force at the coaser grid and density at the finner grid ***/
/***************************************************************/

void FFTfrceFL1DL2(vector<int> FP, vector<int> DP, vector<vector<double>> G, vector<double> FFTsrho, vector<double> FFTdxsrho, vector<double> FFTdysrho, vector<double>& LFX0IJ, vector<double>& LFY0IJ){ //LFX0IJ, LFY0IJ should be call by reference
    // FP:force position, DP:density position
    double xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0;

    // NN = number of coaser grid // Correct
    int NN = count_if(G.begin(), G.end(), [](const vector<double>& row){
        return (row[2]-1.0)<0.000001; // return row[2]==1.0;
    });
    
    // NPT = side length of finer grid if all grid are finer // Correct
    double NPT = sqrt(NN) * pow(2.0, G[DP[0]][2]-1.0);
    
    // Set density on the uniform grids at level G( DP(1) , 3 )
    // dx2, dy2, x2, y2, xxx2, yyy2 are all correct
    double dx2 = (xmax-xmin)/NPT;
    double dy2 = (ymax-ymin)/NPT;
    vector<double> x2(NPT); // x2 = xmin+0.5*dx2:dx2:xmax-0.5*dx2;
    generate(x2.begin(), x2.end(), [&, i=0]() mutable{
        return xmin+0.5*dx2 + i++*dx2;
    });
    vector<double> y2(NPT); // y2 = ymin+0.5*dy2:dy2:ymax-0.5*dy2;
    generate(y2.begin(), y2.end(), [&, i=0]() mutable{
        return ymin+0.5*dy2 + i++*dy2;
    });
    vector<double> xxx2(2*NPT); // xxx2 = xmin-(xmax-xmin)+0.5*dx2:dx2:xmax-0.5*dx2;
    generate(xxx2.begin(), xxx2.end(), [&, i=0]() mutable{
        return (xmin-(xmax-xmin)+0.5*dx2) + i++*dx2;
    });
    vector<double> yyy2(2*NPT); // yyy2 = ymin-(ymax-ymin)+0.5*dy2:dy2:ymax-0.5*dy2;
    generate(yyy2.begin(), yyy2.end(), [&, i=0]() mutable{
        return (ymin-(ymax-ymin)+0.5*dy2) + i++*dy2;
    });
    // dx2, dy2, x2, y2, xxx2, yyy2 are all correct

    vector<vector<double>> srho(2*NPT, vector<double>(2*NPT,0.0));
    vector<vector<double>> dxsrho(2*NPT, vector<double>(2*NPT,0.0));
    vector<vector<double>> dysrho(2*NPT, vector<double>(2*NPT,0.0));

    double xi, yi;
    int ii, jj;
    
    for(int i=0; i<DP.size(); i++){
        xi = G[DP[i]][0]; // xc
        yi = G[DP[i]][1]; // yc

        ii = floor((xi-xmin)/dx2)+1 -1;
        jj = floor((yi-ymin)/dy2)+1 -1;

        srho[ii][jj] = FFTsrho[DP[i]] * G[DP[i]][4];
        dxsrho[ii][jj] = FFTdxsrho[DP[i]] * G[DP[i]][4];
        dysrho[ii][jj] = FFTdysrho[DP[i]] * G[DP[i]][4];
        // cout << dysrho[ii][jj] << endl;
    }

    // Correct
    double dx = (xmax-xmin)/sqrt(NN);
    double dy = (ymax-ymin)/sqrt(NN);
    double dx1 = dx/pow(2.0, G[FP[0]][2]-1.0);
    double dy1 = dy/pow(2.0, G[FP[0]][2]-1.0);

    vector<double> x1(sqrt(NN)*pow(2.0, G[FP[0]][2]-1.0)); // x1 = xmin+0.5*dx1:dx1:xmax-0.5*dx1;
    generate(x1.begin(), x1.end(), [&, i=0]() mutable{
        return xmin+0.5*dx1 + i++*dx1;
    });
    vector<double> y1(sqrt(NN)*pow(2.0, G[FP[0]][2]-1.0)); // y1 = ymin+0.5*dy1:dy1:ymax-0.5*dy1;
    generate(y1.begin(), y1.end(), [&, i=0]() mutable{
        return ymin+0.5*dy1 + i++*dy1;
    });
    vector<double> xxx1(2*sqrt(NN)*pow(2.0, G[FP[0]][2]-1.0)); // xxx1 = xmin-(xmax-xmin)+0.5*dx1:dx1:xmax-0.5*dx1;
    generate(xxx1.begin(), xxx1.end(), [&, i=0]() mutable{
        return (xmin-(xmax-xmin)+0.5*dx1) + i++*dx1;
    });
    vector<double> yyy1(2*sqrt(NN)*pow(2.0, G[FP[0]][2]-1.0)); // yyy1 = ymin-(ymax-ymin)+0.5*dy1:dy1:ymax-0.5*dy1;
    generate(yyy1.begin(), yyy1.end(), [&, i=0]() mutable{
        return (ymin-(ymax-ymin)+0.5*dy1) + i++*dy1;
    });


    int DL = G[DP[0]][2]-G[FP[0]][2]; // DL = layer differnece between finer and coaser. Correct
    int WD = pow(2.0, DL); // WD = side grid number difference between finer and coaser layer. Correct

    int NPTC = sqrt(NN) * pow(2.0, G[FP[0]][2]-1.0); // NPTC = side grid number for caoser grid. Correct
    int idx, jdy;

    vector<vector<double>> Gx(2*NPTC, vector<double>(2*NPTC, 0.0));
    vector<vector<double>> xKx(2*NPTC, vector<double>(2*NPTC, 0.0));
    vector<vector<double>> xKy(2*NPTC, vector<double>(2*NPTC, 0.0));

    vector<vector<double>> Gy(2*NPTC, vector<double>(2*NPTC, 0.0));
    vector<vector<double>> yKx(2*NPTC, vector<double>(2*NPTC, 0.0));
    vector<vector<double>> yKy(2*NPTC, vector<double>(2*NPTC, 0.0));

    vector<vector<double>> FX0IJ(NPTC, vector<double>(NPTC, 0.0));
    vector<vector<double>> FY0IJ(NPTC, vector<double>(NPTC, 0.0));

    for(int iiDL=0; iiDL<WD; iiDL++){
        for(int jjDL=0; jjDL<WD; jjDL++){

            // Correct
            idx = NPT+iiDL+1;
            jdy = NPT+jjDL+1;

            // Correct
            for(int i=0; i<2*NPTC; i++){
                for(int j=0; j<2*NPTC; j++){
                    Gx[i][j]  = -xK0(xxx2[idx-1]-(WD-1)*0.5*dx2, yyy2[jdy-1]-(WD-1)*0.5*dy2, xxx2[WD*(i+1-1)+1 -1], yyy2[WD*(j+1-1)+1 -1], dx2, dy2);
                    xKx[i][j] = xKx1(xxx2[idx-1]-(WD-1)*0.5*dx2, yyy2[jdy-1]-(WD-1)*0.5*dy2, xxx2[WD*(i+1-1)+1 -1], yyy2[WD*(j+1-1)+1 -1], dx2, dy2);
                    xKy[i][j] = xKy1(xxx2[idx-1]-(WD-1)*0.5*dx2, yyy2[jdy-1]-(WD-1)*0.5*dy2, xxx2[WD*(i+1-1)+1 -1], yyy2[WD*(j+1-1)+1 -1], dx2, dy2);

                    Gy[i][j]  = -xK0(yyy2[jdy-1]-(WD-1)*0.5*dy2, xxx2[idx-1]-(WD-1)*0.5*dx2, yyy2[WD*(j+1-1)+1 -1], xxx2[WD*(i+1-1)+1 -1], dy2, dx2);
                    yKx[i][j] = xKy1(yyy2[jdy-1]-(WD-1)*0.5*dy2, xxx2[idx-1]-(WD-1)*0.5*dx2, yyy2[WD*(j+1-1)+1 -1], xxx2[WD*(i+1-1)+1 -1], dy2, dx2);
                    yKy[i][j] = xKx1(yyy2[jdy-1]-(WD-1)*0.5*dy2, xxx2[idx-1]-(WD-1)*0.5*dx2, yyy2[WD*(j+1-1)+1 -1], xxx2[WD*(i+1-1)+1 -1], dy2, dx2);
                }
            }
            
            // Correct
            vector<vector<double>> sub_srho = extract_submatrix(srho, iiDL, WD, jjDL, WD);
            vector<vector<double>> sub_dxsrho = extract_submatrix(dxsrho, iiDL, WD, jjDL, WD);
            vector<vector<double>> sub_dysrho = extract_submatrix(dysrho, iiDL, WD, jjDL, WD);

            // Correct
            vector<vector<double>> FX0IJ00 = conv2d(Gx, sub_srho);
            vector<vector<double>> FX0IJX00 = conv2d(xKx, sub_dxsrho);
            vector<vector<double>> FX0IJY00 = conv2d(xKy, sub_dysrho);

            vector<vector<double>> FY0IJ00 = conv2d(Gy, sub_srho);
            vector<vector<double>> FY0IJX00 = conv2d(yKx, sub_dxsrho);
            vector<vector<double>> FY0IJY00 = conv2d(yKy, sub_dysrho);

            // Correct
            double dweight = 1.0;
            for(int i=0; i<NPTC; i++){
                for(int j=0; j<NPTC; j++){
                    FX0IJ[i][j] = FX0IJ00[i+NPTC][j+NPTC] + dweight*FX0IJX00[i+NPTC][j+NPTC] + dweight*FX0IJY00[i+NPTC][j+NPTC];
                    FY0IJ[i][j] = FY0IJ00[i+NPTC][j+NPTC] + dweight*FY0IJX00[i+NPTC][j+NPTC] + dweight*FY0IJY00[i+NPTC][j+NPTC];
                }
            }

            // Correct
            for(int i=0; i<FP.size(); i++){
                if(G[FP[i]][4]==1.0){
                    xi = G[FP[i]][0];
                    yi = G[FP[i]][1];
                    ii = floor((xi-xmin)/dx1);
                    jj = floor((yi-ymin)/dy1);
                    LFX0IJ[FP[i]] += FX0IJ[ii][jj]; // return LFX0IJ
                    LFY0IJ[FP[i]] += FY0IJ[ii][jj]; // return LFY0IJ
                }
            }
            
        }
    }

}



/***************************************************************/
/*** Force at the finner grid and density at the coaser grid ***/
/***************************************************************/
// void FFTfrceFL2DL1(double* FP, double* DP, double** G, double* FFTsrho, double* FFTdxsrho, double* FFTdysrho, double* LFX0IJ, double* LFY0IJ){
void FFTfrceFL2DL1(vector<int> FP, vector<int> DP, vector<vector<double>> G, vector<double> FFTsrho, vector<double> FFTdxsrho, vector<double> FFTdysrho, vector<double>& LFX0IJ, vector<double>& LFY0IJ){
    // FP:force position, DP:density position
    double xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0;
    // NN=length(find(G(:,3)==1)); // NN=number of coaser grid
    int NN = count_if(G.begin(), G.end(), [](const vector<double>& row){
        return (row[2]-1.0)<0.000001; // return row[2]==1.0;
    });
    // NPTC = sqrt(NN)*2^(G(DP(1),3)-1);
    double NPTC = sqrt(NN) * pow(2.0, G[DP[0]][2]-1.0);


    // Set density on the uniform grids at level G[DP[0]][3]
    double dx2 = (xmax-xmin)/NPTC;
    double dy2 = (ymax-ymin)/NPTC;

    vector<double> x2(NPTC); // x2 = xmin+0.5*dx2 : dx2 : xmax-0.5*dx2;
    generate(x2.begin(), x2.end(), [&, i=0]() mutable{
        return xmin+0.5*dx2 + i++*dx2;
    });
    vector<double> y2(NPTC); // y2 = ymin+0.5*dy2 : dy2 : ymax-0.5*dy2;
    generate(y2.begin(), y2.end(), [&, i=0]() mutable{
        return ymin+0.5*dy2 + i++*dy2;
    });
    vector<double> xxx2(2*NPTC); // xxx2 = xmin-(xmax-xmin)+0.5*dx2 : dx2 : xmax-0.5*dx2;
    generate(xxx2.begin(), xxx2.end(), [&, i=0]() mutable{
        return (xmin-(xmax-xmin)+0.5*dx2) + i++*dx2;
    });
    vector<double> yyy2(2*NPTC); // yyy2 = ymin-(ymax-ymin)+0.5*dy2 : dy2 : ymax-0.5*dy2;
    generate(yyy2.begin(), yyy2.end(), [&, i=0]() mutable{
        return (ymin-(ymax-ymin)+0.5*dy2) + i++*dy2;
    });

    vector<vector<double>> srho(2*NPTC, vector<double>(2*NPTC,0.0));
    vector<vector<double>> dxsrho(2*NPTC, vector<double>(2*NPTC,0.0));
    vector<vector<double>> dysrho(2*NPTC, vector<double>(2*NPTC,0.0));

    double xi, yi;
    int ii, jj;
    for(int i=0; i<DP.size(); i++){
        xi = G[DP[i]][0]; // xc
        yi = G[DP[i]][1]; // yc

        ii = floor((xi-xmin)/dx2)+1 -1;
        jj = floor((yi-ymin)/dy2)+1 -1;

        srho[ii][jj] = FFTsrho[DP[i]] * G[DP[i]][4];
        dxsrho[ii][jj] = FFTdxsrho[DP[i]] * G[DP[i]][4];
        dysrho[ii][jj] = FFTdysrho[DP[i]] * G[DP[i]][4];
    }

    double NPTF = sqrt(NN) * pow(2.0, (G[FP[0]][2]-1.0));
    double dx1 = (xmax-xmin)/NPTF;
    double dy1 = (ymax-ymin)/NPTF;

    vector<double> x1(sqrt(NN)*pow(2.0, G[FP[0]][2]-1.0)); // x1 = xmin+0.5*dx1:dx1:xmax-0.5*dx1;
    generate(x1.begin(), x1.end(), [&, i=0]() mutable{
        return xmin+0.5*dx1 + i++*dx1;
    });
    vector<double> y1(sqrt(NN)*pow(2.0, G[FP[0]][2]-1.0)); // y1 = ymin+0.5*dy1:dy1:ymax-0.5*dy1;
    generate(y1.begin(), y1.end(), [&, i=0]() mutable{
        return ymin+0.5*dy1 + i++*dy1;
    });
    vector<double> xxx1(2*sqrt(NN)*pow(2.0, G[FP[0]][2]-1.0)); // xxx1 = xmin-(xmax-xmin)+0.5*dx1:dx1:xmax-0.5*dx1;
    generate(xxx1.begin(), xxx1.end(), [&, i=0]() mutable{
        return (xmin-(xmax-xmin)+0.5*dx1) + i++*dx1;
    });
    vector<double> yyy1(2*sqrt(NN)*pow(2.0, G[FP[0]][2]-1.0)); // yyy1 = ymin-(ymax-ymin)+0.5*dy1:dy1:ymax-0.5*dy1;
    generate(yyy1.begin(), yyy1.end(), [&, i=0]() mutable{
        return (ymin-(ymax-ymin)+0.5*dy1) + i++*dy1;
    });

    int DL = G[FP[0]][2]-G[DP[0]][2]; // DL = layer differnece between finer and coaser
    int WD = pow(2.0, DL); // WD = side grid number difference between finer and coaser layer
    int kkk = 0;

    vector<vector<double>> Gx(2*NPTC, vector<double>(2*NPTC, 0.0));
    vector<vector<double>> Gy(2*NPTC, vector<double>(2*NPTC, 0.0));
    vector<vector<double>> xKx(2*NPTC, vector<double>(2*NPTC, 0.0));
    vector<vector<double>> xKy(2*NPTC, vector<double>(2*NPTC, 0.0));
    vector<vector<double>> yKx(2*NPTC, vector<double>(2*NPTC, 0.0));
    vector<vector<double>> yKy(2*NPTC, vector<double>(2*NPTC, 0.0));

    vector<vector<double>> FX0IJ(NPTC, vector<double>(NPTC, 0.0));
    vector<vector<double>> FY0IJ(NPTC, vector<double>(NPTC, 0.0));
    double idx, jdy;


    for(int iiDL=0; iiDL<WD; iiDL++){
        for(int jjDL=0; jjDL<WD; jjDL++){

            idx = NPTF + WD - (iiDL +1 -1);
            jdy = NPTF + WD - (jjDL +1 -1);


            for(int i=0; i<NPTC*2; i++){
                for(int j=0; j<NPTC*2; j++){
                    Gx[i][j]  = -xK0(xxx1[idx-1], yyy1[jdy-1], xxx2[i], yyy2[j], dx2, dy2);
                    xKx[i][j] = xKx1(xxx1[idx-1], yyy1[jdy-1], xxx2[i], yyy2[j], dx2, dy2);
                    xKy[i][j] = xKy1(xxx1[idx-1], yyy1[jdy-1], xxx2[i], yyy2[j], dx2, dy2);

                    Gy[i][j]  = -xK0(yyy1[jdy-1], xxx1[idx-1], yyy2[j], xxx2[i], dy2, dx2);
                    yKx[i][j] = xKy1(yyy1[jdy-1], xxx1[idx-1], yyy2[j], xxx2[i], dy2, dx2);
                    yKy[i][j] = xKx1(yyy1[jdy-1], xxx1[idx-1], yyy2[j], xxx2[i], dy2, dx2);
                }
            }

            // vector<vector<double>> conv2d(const vector<vector<double>>& K, const vector<vector<double>>& rho);
            vector<vector<double>> FX0IJ00 = conv2d(Gx, srho);
            vector<vector<double>> FX0IJX00 = conv2d(xKx, dxsrho);
            vector<vector<double>> FX0IJY00 = conv2d(xKy, dysrho);

            vector<vector<double>> FY0IJ00 = conv2d(Gy, srho);
            vector<vector<double>> FY0IJX00 = conv2d(yKx, dxsrho);
            vector<vector<double>> FY0IJY00 = conv2d(yKy, dysrho);

            double dweight = 1.0;
            for(int i=0; i<NPTC; i++){
                for(int j=0; j<NPTC; j++){
                    FX0IJ[i][j] = FX0IJ00[i+NPTC][j+NPTC] + dweight*FX0IJX00[i+NPTC][j+NPTC] + dweight*FX0IJY00[i+NPTC][j+NPTC];
                    FY0IJ[i][j] = FY0IJ00[i+NPTC][j+NPTC] + dweight*FY0IJX00[i+NPTC][j+NPTC] + dweight*FY0IJY00[i+NPTC][j+NPTC];
                }
            }

            for(int i=0; i<FP.size(); i++){
                if(G[FP[i]][4] == 1.0){
                    xi = G[FP[i]][0];
                    yi = G[FP[i]][1];
                    int ii = floor((xi-xmin)/dx1)+1;
                    int jj = floor((yi-ymin)/dy1)+1;

                    if (((ii-1)%WD == (iiDL)) && ((jj-1)%WD == (jjDL))) {
                        int iii = floor((xi - xmin) /dx1/WD) +1;
                        int jjj = floor((yi - ymin) /dy1/WD) +1;
                        
                        LFX0IJ[FP[i]] = LFX0IJ[FP[i]] + FX0IJ[iii-1][jjj-1];
                        LFY0IJ[FP[i]] = LFY0IJ[FP[i]] + FY0IJ[iii-1][jjj-1];

                        kkk++;
                    }
                    // LFX0IJ[FP[i]] += FX0IJ[ii][jj]; // return LFX0IJ
                    // LFY0IJ[FP[i]] += FY0IJ[ii][jj]; // return LFY0IJ
                }
            }
        }
    }

}



// unique() 
vector<int> find_unique(vector<double> &arr, int n){
    vector<int> B (arr.begin(), arr.end());
    sort(B.begin(), B.end());
    auto last = unique(B.begin(), B.end());
    B.erase(last, B.end());
    return B;
}


vector<vector<double>> conv2d(const vector<vector<double>>& K, const vector<vector<double>>& rho){

    int rows = K.size();
    int cols = K[0].size();

    fftw_complex *in_K = fftw_alloc_complex(rows*cols);
    fftw_complex *out_K = fftw_alloc_complex(rows*cols);

    fftw_complex *in_rho = fftw_alloc_complex(rows*cols);
    fftw_complex *out_rho = fftw_alloc_complex(rows*cols);

    fftw_complex *force_FFT = fftw_alloc_complex(rows*cols);
    fftw_complex *force = fftw_alloc_complex(rows*cols);

    // execute fft2 on both K and rho
    for(int i=0; i<rows; ++i){
        for(int j=0; j<cols; ++j){
            in_K[i * cols + j][0] = K[i][j];
            in_K[i * cols + j][1] = 0.0;

            in_rho[i * cols + j][0] = rho[i][j];
            in_rho[i * cols + j][1] = 0.0;
        }
    }

    fftw_plan p1 = fftw_plan_dft_2d(rows, cols, in_K, out_K, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p1);
    fftw_plan p2 = fftw_plan_dft_2d(rows, cols, in_rho, out_rho, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p2);

    // multiply
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            force_FFT[i*cols + j][0] = out_K[i*cols + j][0] * out_rho[i*cols + j][0] - out_K[i*cols + j][1] * out_rho[i*cols + j][1];
            force_FFT[i*cols + j][1] = out_K[i*cols + j][0] * out_rho[i*cols + j][1] + out_K[i*cols + j][1] * out_rho[i*cols + j][0];
        }
    }

    // execute ifft2 
    fftw_plan p3 = fftw_plan_dft_2d(rows, cols, force_FFT, force, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p3);

    // take the real part to be the output result
    vector<vector<double>> result(rows, vector<double>(cols));
    for(int i=0; i<rows; ++i){
        for(int j=0; j<cols; ++j){
            result[i][j] = force[i*cols + j][0]/(rows*cols);
        }
    }

    // free the memories
    fftw_destroy_plan(p1);
    fftw_free(in_K);
    fftw_free(out_K);

    fftw_destroy_plan(p2);
    fftw_free(in_rho);
    fftw_free(out_rho);

    fftw_free(force_FFT);
    fftw_free(force);
    fftw_destroy_plan(p3);

    return result;
}


vector<vector<double>> extract_submatrix(const vector<vector<double>>& arr, int row_start, int row_stride, int col_start, int col_stride) {
    int rows = (arr.size() - row_start + row_stride -1) / row_stride ;
    int cols = (arr[0].size() - col_start + col_stride - 1) / col_stride ;
    // cout << rows << " " << cols << endl;
    // cout << arr.size() << " " << arr[0].size() << endl;
    // cout << (row_start) + (rows) * row_stride<< " " << (col_start) + (cols) * col_stride << endl;
    vector<vector<double>> result(rows, vector<double>(cols, 0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = arr[(row_start) + i * row_stride ][(col_start) + j * col_stride ];
        }
    }

    return result;
}


vector<vector<complex<double>>> fft2(const vector<vector<double>>& input) {
    int rows = input.size();
    int cols = input[0].size();

    fftw_complex *in = fftw_alloc_complex(rows*cols);
    fftw_complex *out = fftw_alloc_complex(rows*cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            in[i * cols + j][0] = input[i][j];
            in[i * cols + j][1] = 0.0;
        }
    }

    fftw_plan plan = fftw_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    vector<vector<complex<double>>> result(rows, vector<complex<double>>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = {out[i * cols + j][0], out[i * cols + j][1]};
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return result;
}

double createSpiral(int N){
    double xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0;
    double dxx = (xmax-xmin)/N;
    double dyy = dxx;
    vector<double> x(N); // x = xmin+0.5*dxx : dxx : xmax-0.5*dxx;
    generate(x.begin(), x.end(), [&, i=0]() mutable{
        return xmin+0.5*dxx + i++*dxx;
    });
    vector<double> y(N); // x = xmin+0.5*dxx : dxx : xmax-0.5*dxx;
    generate(y.begin(), y.end(), [&, i=0]() mutable{
        return ymin+0.5*dyy + i++*dyy;
    });

    vector<double> srhoSort(N,0.0);
    double s = 0.0, r = 0.0;
    double term1, term2, term3, term4;

    for(int i=0; i<N; i++){
        r = sqrt(x[i]*x[i] + y[i]*y[i]);
        complex<double> z(x[i], y[i]);
        s = 0;
        s = arg(z);
        term3 = exp(-abs(r - s/3)*2);
        term4 = exp(-abs(r - s/3 - M_PI/3)*2);
        srhoSort[i] = 10 * (term3 + term4) / (2 + r);
    }

    priority_queue<double, vector<double>, greater<double>> minHeap;
    for(double value : srhoSort){
        minHeap.push(value);
        if(minHeap.size() > (N/4)){
            minHeap.pop();
        }
    }
    // cout << "threshold = " << minHeap.top() << endl;

    return  minHeap.top();
}

/****************************************************************/
/**************************  Compute K  *************************/
/****************************************************************/
double xK0(double xi, double yj, double xii, double yjj, double dx, double dy){
    double u = 0.;
    double eps = 1.0E-12;
    double sx, sy;
    // sx = fabs(sign(xii+0.5*dx-xi));
    sx = fabs(xii+0.5*dx-xi);
    if(sx > eps){
        u = u-log((yjj+0.5*dy-yj)+sqrt((xii+0.5*dx-xi)*(xii+0.5*dx-xi)+(yjj+0.5*dy-yj)*(yjj+0.5*dy-yj)));
        u = u+log((yjj-0.5*dy-yj)+sqrt((xii+0.5*dx-xi)*(xii+0.5*dx-xi)+(yjj-0.5*dy-yj)*(yjj-0.5*dy-yj)));
    }
    else{
        if(fabs(yjj+0.5*dy-yj)>eps){
            sy = sign(yjj+0.5*dy-yj);
            u = u-sy*log(abs(yjj+0.5*dy-yj));
        }
        if(fabs(yjj-0.5*dy-yj)>eps){
            sy = sign(yjj-0.5*dy-yj);
            u = u+sy*log(abs(yjj-0.5*dy-yj));
        }
    }

    // sx = fabs(sign(xii-0.5*dx-xi));
    sx = fabs(xii-0.5*dx-xi);
    if(sx > eps){
        u = u+log((yjj+0.5*dy-yj)+sqrt((xii-0.5*dx-xi)*(xii-0.5*dx-xi)+(yjj+0.5*dy-yj)*(yjj+0.5*dy-yj)));
        u = u-log((yjj-0.5*dy-yj)+sqrt((xii-0.5*dx-xi)*(xii-0.5*dx-xi)+(yjj-0.5*dy-yj)*(yjj-0.5*dy-yj)));
    }
    else{
        if(fabs(yjj+0.5*dy-yj)>eps){
            sy = sign(yjj+0.5*dy-yj);
            u = u+sy*log(abs(yjj+0.5*dy-yj));
        }
        if(fabs(yjj-0.5*dy-yj)>eps){
            sy = sign(yjj-0.5*dy-yj);
            u = u-sy*log(abs(yjj-0.5*dy-yj));
        }
    }

    return u;
}


double xKx1(double xi, double yj, double xii, double yjj, double dx, double dy){
    double u;
    double eps = 1.0E-12;
    u = (xi-xii) * xK0(xi,yj,xii,yjj,dx,dy);
    double sy = fabs(sign(yjj+0.5*dy-yj));
    if(sy > eps){
        u= u+(yjj+0.5*dy-yj)*log((xii+0.5*dx-xi)+sqrt((xii+0.5*dx-xi)*(xii+0.5*dx-xi)+(yjj+0.5*dy-yj)*(yjj+0.5*dy-yj)));
        u= u-(yjj+0.5*dy-yj)*log((xii-0.5*dx-xi)+sqrt((xii-0.5*dx-xi)*(xii-0.5*dx-xi)+(yjj+0.5*dy-yj)*(yjj+0.5*dy-yj)));
    }

    sy = fabs(sign(yjj-0.5*dy-yj));
    if(sy > eps){
        u = u-(yjj-0.5*dy-yj)*log((xii+0.5*dx-xi)+sqrt((xii+0.5*dx-xi)*(xii+0.5*dx-xi)+(yjj-0.5*dy-yj)*(yjj-0.5*dy-yj)));
        u = u+(yjj-0.5*dy-yj)*log((xii-0.5*dx-xi)+sqrt((xii-0.5*dx-xi)*(xii-0.5*dx-xi)+(yjj-0.5*dy-yj)*(yjj-0.5*dy-yj)));
    }

    return u;
}


double xKy1(double xi, double yj, double xii, double yjj, double dx, double dy){
    double u;
    u = (yj-yjj) * xK0(xi,yj,xii,yjj,dx,dy);
    u = u - sqrt((xii+0.5*dx-xi)*(xii+0.5*dx-xi) + (yjj+0.5*dy-yj)*(yjj+0.5*dy-yj));
    u = u + sqrt((xii-0.5*dx-xi)*(xii-0.5*dx-xi) + (yjj+0.5*dy-yj)*(yjj+0.5*dy-yj));
    u = u + sqrt((xii+0.5*dx-xi)*(xii+0.5*dx-xi) + (yjj-0.5*dy-yj)*(yjj-0.5*dy-yj));
    u = u - sqrt((xii-0.5*dx-xi)*(xii-0.5*dx-xi) + (yjj-0.5*dy-yj)*(yjj-0.5*dy-yj));

    return u;
}


int sign(double x){
    if(x>0)
        return 1;
    else if(x<0)
        return -1;
    else
        return 0;
}
