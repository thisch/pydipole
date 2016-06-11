#include <vector>
#include <complex>
#include <iostream>
#include "boost/multi_array.hpp"
#include <omp.h>


#include "field.hpp"

using namespace std;

typedef boost::multi_array<complex<double>, 3> restype;
typedef boost::multi_array<double, 2> ffrestype;

static double c = 299792458.;
static double mu0 = 4*M_PI*1e-7;
static double eps0 = 1./(mu0*c*c);
static double Z = mu0*c;


ffrestype dipole_radiant_intensity(boost::multi_array<double, 2>& T,
                                   boost::multi_array<double, 2>& P,
                                   boost::multi_array<double, 2>& p,
                                   boost::multi_array<double, 2>& R,
                                   std::vector<double>& phases, //1d
                                   double k)
{
    // computes the radiant intensity (radiant flux per solid angle) of a
    // set of oscillating dipoles.

    // Note: we use the following time dependence of the phasors: exp(-i*w*t)

    // Parameters
    // ----------
    // T: real NxM matrix (observation thetas)
    // P: real NxM matrix (observation phis)
    // p: real Lx3 matrix
    // R: real Lx3 matrix
    // phases: real length L vector
    // k: real
    //    wavevector(scalar)
    // res: NxM real array
    //    radiant intensity

    const int N = T.shape()[0];
    const int M = T.shape()[1];
    const int L = p.shape()[0]; // number of dipoles

    ffrestype res(boost::extents[N][M]);

    const double prefac = k*k*k*k/(32*M_PI*M_PI*eps0*eps0*Z);

#pragma omp parallel for //shared(res, r, R, p)
    for (int i=0; i < N; i++) {
        for (int j=0; j < M; j++) {
            // todo move this out of the c++code ??
            const vector<double> r = {
                sin(T[i][j])*cos(P[i][j]),
                sin(T[i][j])*sin(P[i][j]),
                cos(T[i][j])};

            // TODO sum over dipoles
            // p tot
            vector<complex<double>> p_vec(3, 0);
            for (int d=0; d < L; ++d) {
                double rinR = 0.;
                for (int g=0; g < 3; ++g) {
                    rinR += r[g]*R[d][g];
                }
                auto expfac = exp(complex<double>(0, -(k*rinR + phases[d])));
                for (int g=0; g < 3; ++g) {
                    p_vec[g] += p[d][g]*expfac;
                }
            }

            // r x p
            const vector<complex<double>> r_cross_p = {
                r[1]*p_vec[2] - r[2]*p_vec[1],
                -r[0]*p_vec[2] + r[2]*p_vec[0],
                r[0]*p_vec[1] - r[1]*p_vec[0]
            };

            // (r x p) x r
            const vector<complex<double>> tmpres = {
                r_cross_p[1]*r[2] - r_cross_p[2]*r[1],
                -r_cross_p[0]*r[2] + r_cross_p[2]*r[0],
                r_cross_p[0]*r[1] - r_cross_p[1]*r[0]
            };

            complex<double> rint = 0.;
            for (int l=0; l < 3; l++) {
                rint += tmpres[l]*conj(tmpres[l]);
            }
            res[i][j] = prefac*real(rint);
        }
    }
    return res;
}

restype dipole_field_ff(boost::multi_array<double, 3>& r,
                        boost::multi_array<double, 2>& p,
                        boost::multi_array<double, 2>& R,
                        std::vector<double>& phases, //1d
                        double k, double t, bool calc_H)
{
    // computes the vectorial E or H field of a set of oscillating dipoles in the
    // far-field region

    // Note: we use the following time dependence of the phasors: exp(-i*w*t)

    // Parameters
    // ----------
    // r: real NxMx3 matrix
    // p: real Lx3 matrix
    // R: real Lx3 matrix
    // phases: real length L vector
    // k: real
    //    wavevector(scalar)
    // t: real
    //    time
    // res: NxMx3 complex array
    //    E-Field
    // calc_H: calculate H field if true otherwise E field

    int N = r.shape()[0];
    int M = r.shape()[1];
    int L = p.shape()[0]; // number of dipoles

    restype res(boost::extents[r.shape()[0]][r.shape()[1]][3]);

    // cout << "omp_get_max_threads() " << omp_get_max_threads() << endl;
    // cout << "omp_get_num_threads() " << omp_get_num_threads() << endl;

#pragma omp parallel for //shared(res, r, R, p)
    for (int i=0; i < N; i++) {
        // cout << "omp_get_num_threads() " << omp_get_num_threads() << " "
             // << omp_get_thread_num()<< endl;

        for (int j=0; j < M; j++) {
            // cout << "i " << i << " j " << j << endl;
            for (int l=0; l < 3; l++) {
                res[i][j][l] = 0.;
            }
            // cout << (r[i][j] - R[0]) << endl;
            for (int d=0; d < L; ++d) {
                double magr = 0.;
                vector<double> r_vec(3);
                vector<double> p_vec(3);

                // TODO move magr part one level up
                double rinp = 0.;
                for (int g=0; g < 3; ++g) {
                    p_vec[g] = p[d][g];
                    rinp += r[i][j][g]*R[d][g];
                    r_vec[g] = r[i][j][g];
                    magr += r[i][j][g]*r[i][j][g];
                }
                magr = sqrt(magr);
                for (int g=0; g < 3; ++g) {
                    r_vec[g] /= magr;
                }

                const double krinp = k*rinp/magr;
                auto expfac = exp(complex<double>(0, (k*magr - krinp) - phases[d]));
                auto efac = k*k/magr;
                if (calc_H) {
                    // TODO this code is not tested
                    vector<double> r_cross_p(3);
                    // r x p  (note r is not a unit vector)
                    r_cross_p[0] = r_vec[1]*p_vec[2] - r_vec[2]*p_vec[1];
                    r_cross_p[1] = -r_vec[0]*p_vec[2] + r_vec[2]*p_vec[0];
                    r_cross_p[2] = r_vec[0]*p_vec[1] - r_vec[1]*p_vec[0];
                    expfac *= c/(4*M_PI);
                    for (int l=0; l < 3; l++) {
                        res[i][j][l] += efac * expfac * r_cross_p[l];
                    }
                }
                else {
                    vector<double> r_cross_p(3);
                    vector<double> rpcp(3);

                    // r x p  (note r is not a unit vector)
                    r_cross_p[0] = r_vec[1]*p_vec[2] - r_vec[2]*p_vec[1];
                    r_cross_p[1] = -r_vec[0]*p_vec[2] + r_vec[2]*p_vec[0];
                    r_cross_p[2] = r_vec[0]*p_vec[1] - r_vec[1]*p_vec[0];

                    // (r x p) x r  (note r is not a unit vector)
                    rpcp[0] = r_cross_p[1]*r_vec[2] - r_cross_p[2]*r_vec[1];
                    rpcp[1] = -r_cross_p[0]*r_vec[2] + r_cross_p[2]*r_vec[0];
                    rpcp[2] = r_cross_p[0]*r_vec[1] - r_cross_p[1]*r_vec[0];
                    expfac /= 4*M_PI*eps0;
                    for (int l=0; l < 3; l++) {
                        res[i][j][l] += efac * expfac * rpcp[l];
                    }
                }
            }
        }
    }
    return res;
}

restype dipole_field_general(boost::multi_array<double, 3>& r,
                            boost::multi_array<double, 2>& p,
                            boost::multi_array<double, 2>& R,
                            std::vector<double>& phases, //1d
                            double k, double t, bool calc_H)
{
    // computes the vectorial E or H  field of a set of oscillating dipoles

    // Note: we use the following time dependence of the phasors: exp(-i*w*t)

    // Parameters
    // ----------
    // r: real NxMx3 matrix
    // p: real Lx3 matrix
    // R: real Lx3 matrix
    // phases: real length L vector
    // k: real
    //    wavevector(scalar)
    // t: real
    //    time
    // calc_H: calculate H field if true otherwise E field
    //
    // Returns
    // -------
    // res: NxMx3 complex array
    //    E or H-Field

    int N = r.shape()[0];
    int M = r.shape()[1];
    int L = p.shape()[0]; // number of dipoles

    double prefac = calc_H ? k*k*c/(4*M_PI) : 1./(4*M_PI*eps0);

    restype res(boost::extents[r.shape()[0]][r.shape()[1]][3]);

    // cout << "omp_get_max_threads() " << omp_get_max_threads() << endl;
    // cout << "omp_get_num_threads() " << omp_get_num_threads() << endl;

#pragma omp parallel for //shared(res, r, R, p)
    for (int i=0; i < N; i++) {
        // cout << "omp_get_num_threads() " << omp_get_num_threads() << " "
             // << omp_get_thread_num()<< endl;

        for (int j=0; j < M; j++) {
            // cout << "i " << i << " j " << j << endl;
            // double r0 = r[i][j][0];
            // double r1 = r[i][j][1];
            // double r2 = r[i][j][2];
            // cout << "0: " << r[i][j][0] << " 1: " << r[i][j][1] << " 2: "
            //      << r[i][j][2] << " NORM " << r0*r0+r1*r1+r2*r2 << endl;

            for (int l=0; l < 3; l++) {
                res[i][j][l] = 0.;
            }

            for (int d=0; d < L; ++d) {
                double magrprime = 0.;
                vector<double> rprime_vec(3);
                vector<double> p_vec(3);

                for (int g=0; g < 3; ++g) {
                    double rprime = r[i][j][g] - R[d][g];
                    rprime_vec[g] = rprime;
                    p_vec[g] = p[d][g];
                    magrprime += rprime*rprime;
                }
                magrprime = sqrt(magrprime);
                for (int g=0; g < 3; ++g) {
                    rprime_vec[g] /= magrprime;
                }

                if (calc_H) {
                    // H FIELD
                    const double krp = k*magrprime;
                    // todo long double?
                    auto expfac = exp(complex<double>(0, krp - (k*c*t + phases[d])));

                    vector<double> rprime_cross_p(3);
                    // r' x p (r' ist not a unit vector !)
                    rprime_cross_p[0] = rprime_vec[1]*p_vec[2] - rprime_vec[2]*p_vec[1];
                    rprime_cross_p[1] = -rprime_vec[0]*p_vec[2] + rprime_vec[2]*p_vec[0];
                    rprime_cross_p[2] = rprime_vec[0]*p_vec[1] - rprime_vec[1]*p_vec[0];

                    for (int l=0; l < 3; l++) {
                        res[i][j][l] += (prefac/magrprime) * expfac * (1. - 1./complex<double>(0, krp)) * rprime_cross_p[l];
                    }
                } else {
                    // E FIELD
                    const double krp = k*magrprime;
                    auto expfac = exp(complex<double>(0, krp - (k*c*t + phases[d])));

                    auto e1fac = k*k/magrprime;
                    auto e2fac = complex<double>(1, -krp)/(magrprime*magrprime*magrprime);

                    vector<double> rprime_cross_p(3);
                    vector<double> rpcp(3);
                    vector<double> rrpp(3);

                    // r' x p   (note that r' is not a unit vector)
                    rprime_cross_p[0] = rprime_vec[1]*p_vec[2] - rprime_vec[2]*p_vec[1];
                    rprime_cross_p[1] = -rprime_vec[0]*p_vec[2] + rprime_vec[2]*p_vec[0];
                    rprime_cross_p[2] = rprime_vec[0]*p_vec[1] - rprime_vec[1]*p_vec[0];

                    // (r' x p) x r'  (note that r' is not a unit vector)
                    rpcp[0] = rprime_cross_p[1]*rprime_vec[2] - rprime_cross_p[2]*rprime_vec[1];
                    rpcp[1] = -rprime_cross_p[0]*rprime_vec[2] + rprime_cross_p[2]*rprime_vec[0];
                    rpcp[2] = rprime_cross_p[0]*rprime_vec[1] - rprime_cross_p[1]*rprime_vec[0];

                    // r' . p
                    double rpinp = 0;
                    for (int g=0; g < 3; ++g) {
                        rpinp += p_vec[g]*rprime_vec[g];
                    }

                    // 3*rhat' (rhat' . p) - p
                    for (int g=0; g < 3; ++g) {
                        rrpp[g] = 3*rprime_vec[g]*rpinp - p_vec[g];
                    }

                    for (int l=0; l < 3; l++) {
                        res[i][j][l] += prefac * expfac * (e1fac * rpcp[l] + e2fac*rrpp[l]);
                    }
                }
            }
        }
    }
    return res;
}


vector<vector<double>>
dipole_radiant_intensity_wrapper(vector<vector<double>> T,
                                 vector<vector<double>> P,
                                 vector<vector<double>> p,
                                 vector<vector<double>> R,
                                 vector<double> phases,
                                 double k) {
    size_t N1 = T.size();
    size_t N2 = T[0].size();

    // THETA
    auto T_ma = boost::multi_array<double, 2>(boost::extents[N1][N2]);
    for (size_t i=0; i<N1; ++i)
        for (size_t j=0; j<N2; ++j)
            T_ma[i][j] = T[i][j];

    // PHI
    auto P_ma = boost::multi_array<double, 2>(boost::extents[N1][N2]);
    for (size_t i=0; i<N1; ++i)
        for (size_t j=0; j<N2; ++j)
            P_ma[i][j] = P[i][j];

    // dipole moments
    N1 = p.size(); //
    N2 = p[0].size();
    auto p_ma = boost::multi_array<double, 2>(boost::extents[N1][N2]);
    for (size_t i=0; i<N1; ++i)
        for (size_t j=0; j<N2; ++j)
            p_ma[i][j] = p[i][j];

    // dipole positions
    N1 = R.size();
    N2 = R[0].size();
    auto R_ma = boost::multi_array<double, 2>(boost::extents[N1][N2]);
    for (size_t i=0; i<N1; ++i)
        for (size_t j=0; j<N2; ++j)
            R_ma[i][j] = R[i][j];

    // main
    auto myres = dipole_radiant_intensity(T_ma, P_ma, p_ma, R_ma, phases, k);

    N1 = myres.shape()[0];
    N2 = myres.shape()[1];
    auto myres_vec = vector<vector<double>>(N1, vector<double>(N2));
    for (size_t i=0; i<N1; ++i)
        for (size_t j=0; j<N2; ++j)
            myres_vec[i][j] = myres[i][j];
    return myres_vec;
}


vector<vector<vector<complex<double>>>>
farfield_dipole_wrapper(vector<vector<vector<double>>> r,
                        vector<vector<double>> P,
                        vector<vector<double>> R,
                        vector<double> phases,
                        double k, double t, bool calc_H) {
    size_t N1 = r.size();
    size_t N2 = r[0].size();
    size_t N3 = r[0][0].size();

    auto r_ma = boost::multi_array<double, 3>(boost::extents[N1][N2][N3]);
    for (size_t i=0; i<N1; ++i)
        for (size_t j=0; j<N2; ++j)
            for (size_t k=0; k<N3; ++k)
                r_ma[i][j][k] = r[i][j][k];

    N1 = P.size();
    N2 = P[0].size();
    auto P_ma = boost::multi_array<double, 2>(boost::extents[N1][N2]);
    for (size_t i=0; i<N1; ++i)
        for (size_t j=0; j<N2; ++j)
            P_ma[i][j] = P[i][j];

    // TODO assert same shape of R and P
    N1 = R.size();
    N2 = R[0].size();
    auto R_ma = boost::multi_array<double, 2>(boost::extents[N1][N2]);
    for (size_t i=0; i<N1; ++i)
        for (size_t j=0; j<N2; ++j)
            R_ma[i][j] = R[i][j];

    // main
    auto myres = dipole_field_ff(r_ma, P_ma, R_ma, phases, k, t, calc_H);

    N1 = myres.shape()[0];
    N2 = myres.shape()[1];
    N3 = myres.shape()[2];
    auto myres_vec = vector<vector<vector<complex<double>>>>(
        N1, vector<vector<complex<double>>>(N2, vector<complex<double>>(N3)));
    for (size_t i=0; i<N1; ++i)
        for (size_t j=0; j<N2; ++j)
            for (size_t k=0; k<N3; ++k)
                myres_vec[i][j][k] = myres[i][j][k];
    return myres_vec;
}


vector<vector<vector<complex<double>>>>
general_dipole_wrapper(vector<vector<vector<double>>> r,
                       vector<vector<double>> P,
                       vector<vector<double>> R,
                       vector<double> phases,
                       double k, double t, bool calc_H) {
    size_t N1 = r.size();
    size_t N2 = r[0].size();
    size_t N3 = r[0][0].size();

    auto r_ma = boost::multi_array<double, 3>(boost::extents[N1][N2][N3]);
    for (size_t i=0; i<N1; ++i)
        for (size_t j=0; j<N2; ++j)
            for (size_t k=0; k<N3; ++k)
                r_ma[i][j][k] = r[i][j][k];

    N1 = P.size();
    N2 = P[0].size();
    auto P_ma = boost::multi_array<double, 2>(boost::extents[N1][N2]);
    for (size_t i=0; i<N1; ++i)
        for (size_t j=0; j<N2; ++j)
            P_ma[i][j] = P[i][j];

    // TODO assert same shape of R and P
    N1 = R.size();
    N2 = R[0].size();
    auto R_ma = boost::multi_array<double, 2>(boost::extents[N1][N2]);
    for (size_t i=0; i<N1; ++i)
        for (size_t j=0; j<N2; ++j)
            R_ma[i][j] = R[i][j];

    // main
    auto myres = dipole_field_general(r_ma, P_ma, R_ma, phases, k, t, calc_H);

    N1 = myres.shape()[0];
    N2 = myres.shape()[1];
    N3 = myres.shape()[2];
    auto myres_vec = vector<vector<vector<complex<double>>>>(
        N1, vector<vector<complex<double>>>(N2, vector<complex<double>>(N3)));
    for (size_t i=0; i<N1; ++i)
        for (size_t j=0; j<N2; ++j)
            for (size_t k=0; k<N3; ++k)
                myres_vec[i][j][k] = myres[i][j][k];
    return myres_vec;
}
