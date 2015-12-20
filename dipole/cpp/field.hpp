#include <vector>
#include <complex>

std::vector<std::vector<std::vector<std::complex<double>>>>
farfield_dipole_wrapper(std::vector<std::vector<std::vector<double>>> r,
               std::vector<std::vector<double>> P,
               std::vector<std::vector<double>> R,
               std::vector<double> phases,
               double k, double t);

std::vector<std::vector<std::vector<std::complex<double>>>>
general_dipole_wrapper(std::vector<std::vector<std::vector<double>>> r,
                       std::vector<std::vector<double>> P,
                       std::vector<std::vector<double>> R,
                       std::vector<double> phases,
                       double k, double t, bool calc_H);
