#include <vector>
#include <complex>

std::vector<std::vector<double>>
dipole_radiant_intensity_wrapper(
    std::vector<std::vector<double>> T,
    std::vector<std::vector<double>> P,
    std::vector<std::vector<double>> p,
    std::vector<std::vector<double>> R,
    std::vector<double> phases,
    double k);

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
