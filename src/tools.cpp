#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // validation checks
  if (estimations.size() == 0 || estimations.size() != ground_truth.size())
  {
    cout << "Invalid estimation/ground vector";
    return rmse;
  }

  //accumulate squared residuals
  for (int i = 0; i < estimations.size(); ++i) {
    VectorXd residue = estimations[i] - ground_truth[i];

    residue = residue.array()*residue.array();
    rmse += residue;
  }

  //calculate the mean
  rmse = rmse / estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

double Tools::NormalizeAngle(double &angle) {
  while (angle > M_PI) angle -= 2.*M_PI;
  while (angle <-M_PI) angle += 2.*M_PI;
  return angle;
}
