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
  rmse << 0,0,0,0;
  
  if (estimations.size() != ground_truth.size()) {
    cout << "Estimations and ground_truth must be of the same size!" << endl;
  }
  
  if (estimations.size() == 0) {
    cout << "There must be at least 1 element in estimations and ground_truth " << endl;
  }
  
  VectorXd diff;
  VectorXd diff2;
  //accumulate squared residuals
  for(int i=0; i < estimations.size(); ++i){
    diff = estimations[i] - ground_truth[i];
    diff2 = (diff.array() * diff.array());
    rmse += diff2;
  }
  
  rmse = rmse.array() / estimations.size();
  
  //calculate the squared root
  rmse = rmse.array().sqrt();
  
  //return the result
  return rmse;
}

float Tools::NormalizeAngle(float phi) {
  //angle normalization
  return atan2( sin(phi), cos(phi) );
}

void Tools::NormalizeAngle(MatrixXd* x, int i) {
  //angle normalization
  int n_col = x->cols();
  for (int j = 0; j < n_col; j++) (*x)(i, j) = Tools::NormalizeAngle((*x)(i, j));
}

void Tools::NormalizeAngle(VectorXd* x, int i) {
  //angle normalization
  (*x)(i) = Tools::NormalizeAngle((*x)(i));
}
