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

void Tools::NormalizeAngle(MatrixXd* x, int i) {
  //angle normalization
  while ((*x)(i)> M_PI) (*x)(1)-=2.*M_PI;
  while ((*x)(i)<-M_PI) (*x)(1)+=2.*M_PI;
}

void Tools::NormalizeAngle(VectorXd* x, int i) {
  //angle normalization
  while ((*x)(i)> M_PI) (*x)(1)-=2.*M_PI;
  while ((*x)(i)<-M_PI) (*x)(1)+=2.*M_PI;
}
